import os

import torch
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import optim
from torch.utils.data import DataLoader

from configs.base_config import base_dir
from data.augment import get_PIL_image
from data.datasets.golden_panels import GoldenPanelsDataset
from functional.losses.elbo import elbo
from networks.pl_ssuper_model import SSuperModel
from functional.metrics.psnr import PSNR
import pytorch_lightning as pl

from utils.config_utils import read_config, Config
from utils.datetime_utils import get_dt_string


class SSuperVAE(SSuperModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process_batch(self, batch):
        if type(batch) == list and len(batch) == 2:
            x, y = batch[0], batch[1]
        elif type(batch) == list and len(batch) == 3:
            x, y, mask = batch[0], batch[1], batch[2]
        elif type(batch) == list and len(batch) == 4:
            x, y, mask, coords = batch[0], batch[1], batch[2], batch[3]
        else:
            x, y = batch, batch
        mu_z, logstd_z = self(x, f='seq_encode')
        z = torch.distributions.Normal(mu_z, logstd_z.exp()).rsample()
        mu_x = self(z, f='generate')
        return z, y, mu_z, mu_x, logstd_z

    def _calculate_loss(self, batch, mode):
        z, y, mu_z, mu_x, logstd_z = self.process_batch(batch)

        psnr = PSNR.__call__(mu_x, y, fit_range=True)
        l1 = torch.abs(y - mu_x).mean()
        out = elbo(z, y, mu_z, mu_x, logstd_z)

        self.log_dict(out, prog_bar=True)
        self.log("%s_psnr" % mode, psnr, on_step=False, on_epoch=True)
        self.log("%s_l1" % mode, l1, on_step=False, on_epoch=True)
        return out['loss']

    # TODO: Add sampling at the end of epoch
    #  grid = torchvision.utils.make_grid(sample_imgs)
    #  self.logger.experiment.add_image('generated_images', grid, 0)
    #  additional source: https://pytorch-lightning-bolts.readthedocs.io/en/latest/vision_callbacks.html#tensorboard-image-generator

    def generate_recons(self, batch):
        epoch = self.trainer.current_epoch
        z, y, mu_z, mu_x, logstd_z = self.process_batch(batch)
        h, w = y.shape[2:]
        wsize, hsize = 2, y.shape[0]
        w = (w + 100) * wsize
        h = (h + 100) * hsize
        px = 1 / plt.rcParams['figure.dpi']
        f, ax = plt.subplots(hsize, wsize)
        f.set_size_inches(w * px, h * px)

        for bs in range(y.shape[0]):
            ax[bs, 0].imshow(get_PIL_image(y[bs, :, :, :]))
            ax[bs, 0].axis('off')
            ax[bs, 1].imshow(get_PIL_image(mu_x[bs, :, :, :].clamp(-1, 1)))
            ax[bs, 1].axis('off')

        ax[0, 0].title.set_text("Original")
        ax[0, 1].title.set_text("Recon")

        saved_fig = plt.savefig(self.save_dir + 'results/' + self.model_name + f'_epoch{epoch}_recons.png')
        self.logger.experiment.add_figure(tag='epoch{epoch}_recons', figure=saved_fig)

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="val")
        # if batch_idx == 0:
        #    self.generate_recons(batch)

    def test_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="test")
        # if batch_idx == 0:
        #     self.generate_recons(batch)

    # https://pytorch-lightning.readthedocs.io/en/latest/common/optimizers.html
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),
                               lr=self.hparams.lr,
                               betas=(self.hparams.beta_1, self.hparams.beta_2),
                               weight_decay=self.hparams.weight_decay)

        scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                lambda epoch: (
                                                                      self.hparams.train_epochs - epoch) / self.hparams.train_epochs,
                                                last_epoch=-1)

        return [optimizer], [scheduler]

    def optimizer_step(self, *args, **kwargs):
        super(SSuperModel, self).optimizer_step(*args, **kwargs)
        # hacky way of exploting optimizer step - self.lr_scheduler.step()  # Step per iteration


def train_ssupervae(train_loader,
                    val_loader,
                    checkpoint_path,
                    max_epochs,
                    experiment_name="SSuperVAE" + get_dt_string(),
                    **kwargs):
    root_dir = os.path.join(checkpoint_path, experiment_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_psnr")],
                         gpus=torch.cuda.device_count(),
                         max_epochs=max_epochs,
                         gradient_clip_val=2,
                         progress_bar_refresh_rate=1)
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(checkpoint_path, experiment_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = SSuperVAE.load_from_checkpoint(pretrained_filename)
    else:
        model = SSuperVAE(**kwargs)
        trainer.fit(model, train_loader, val_loader)
        model = SSuperVAE.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation and test set
    train_result = trainer.test(model, test_dataloaders=train_loader, verbose=False)
    val_result = trainer.test(model, test_dataloaders=val_loader, verbose=False)
    result = {"val_psnr": val_result[0]["test_psnr"],
              "train_psnr": train_result[0]["test_psnr"]}

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    return model, result


if __name__ == '__main__':
    config = read_config(Config.SSUPERVAE)

    # Data Loading

    golden_age_config = read_config(Config.GOLDEN_AGE)
    # cont_model = "playground/ssuper_global_dcgan/ckpts/lstm_ssuper_global_dcgan_model-checkpoint-epoch99.pth"
    cont_model = None

    tr_data = GoldenPanelsDataset(
        golden_age_config.panel_path,
        golden_age_config.sequence_path,
        config.panel_size,
        config.img_size,
        augment=False,
        mask_val=golden_age_config.mask_val,
        mask_all=golden_age_config.mask_all,
        return_mask=golden_age_config.return_mask,
        return_mask_coordinates=golden_age_config.return_mask_coordinates,
        train_test_ratio=golden_age_config.train_test_ratio,
        train_mode=True,
        limit_size=-1)

    val_data = GoldenPanelsDataset(
        golden_age_config.panel_path,
        golden_age_config.sequence_path,
        config.panel_size,
        config.img_size,
        augment=False,
        mask_val=golden_age_config.mask_val,
        mask_all=golden_age_config.mask_all,
        return_mask=golden_age_config.return_mask,
        return_mask_coordinates=golden_age_config.return_mask_coordinates,
        train_test_ratio=golden_age_config.train_test_ratio,
        train_mode=False,
        limit_size=-1)

    tr_data_loader = DataLoader(tr_data, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_data_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, num_workers=4)

    experiment_name = "SSuperVAE" + get_dt_string()

    model, result = train_ssupervae(
        train_loader=tr_data_loader,
        val_loader=val_data_loader,
        experiment_name=experiment_name,
        checkpoint_path=base_dir + "playground/ssupervae/",
        max_epochs=config.train_epochs,
        
        use_seq_enc=True,
        enc_choice=None,
        gen_choice="vae",
        local_disc_choice=None,
        global_disc_choice=None,

        lr=config.lr,
        beta_1=config.beta_1,
        beta_2=config.beta_2,
        weight_decay=config.weight_decay,
        train_epochs=config.train_epochs,

        save_dir=base_dir + "playground/ssupervae/",
        model_name=experiment_name,
        backbone=config.backbone,
        latent_dim=config.latent_dim,
        embed_dim=config.embed_dim,
        use_lstm=config.use_lstm,
        seq_size=config.seq_size,
        enc_channels=config.enc_channels,
        img_size=config.img_size,
        lstm_hidden=config.lstm_hidden,
        lstm_dropout=config.lstm_dropout,
        fc_hidden_dims=config.fc_hidden_dims,
        fc_dropout=config.fc_dropout,
        num_lstm_layers=config.num_lstm_layers,
        masked_first=config.masked_first)
