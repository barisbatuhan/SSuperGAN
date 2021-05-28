import copy
from collections import OrderedDict
from tqdm import tqdm
import torch
import torch.nn.functional as F
from functional.losses.kl_loss import kl_loss
from networks.base.base_vae import BaseVAE
from training.base_trainer import BaseTrainer
from utils.structs.metric_recorder import *
from utils.logging_utils import *
from utils import pytorch_util as ptu
from utils.image_utils import *


class IntroVAETrainer(BaseTrainer):
    def __init__(self,
                 model: BaseVAE,
                 model_name: str,
                 test_criterion,
                 train_loader,
                 test_loader,
                 epochs: int,
                 optimizer_e,
                 optimizer_g,
                 scheduler_e=None,
                 scheduler_g=None,
                 quiet: bool = False,
                 grad_clip=None,
                 best_loss_action=None,
                 save_dir=base_dir + 'playground/intro_vae/',
                 checkpoint_every_epoch=False,
                 adversarial_alpha=0.25,
                 ae_beta=0.5,
                 adversarial_margin=110
                 ):
        super().__init__(model,
                         model_name,
                         test_criterion,
                         epochs,
                         save_dir,
                         {"optimizer_e": optimizer_e, "optimizer_g": optimizer_g},
                         {"scheduler_e": scheduler_e, "scheduler_g": scheduler_g},
                         quiet,
                         grad_clip,
                         best_loss_action,
                         checkpoint_every_epoch)
        self.adversarial_alpha = adversarial_alpha
        self.ae_beta = ae_beta
        self.adversarial_margin = adversarial_margin
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer_e = optimizer_e
        self.optimizer_g = optimizer_g
        self.scheduler_e = scheduler_e
        self.scheduler_g = scheduler_g

    def train_epochs(self, starting_epoch=None, losses={}):
        metric_recorder = MetricRecorder(experiment_name=self.model_name,
                                         save_dir=self.save_dir + '/results/')
        # TODO: becareful about best loss here this might override the actual best loss
        #  in case of continuation of training
        best_loss = 99999999

        train_losses = losses.get("train_losses", OrderedDict())
        test_losses = losses.get("test_losses", OrderedDict())

        for epoch in range(self.epochs):
            if starting_epoch is not None and starting_epoch >= epoch:
                continue
            logging.info("epoch start: " + str(epoch))
            train_loss = self.train_vae(epoch)
            if self.test_loader is not None:
                test_loss = self.eval_loss(self.test_loader)
            else:
                test_loss = {"loss": 0, "kl_loss": 0, "reconstruction_loss": 0}

            for k in train_loss.keys():
                if k not in train_losses:
                    train_losses[k] = []
                    test_losses[k] = []
                train_losses[k].extend(train_loss[k])
                test_losses[k].append(test_loss.get(k, 0))
                if k == "loss":
                    current_test_loss = test_loss[k]
                    if current_test_loss < best_loss:
                        best_loss = current_test_loss
                        if self.best_loss_action != None:
                            self.best_loss_action(self.model, best_loss)

            if self.checkpoint_every_epoch:
                self.save_checkpoint(current_loss={
                    "train_losses": train_losses,
                    "test_losses": test_losses
                },
                    current_epoch=epoch)

            metric_recorder.update_metrics(train_losses, test_losses)
            metric_recorder.save_recorder()
        return train_losses, test_losses

    @torch.no_grad()
    def eval_loss(self, data_loader):
        self.model.eval()
        total_losses = OrderedDict()
        for batch in data_loader:

            if type(batch) == list and len(batch) == 2:
                x, y = batch[0].cuda(), batch[1].cuda()
            else:
                x, y = batch.cuda(), None

            z, _, mu_z, mu_x, logstd_z = self.model(x)
            target = x if y is None else y
            out = self.criterion(z, target, mu_z, mu_x, logstd_z)

            for k, v in out.items():
                total_losses[k] = total_losses.get(k, 0) + v.item() * x.shape[0]

        desc = 'Test --> '
        for k in total_losses.keys():
            total_losses[k] /= len(data_loader.dataset)
            desc += f'{k} {total_losses[k]:.4f}'
        if not self.quiet:
            print(desc)
            logging.info(desc)
        return total_losses

    def compute_ae(self, x_r, x):
        ae = (x_r - x) ** 2
        ae = 0.5 * ae.view(ae.shape[0], -1).sum(dim=1).mean()
        return ae

    def train_vae(self, epoch):
        self.model.train()
        if not self.quiet:
            pbar = tqdm(total=len(self.train_loader.dataset))
        losses = OrderedDict()
        for batch in self.train_loader:
            
            if type(batch) == list and len(batch) == 2:
                x, y = batch[0].cuda(), batch[1].cuda()
            else:
                x, y = batch.cuda(), None

            batch_size = x.shape[0]
            target = x if y is None else y
            # =========== Update E ================
            self.optimizer_e.zero_grad()

            z_p = self.model.latent_dist.rsample((batch_size, self.model.latent_dim)).squeeze()
            x_p = self.model.decode(z_p)

            z, _, mu_z, x_r, logstd_z = self.model(x)

            loss_ae = self.compute_ae(x_r, target)  # F.mse_loss(x_r, target)

            z_r, mu_z_r, logstd_z_r = tuple(self.model.encode(x_r.detach()))
            z_pp, mu_z_pp, logstd_z_pp = tuple(self.model.encode(x_p.detach()))

            loss_reg = kl_loss(z, mu_z, torch.exp(logstd_z))
            loss_reg_r = kl_loss(z_r, mu_z_r, torch.exp(logstd_z_r))
            loss_reg_pp = kl_loss(z_pp, mu_z_pp, torch.exp(logstd_z_pp))

            loss_encoder_adverserial = loss_reg + self.adversarial_alpha * (
                    F.relu(self.adversarial_margin - loss_reg_r) +
                    F.relu(self.adversarial_alpha - loss_reg_pp))

            loss_encoder = loss_encoder_adverserial + self.ae_beta * loss_ae

            out = OrderedDict(loss=loss_encoder, loss_ae=loss_ae, loss_kl=loss_reg)
            out['loss'].backward(retain_graph=True)

            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), self.grad_clip)
            self.optimizer_e.step()
            # ========= Update G ==================
            self.optimizer_g.zero_grad()

            _, _, _, x_r, _ = self.model(x)
            x_p = self.model.decode(z_p)

            z_r, mu_z_r, logstd_z_r = tuple(self.model.encode(x_r))
            z_pp, mu_z_pp, logstd_z_pp = tuple(self.model.encode(x_p))

            loss_generator_adversarial = self.adversarial_alpha * (kl_loss(z_r, mu_z_r, torch.exp(logstd_z_r)) + \
                                                                   kl_loss(z_pp, mu_z_pp, torch.exp(logstd_z_pp)))

            loss_ae = self.compute_ae(x_r, target)  # F.mse_loss(x_r, target)
            loss_generator = loss_generator_adversarial + self.ae_beta * loss_ae
            out['loss_generator'] = loss_generator
            out['loss_generator'].backward()

            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.decoder.parameters(), self.grad_clip)
            self.optimizer_g.step()
            # ===========================

            desc = f'Epoch {epoch}'
            for k, v in out.items():
                if k not in losses:
                    losses[k] = []
                losses[k].append(v.item())
                avg = np.mean(losses[k][-50:])
                desc += f', {k} {avg:.4f}'

            if not self.quiet:
                pbar.set_description(desc)
                pbar.update(x.shape[0])

        self.scheduler_e.step()
        self.scheduler_g.step()
        self.model.save_samples(10, self.save_dir + '/results/' + f'epoch{epoch}_samples.png')
        if not self.quiet:
            pbar.close()
        return losses
