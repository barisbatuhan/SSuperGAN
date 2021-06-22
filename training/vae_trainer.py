from tqdm import tqdm
import torch

from data.augment import get_PIL_image
from training.base_trainer import BaseTrainer
from utils.structs.metric_recorder import *
from utils.logging_utils import *
from utils.image_utils import *
from functional.metrics.psnr import PSNR


class VAETrainer(BaseTrainer):
    def __init__(self,
                 model,
                 model_name: str,
                 criterion,
                 train_loader,
                 test_loader,
                 epochs: int,
                 optimizer,
                 scheduler=None,
                 quiet: bool = False,
                 grad_clip=None,
                 parallel=False,
                 best_loss_action=None,
                 save_dir=base_dir + 'playground/vae/',
                 checkpoint_every_epoch=False):
        super().__init__(model,
                         model_name,
                         criterion,
                         epochs,
                         save_dir,
                         {"optimizer": optimizer},
                         {"scheduler": scheduler},
                         quiet,
                         grad_clip,
                         best_loss_action,
                         checkpoint_every_epoch)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.parallel = parallel

    def train_epochs(self, starting_epoch=None, losses={}):
        metric_recorder = MetricRecorder(experiment_name=self.model_name,
                                         save_dir=self.save_dir + '/results/')
        best_loss = float('inf')

        train_losses = losses.get("train_losses", OrderedDict())
        test_losses = losses.get("test_losses", OrderedDict())

        for epoch in range(self.epochs):
            if starting_epoch is not None and starting_epoch >= epoch:
                continue
            logging.info("epoch start: " + str(epoch))
            train_loss = self.train_model(epoch)
            if self.test_loader is not None:
                test_loss = self.eval_model(epoch)
            else:
                test_loss = {"loss": 0, "kl_loss": 0, "reconstruction_loss": 0}

            for k in train_loss.keys():
                if k not in train_losses:
                    train_losses[k] = []
                    test_losses[k] = []
                train_losses[k].extend(train_loss[k])
                test_losses[k].append(test_loss[k])
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
    def eval_model(self, epoch):
        self.model.eval()
        psnrs, l1s, iter_cnt, recon_print, bs = 0, 0, 0, False, self.test_loader.batch_size
        total_losses = OrderedDict()
        for batch in self.test_loader:

            if type(batch) == list and len(batch) == 2:
                x, y = batch[0].cuda(), batch[1].cuda()
            elif type(batch) == list and len(batch) == 3:
                x, y, mask = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
            elif type(batch) == list and len(batch) == 4:
                x, y, mask, coords = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()
            else:
                x, y = batch.cuda(), batch.cuda()

            mu_z, logstd_z = self.model(x, f='seq_encode')
            z = torch.distributions.Normal(mu_z, logstd_z.exp()).rsample()
            mu_x = self.model(z, f='generate')

            target = x if y is None else y
            out = self.criterion(z, target, mu_z, mu_x, logstd_z)

            for k, v in out.items():
                total_losses[k] = total_losses.get(k, 0) + v.item() * x.shape[0]

            # TODO: THIS CAN BE MODULIZED
            psnrs += PSNR.__call__(mu_x, y, fit_range=True)
            l1s += torch.abs(y - mu_x).mean()
            iter_cnt += 1

            if recon_print == False:
                recon_print = True
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

                plt.savefig(self.save_dir + 'results/' + f'_epoch{epoch}_recons.png')

        print("\n\n-- Epoch:", epoch,
              " --> PSNR:", psnrs.item() / iter_cnt,
              " & L1 Loss:", l1s.item() / iter_cnt,
              "\n")

        desc = 'Test --> '
        for k in total_losses.keys():
            total_losses[k] /= len(self.test_loader.dataset)
            desc += f'{k} {total_losses[k]:.4f}'
        if not self.quiet:
            print(desc)
            logging.info(desc)
        return total_losses

    def train_model(self, epoch):
        self.model.train()
        if not self.quiet:
            pbar = tqdm(total=len(self.train_loader.dataset))
        losses = OrderedDict()
        for batch in self.train_loader:

            if type(batch) == list and len(batch) == 2:
                x, y = batch[0].cuda(), batch[1].cuda()
            elif type(batch) == list and len(batch) == 3:
                x, y, mask = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
            elif type(batch) == list and len(batch) == 4:
                x, y, mask, coords = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()
            else:
                x, y = batch.cuda(), batch.cuda()

            self.optimizer.zero_grad()

            mu_z, logstd_z = self.model(x, f='seq_encode')
            z = torch.distributions.Normal(mu_z, logstd_z.exp()).rsample()
            mu_x = self.model(z, f='generate')

            target = x if y is None else y

            out = self.criterion(z, target, mu_z, mu_x, logstd_z)
            out['loss'].backward()

            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

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

        self.scheduler.step()

        if self.parallel:
            self.model.module.save_samples(100, self.save_dir + 'results/' + f'_epoch{epoch}_samples.png')
        else:
            self.model.save_samples(100, self.save_dir + 'results/' + f'_epoch{epoch}_samples.png')

        if not self.quiet:
            pbar.close()
        return losses
