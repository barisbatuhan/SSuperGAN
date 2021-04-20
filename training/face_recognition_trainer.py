# TODO: implement
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch.optim as optim


# TODO: convert this to be a CLASS and conform to ABSTRACT CLASS as a protocol
# add optimizer as a part of this as well

# TODO: write your own decorator to insert model.eval as well!!!

@torch.no_grad()
def eval_loss(model,
              data_loader,
              quiet: bool = False):
    model.eval()
    total_losses = OrderedDict()
    for x in data_loader:
        out = model.loss(x)
        for k, v in out.items():
            total_losses[k] = total_losses.get(k, 0) + v.item() * x.shape[0]

    desc = 'Test '
    for k in total_losses.keys():
        total_losses[k] /= len(data_loader.dataset)
        desc += f', {k} {total_losses[k]:.4f}'
    if not quiet:
        print(desc)
    return total_losses


def train_epochs(model,
                 optimizer,
                 train_loader,
                 test_loader,
                 train_args,
                 quiet=False):
    epochs = train_args['epochs']
    grad_clip = train_args.get('grad_clip', None)

    train_losses, test_losses = OrderedDict(), OrderedDict()
    for epoch in range(epochs):
        model.train()
        train_loss = train_recognition(model, train_loader, optimizer, epoch, quiet, grad_clip)
        test_loss = eval_loss(model, test_loader, quiet)

        for k in train_loss.keys():
            if k not in train_losses:
                train_losses[k] = []
                test_losses[k] = []
            train_losses[k].extend(train_loss[k])
            test_losses[k].append(test_loss[k])
    return train_losses, test_losses


def train_recognition(model: nn.Module,
                      train_loader: DataLoader,
                      optimizer: optim.Optimizer,
                      epoch: int,
                      quiet: bool = False,
                      grad_clip=None):
    model.train()
    if not quiet:
        pbar = tqdm(total=len(train_loader.dataset))
    losses = OrderedDict()
    for batch in train_loader:
        out = model.loss(batch)
        optimizer.zero_grad()
        out['loss'].backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        desc = f'Epoch {epoch}'
        for k, v in out.items():
            if k not in losses:
                losses[k] = []
            losses[k].append(v.item())
            avg_loss = np.mean(losses[k][-50:])
            desc += f', {k} {avg_loss:.4f}'

        if not quiet:
            pbar.set_description(desc)
            if batch is list:
                pbar.update(batch[0].shape[0])
            else:
                pbar.update(batch.shape[0])
    if not quiet:
        pbar.close()
    return losses
