import os
from os.path import join, dirname, exists
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision.utils import save_image

from data.augment import get_PIL_image
from networks.panel_encoder.cnn_embedder import CNNEmbedder


def savefig(fname, show_figure=True):
    if not exists(dirname(fname)):
        os.makedirs(dirname(fname))
    plt.tight_layout()
    plt.savefig(fname)
    if show_figure:
        plt.show()


def save_training_plot(train_losses, test_losses, title, fname):
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, train_losses, label='train loss')
    plt.plot(x_test, test_losses, label='test loss')
    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('NLL')
    savefig(fname)

# TODO: impl save accuracy plot


def save_scatter_2d(data, title, fname):
    plt.figure()
    plt.title(title)
    plt.scatter(data[:, 0], data[:, 1])
    savefig(fname)


def save_distribution_1d(data, distribution, title, fname):
    d = len(distribution)

    plt.figure()
    plt.hist(data, bins=np.arange(d) - 0.5, label='train data', density=True)

    x = np.linspace(-0.5, d - 0.5, 1000)
    y = distribution.repeat(1000 // d)
    plt.plot(x, y, label='learned distribution')

    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('Probability')
    plt.legend()
    savefig(fname)


def save_distribution_2d(true_dist, learned_dist, fname):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.imshow(true_dist)
    ax1.set_title('True Distribution')
    ax1.axis('off')
    ax2.imshow(learned_dist)
    ax2.set_title('Learned Distribution')
    ax2.axis('off')
    savefig(fname)


def show_samples(samples, fname=None, nrow=10, title='Samples'):
    samples = (torch.FloatTensor(samples) / 255).permute(0, 3, 1, 2)
    grid_img = make_grid(samples, nrow=nrow)
    plt.figure()
    plt.title(title)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')

    if fname is not None:
        savefig(fname)
    else:
        plt.show()
        

def plot_panels_and_faces(panels_tensor, face_tensor, recon_face_tensor):
    
    y_recon = get_PIL_image(recon_face_tensor[0,:,:,:])
    
    y = get_PIL_image(face_tensor[0,:,:,:])
    
    panels = []
    for i in range(panels_tensor.shape[1]):
        panels.append(get_PIL_image(panels_tensor[0,i,:,:,:]))
    
    w, h = panels[0].size
    wsize, hsize = panels_tensor.shape[1] + 2, 1
    w = (w + 100) * wsize
    h = (h + 50) * hsize
    
    px = 1/plt.rcParams['figure.dpi']
    f, ax = plt.subplots(hsize, wsize)
    f.set_size_inches(w*px, h*px)
    
    ax[0].imshow(y)
    ax[0].title.set_text("Original")
    
    ax[1].imshow(y_recon)
    ax[1].title.set_text("Recon")
    
    for i in range(len(panels)):
        ax[i+2].imshow(panels[i])
        ax[i+2].title.set_text("Panel" + str(i+1))
    
    plt.show()
    
def draw_saliency(model, imgs, y):
    h, w = imgs.shape[-2:]
    px = 1/plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(3, 5)
    fig.set_size_inches(5*w*px, 2.5*h*px)
    
    imgs.requires_grad_()
    _, _, _, outs, _ = model(imgs.cuda())
    out = outs.sum()
    out.backward()
    
    g_imgs = imgs.grad
    
    for i in range(3):
        saliency, _ = torch.max(g_imgs[0,i,:,:,:].data.abs(), dim=0) 
        saliency = saliency.reshape(h, w)
        init_img = get_PIL_image(imgs[0,i,:,:,:])
        # Visualize the image and the saliency map
        ax[0, i].imshow(init_img)
        ax[0, i].axis('off')
        ax[0, i].title.set_text("Panel" + str(i+1))
        ax[1, i].imshow(saliency.cpu(), cmap='hot')
        ax[1, i].axis('off')
        ax[2, i].imshow(init_img)
        ax[2, i].imshow(saliency.cpu(), cmap='hot', alpha=0.7)
        ax[2, i].axis('off')
    
    target = get_PIL_image(y[0,:,:,:])
    ax[0, 3].imshow(target)
    ax[0, 3].title.set_text("Original Face")
    ax[1, 3].imshow(target)
    ax[2, 3].imshow(target)
    
    y_recon = get_PIL_image(outs[0,:,:,:])
    ax[0, 4].imshow(y_recon)
    ax[0, 4].title.set_text("Reconstructed")
    ax[1, 4].imshow(y_recon)
    ax[2, 4].imshow(y_recon)
    
    plt.tight_layout()
    plt.show() 

def draw_backbone_saliency(model, img, idx):
    h, w = img.shape[-2:]
    px = 1/plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(3*w*px, 1.2*h*px)
    
    img.requires_grad_()
    outs = model(img.cuda())
    out = outs.sum()
    out.backward()
    
    g_img = img.grad
    
    saliency, _ = torch.max(g_img[0,idx,:,:,:].data.abs(), dim=0) 
    saliency = saliency.reshape(h, w)
    
    # Visualize the image and the saliency map
    init_img = get_PIL_image(img[0,idx,:,:,:])
    ax[0].imshow(init_img)
    ax[0].axis('off')
    ax[1].imshow(saliency.cpu(), cmap='hot')
    ax[1].axis('off')
    ax[2].imshow(init_img)
    ax[2].imshow(saliency.cpu(), cmap='hot', alpha=0.7)
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.show() 


def draw_cnn_embedder_saliency(model: CNNEmbedder, img, idx):
    c, h, w = img.shape[-3:]
    px = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(3 * w * px, 1.2 * h * px)

    img.requires_grad_()
    outs = model.model.extract_features(img.cuda().reshape(-1, c, h, w))
    out = outs.sum()
    out.backward()

    g_img = img.grad

    saliency, _ = torch.max(g_img[0, idx, :, :, :].data.abs(), dim=0)
    saliency = saliency.reshape(h, w)

    # Visualize the image and the saliency map
    init_img = get_PIL_image(img[0, idx, :, :, :])
    ax[0].imshow(init_img)
    ax[0].axis('off')
    ax[1].imshow(saliency.cpu(), cmap='hot')
    ax[1].axis('off')
    ax[2].imshow(init_img)
    ax[2].imshow(saliency.cpu(), cmap='hot', alpha=0.7)
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()