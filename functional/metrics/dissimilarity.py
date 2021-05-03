import torchvision
import torch
from utils import pytorch_util as ptu
from utils.image_utils import imshow
import torch.nn.functional as F
import numpy as np


@torch.no_grad()
def compare_image_pairs(dataiterator, net, max_display=None):
    net.eval()
    data_length = len(dataiterator)
    for i in range(data_length):
        if max_display != None and i > max_display:
            break
        x0, x1, label = next(dataiterator)
        concatenated = torch.cat((x0, x1), 0)
        output1, output2, _ = net((x0, x1, label))
        euclidean_distance = F.pairwise_distance(output1, output2)
        imshow(torchvision.utils.make_grid(concatenated), 'Dissimilarity {:.2f}'.format(euclidean_distance.item()))
        print('original label: ' + str(ptu.get_numpy(label)))


@torch.no_grad()
def compute_mean_accuracy(dataiterator, model, acc_threshold=0.5):
    model.eval()
    all_acc = []
    total_element_count = 0
    print("Started to compute accuracy for model")
    for batch in dataiterator:
        out = model(batch)
        output1 = out[0]
        output2 = out[1]
        label = out[2]
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        pred = ptu.get_numpy(euclidean_distance).ravel() < acc_threshold
        label = (ptu.get_numpy(label) == 0).ravel()
        curr_acc = sum(pred == label)
        all_acc.append(curr_acc)
        total_element_count += len(euclidean_distance)
        print("Batch mean acc: " + str(curr_acc) + " / " + str(len(euclidean_distance)))
    return sum(all_acc) / total_element_count
