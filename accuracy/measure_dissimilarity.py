import torchvision
import torch
from utils import pytorch_util as ptu
from utils.image_utils import imshow
import torch.nn.functional as F


@torch.no_grad()
def compare_image_pairs(dataiterator, net, max_display=None):
    data_length = len(dataiterator)
    for i in range(data_length):
        if max_display != None and i > max_display:
            break
        x0, x1, label = next(dataiterator)
        concatenated = torch.cat((x0, x1), 0)
        output1, output2, _ = net((x0, x1, label))
        euclidean_distance = F.pairwise_distance(output1, output2)
        imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity {:.2f}'.format(euclidean_distance.item()))
        print('original label: ' + str(ptu.get_numpy(label)))

