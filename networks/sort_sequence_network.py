import copy
import enum
import itertools
import random
from functools import reduce
import torch
from torchvision.models import *
import torch.nn as nn
from networks.mlp import MLP
from networks.panel_encoder.cnn_embedder import CNNEmbedder


class SortSequenceNetwork(nn.Module):
    def __init__(self,
                 embedder: CNNEmbedder,
                 pairwise_extraction_in_size: int,
                 num_elements_in_sequence: int):
        super(SortSequenceNetwork, self).__init__()
        self.pairwise_extraction_in_size = pairwise_extraction_in_size
        self.num_elements_in_sequence = num_elements_in_sequence
        self.num_pairs = len(list(itertools.combinations(list(range(0, num_elements_in_sequence)), 2)))
        self.order_choices = list(itertools.permutations(list(range(0, num_elements_in_sequence))))
        self.embed_dim = embedder.embed_dim

        self.feature_extractor = embedder
        self.pairwise_feature_extraction = nn.Linear(in_features=pairwise_extraction_in_size,
                                                     out_features=self.embed_dim)

        # impl order prediction layer
        self.order_prediction_layer = nn.Linear(in_features=self.num_pairs * self.embed_dim,
                                                out_features=len(self.order_choices))

    def forward(self, x):
        return self.shuffle_forward(x)
    
    def shuffle_forward(self, x):
        B, S, C, H, W = x.shape
        labels = []
        for b in range(B):
            original_seq = list(range(0, S))
            shuffled_seq = copy.deepcopy(original_seq)
            random.shuffle(shuffled_seq)
            shuffled_x = copy.deepcopy(x[b])
            labels.append(self.order_choices.index(tuple(shuffled_seq)))
            for count, i in enumerate(shuffled_seq):
                shuffled_x[count] = x[b, i]
            x[b] = shuffled_x
        labels = torch.Tensor(labels).long().cuda()
        output = self.plain_forward(x)
        return output, labels

    def shuffle_forward_loss(self, x):
        output, labels = self.shuffle_forward(x)
        loss = nn.CrossEntropyLoss()
        loss_val = loss(output, labels)
        return loss_val
    
    def plain_forward(self, x):
        B, S, C, H, W = x.shape
        embeddings = self.feature_extractor.model.extract_features(x.reshape(-1, C, H, W))
        embeddings = embeddings.reshape(B, S, -1)
        seq_elements_individually = torch.chunk(embeddings, self.num_elements_in_sequence, dim=1)
        element_indexes = list(range(0, self.num_elements_in_sequence))
        pair_features = []
        for pair in itertools.combinations(element_indexes, 2):
            first_el = seq_elements_individually[pair[0]]
            second_el = seq_elements_individually[pair[1]]
            combined_pair = torch.cat([first_el, second_el], dim=1)
            # I believe this can be optimized
            pair_feature = self.pairwise_feature_extraction(combined_pair.view(B, -1)).view(B, 1, self.embed_dim)
            pair_features.append(pair_feature)
        order_features = torch.cat(pair_features, dim=1)
        result = self.order_prediction_layer(order_features.view(B, -1))
        return result


if __name__ == '__main__':
    embed_dim = 256
    cnn_embedder = CNNEmbedder("efficientnet-b5", embed_dim=embed_dim)
    net = SortSequenceNetwork(cnn_embedder,
                              num_elements_in_sequence=3,
                              pairwise_extraction_in_size=8192 * 2).cuda()
    B = 4
    S = 3
    C = 3
    H = 64
    W = 64
    test_batch = torch.randn(B, S, C, H, W).cuda()
    net.shuffle_forward_loss(test_batch)
