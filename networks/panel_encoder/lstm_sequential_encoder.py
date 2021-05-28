import torch
import torch.nn as nn
import torchvision
import torchvision.models as models

import numpy as np

from networks.panel_encoder.cnn_embedder import CNNEmbedder

class LSTMSequentialEncoder(nn.Module):

    def __init__(self, 
                 backbone,
                 latent_dim=256,
                 embed_dim=256,
                 lstm_hidden=256,
                 lstm_dropout=0,
                 lstm_bidirectional=False,
                 fc_hidden_dims=[],
                 fc_dropout=0,
                 num_lstm_layers=1,
                 masked_first=True):
        
        super(LSTMSequentialEncoder, self).__init__()
        
        self.masked_first = masked_first
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden = lstm_hidden if not lstm_bidirectional else lstm_hidden//2
        self.lstm_bidirectional = lstm_bidirectional
        self.embedder = CNNEmbedder(backbone, embed_dim=embed_dim)
        
        # LSTM, sequential processing unit
        
        self.lstm = nn.LSTM(embed_dim, self.lstm_hidden,
                            bidirectional=lstm_bidirectional,
                            dropout=lstm_dropout, 
                            num_layers=num_lstm_layers)

        # Additional FC layers to further process the LSTM output
        if len(fc_hidden_dims) > 0:
            fc_hidden_sizes = [lstm_hidden, *fc_hidden_dims]
            fc_layers = []
            for i in range(len(fc_hidden_sizes) - 1):
                fc_layers.append(nn.Dropout(fc_dropout))
                fc_layers.append(nn.Linear(fc_hidden_sizes[i], fc_hidden_sizes[i + 1]))

            self.fc_projector = nn.Sequential(*fc_layers)

        else:
            self.fc_projector = None

        # Mean and Variance Calculator
        last_size = lstm_hidden if len(fc_hidden_dims) < 1 else fc_hidden_dims[-1]
        self.fc_mean = nn.Linear(last_size, latent_dim)
        self.fc_lgstd = nn.Linear(last_size, latent_dim)

        
    def forward(self, x):
        B, S, C, H, W = x.shape

        # Retrieved the embeddings for each of the panels
        outs = self.embedder(x).reshape(B, S, -1).permute(1, 0, 2)
        
        if self.masked_first: # brings the embedding of the last panel to the first
            outs = outs[[-1, *np.arange(S-1)],:,:]

        # Embedding outputs are passed to the lstm
        first_h_dim = self.num_lstm_layers if not self.lstm_bidirectional else self.num_lstm_layers * 2
        
        _, ( outs, _ ) = self.lstm(
            outs,
            (
                torch.zeros(first_h_dim, B, self.lstm_hidden).cuda(), # h0
                torch.zeros(first_h_dim, B, self.lstm_hidden).cuda()  # c0
            ) 
        )
        
        num_directions = 2 if self.lstm_bidirectional else 1
        outs = outs.view(self.num_lstm_layers, num_directions, B, -1)
        outs = outs.permute(2, 0, 1, 3)[:,-1,:,:].reshape(B, -1)

        # Additional FC layers
        if self.fc_projector is not None:
            outs = self.fc_projector(outs)

        # Extract mean and variance
        mu = self.fc_mean(outs)
        log_std = self.fc_lgstd(outs)

        return mu, log_std
