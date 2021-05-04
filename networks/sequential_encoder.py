import torch
import torch.nn as nn
import torchvision
import torchvision.models as models


class SequentialEncoder(nn.Module):

    def __init__(self, args=None, pretrained_cnn=None):
        super(SequentialEncoder, self).__init__()

        defaults = {
            "lstm_hidden": 256, # hidden size of LSTM module
            "embed": 512, # last size for mean and std outputs
            "cnn_embed": 2048, # the output dim retrieved from CNN embedding module
            "fc_hiddens": [], # sizes of FC layers after LSTM output, if there are any
            "lstm_dropout": 0, # set to 0 if num_lstm_layers is 1, otherwise set to [0, 0.5]
            "fc_dropout": 0.2, # dropout ratio of FC layers if there are any
            "num_lstm_layers": 1 # number of stacked LSTM layers
        }

        if args is not None:
            for k in defaults.keys():
                if k not in args.keys():
                    args[k] = defaults[k]
        else:
            args = defaults

        self.embed_size = args["embed"]
        self.hidden_size = args["lstm_hidden"]
        self.num_lstm_layers = args["num_lstm_layers"]
        
        # CNN based panel image embedder method
        if pretrained_cnn is None:
            # all layers except the lst FC
            self.backbone =  torch.nn.Sequential(
                *(list(models.resnet50(pretrained=True).children())[:-1]))  
        else:
            self.backbone = pretrained_cnn

            # LSTM, sequential processing unit
        self.lstm = nn.LSTM(
            args["cnn_embed"], self.hidden_size,
            dropout=args["lstm_dropout"], num_layers=args["num_lstm_layers"]
        )

        # Additional FC layers to further process the LSTM output
        if len(args["fc_hiddens"]) > 0:
            fc_hidden_sizes = [self.hidden_size, *args["fc_hiddens"]]
            fc_layers = []
            for i in range(len(fc_hidden_sizes) - 1):
                fc_layers.append(nn.Dropout(args["fc_dropout"]))
                fc_layers.append(nn.Linear(fc_hidden_sizes[i], fc_hidden_sizes[i + 1]))

            self.fc_projector = nn.Sequential(*fc_layers)

        else:
            self.fc_projector = None

        # Mean and Variance Calculator
        last_size = self.hidden_size if len(args["fc_hiddens"]) < 1 else args["fc_hiddens"][-1]
        self.fc_mean = nn.Linear(last_size, self.embed_size)
        self.fc_var = nn.Linear(last_size, self.embed_size)

    def forward(self, x):
        B, S, C, H, W = x.shape
        device = x.get_device()

        # Retrieved the embeddings for each of the panels
        outs = []
        for s in range(S):
            outs.append(self.backbone(x[:, s, :, :, :]).unsqueeze(1))
        outs = torch.cat(outs, dim=1)

        # Embedding outputs are passed to the lstm
        outs, _ = self.lstm(
            outs,
            (
                torch.zeros(self.num_lstm_layers, S, self.hidden_size).to(device), # h0
                torch.zeros(self.num_lstm_layers, S, self.hidden_size).to(device)  # c0
            ) 
        )
        outs = outs[:, -1, :]

        # Additional FC layers
        if self.fc_projector is not None:
            outs = self.fc_projector(outs)

        # Extract mean and variance
        mu = self.fc_mean(outs)
        log_var = self.fc_var(outs)
        std = torch.exp(log_var / 2)

        return mu, std
