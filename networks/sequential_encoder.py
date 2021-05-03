import torch
import torch.nn as nn
import torchvision
import torchvision.models as models

class SequentialEncoder(nn.Module):
    
    def __init__(self, args=None, pretrained_cnn=None):
        super(SequentialEncoder, self).__init__()
        
        if dim_args is None:
            args = {
                "lstm_hidden": hidden_size, 
                "embed": -1, 
                "cnn_embed": 1000, 
                "fc_hiddens": [],
                "lstm_dropout": 0.2,
                "fc_dropout": 0.2,
                "num_lstm_layers": 1
            }
        
        self.embed_size = args["embed"]
        self.hidden_size = args["lstm_hidden"]
        
        # CNN based panel iamge embedder method
        if pretrained_cnn is None:
            self.backbone =  models.resnet50(pretrained=True)  
        else:
            self.backbone = pretrained_cnn                           
        
        # LSTM, sequential processing unit
        self.lstm = nn.LSTM(
            args["cnn_embed"], self.hidden_size, 
            dropout=args["lstm_dropout"], num_layers=args["num_lstm_layers"]
        )
        self.register_buffer('h0', torch.zeros(args["num_lstm_layers"], args["lstm_hidden"]))
        self.register_buffer('c0', torch.zeros(args["num_lstm_layers"], args["lstm_hidden"]))
        
        # Additional FC layers to further process the LSTM output
        if self.embed_size > 0:
            fc_hidden_sizes = [self.hidden_size, *args["fc_hiddens"], self.embed_size]
            fc_layers = []
            for i in range(len(fc_hidden_sizes) - 1):
                fc_layers.append(nn.Dropout(args["fc_dropout"]))
                fc_layers.append(nn.Linear(fc_hidden_sizes[i], fc_hidden_sizes[i+1]))
            
            self.fc_projector = nn.Sequential(*fc_layers)
        
        else:
            self.fc_projector = None
            
    
    def forward(self, x):
        B, S, C, H, W = x.shape
        
        # Retrieved the embeddings for each of the panels
        outs = []
        for s in range(S):
            outs.append(self.backbone(x[:,s,:,:,:]).unsqueeze(1))
        outs = torch.cat(outs, dim=1)
        
        # Embedding outputs are passed to the lstm
        outs, (h, c) = self.lstm(
            outs, 
            self.h0.unsqueeze(1).repeat(1, S, 1),
            self.c0.unsqueeze(1).repeat(1, S, 1)
        )
        
        # Since there are S many outputs, final output is required only
        outs = outs[:,-1,:] 
        
        # Additional FC layers
        if self.fc_projector is not None:
            outs = self.fc_projector(outs)