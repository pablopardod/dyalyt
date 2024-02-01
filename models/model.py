from torch.nn import functional as F
from torch import nn
from torch import Tensor
import numpy as np
from timm.layers.norm_act import BatchNormAct2d

from models.encoder import SqueezeformerEncoder


class ClassDecoder(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        
        # Fully connected
        self.fc = nn.Linear(input_channels, num_classes)
        self.ln = nn.LayerNorm(input_channels)
        
    def forward(self, x):
        # Flatten input (batch_size, 384*144)
        x = x.view(x.size(0), -1)  
        x = self.ln(x)
        x = self.fc(x)
        
        return x

class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        
        self.conv = nn.Conv2d(3, cfg.encoder_dim, (1, cfg.n_landmarks), stride=1) #(1, 13)
        self.bn_conv = BatchNormAct2d(cfg.encoder_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True,act_layer = nn.SiLU,drop_layer=None) 


        self.encoder = SqueezeformerEncoder(
                input_dim=cfg.input_dim,
                encoder_dim=cfg.encoder_dim,
                num_layers=cfg.num_layers,
                num_attention_heads= cfg.num_attention_heads,
                feed_forward_expansion_factor=cfg.feed_forward_expansion_factor,
                conv_expansion_factor= cfg.conv_expansion_factor,
                input_dropout_p= cfg.input_dropout_p,
                feed_forward_dropout_p= cfg.feed_forward_dropout_p,
                attention_dropout_p= cfg.attention_dropout_p,
                conv_dropout_p= cfg.conv_dropout_p,
                conv_kernel_size= cfg.conv_kernel_size,
        )
 
        self.classifier = ClassDecoder(cfg.MAX_LEN*cfg.encoder_dim, cfg.num_classes) #cfg.MAX_LEN * cfg.encoder_config.encoder_dim

    def forward(self, batch):
        
        x = batch['input'] 
        mask = batch['input_mask'].long()

        #Stem layer
        y = self.conv(x) #out [BS, Encoder_dim, MAX_LEN, 1]
        y = self.bn_conv(y)
        features = y.squeeze().transpose(1, 2)

        #Encoder    
        encoder_outputs = self.encoder(features, mask) #out [BS, 10, Encoder_dim]

        #Classification head
        output = self.classifier(encoder_outputs)

        return output

