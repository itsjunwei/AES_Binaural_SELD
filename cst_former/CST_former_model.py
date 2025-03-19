import numpy as np
import torch
import torch.nn as nn

from .CST_details.encoder import Encoder
from .CST_details.CMT_Block import CMT_block
from .CST_details.layers import FC_layer

class CST_former(torch.nn.Module):
    """
    CST_former : Channel-Spectral-Temporal Transformer for SELD task
    """
    def __init__(self, in_feat_shape, out_shape, params):
        super().__init__()
        self.nb_classes = params['unique_classes']
        self.t_pooling_loc = params["t_pooling_loc"]
        params['nb_mel_bins'] = in_feat_shape[-1]
        params['nb_ch'] = in_feat_shape[1]
        params['f_pool_size'] = [2, 2, 2]
        params['t_pool_size'] = [2, 2, 2]
        self.encoder = Encoder(in_feat_shape, params)


        self.conv_block_freq_dim = int(np.floor(in_feat_shape[-1] / np.prod(params['f_pool_size'])))
        self.input_nb_ch = in_feat_shape[1]
        self.temp_embed_dim = self.conv_block_freq_dim * params['nb_cnn2d_filt']

        ## Attention Layer===========================================================================================#
        self.attention_stage = CMT_block(params, self.temp_embed_dim)


        ## Fully Connected Layer ======================================================================================#
        self.fc_layer = FC_layer(out_shape, self.temp_embed_dim, params)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """input: (batch_size, mic_channels, time_steps, mel_bins)"""

        x = self.encoder(x)
        x = self.attention_stage(x)
        x = self.fc_layer(x)

        return x