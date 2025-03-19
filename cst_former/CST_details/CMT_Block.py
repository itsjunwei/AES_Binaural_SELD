import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from .CST_encoder import CST_attention
from .layers import LocalPerceptionUint, InvertedResidualFeedForward

class CMT_Layers(torch.nn.Module):
    def __init__(self, params, temp_embed_dim, ffn_ratio=4., drop_path_rate=0.):
        super().__init__()
        self.temp_embed_dim = temp_embed_dim
        self.ffn_ratio = ffn_ratio
        self.dim = params['nb_cnn2d_filt']
        self.nb_ch = params['nb_ch']

        self.norm1 = nn.LayerNorm(self.dim)
        self.LPU = LocalPerceptionUint(self.dim)
        self.IRFFN = InvertedResidualFeedForward(self.dim, self.ffn_ratio)

        self.cst_attention = CST_attention(temp_embed_dim=self.temp_embed_dim,params=params)

        self.drop_path = nn.Dropout(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):

        lpu = self.LPU(x)
        x = x + lpu

        B, C, T, F = x.size()
        x = rearrange(x, 'b c t f -> b t (f c)')
        x = self.cst_attention(x,self.nb_ch,C, T, F)

        x_2 = rearrange(x, 'b t (f c) -> b (t f) c', f=F).contiguous()
        x_res = rearrange(x, 'b t (f c) -> b c t f', f=F).contiguous()
        norm1 = self.norm1(x_2)
        norm1 = rearrange(norm1, 'b (t f) c -> b c t f', f=F).contiguous()
        ffn = self.IRFFN(norm1)
        x = x_res + self.drop_path(ffn)

        return x

class CMT_block(torch.nn.Module):
    def __init__(self, params, temp_embed_dim, ffn_ratio=4., drop_path_rate=0.1):
        super().__init__()
        self.temp_embed_dim = temp_embed_dim
        self.num_layers = params['nb_self_attn_layers']
        self.ffn_ratio = ffn_ratio
        self.nb_ch = params['nb_ch']

        self.block_list = nn.ModuleList([CMT_Layers(
            params=params,
            temp_embed_dim=self.temp_embed_dim,
            ffn_ratio=self.ffn_ratio,
            drop_path_rate=drop_path_rate
        ) for i in range(self.num_layers)]
        )

    def forward(self, x):
        B, C, T, F = x.size()
        M = self.nb_ch

        for block in self.block_list:
            x = block(x) # After : B C T F

        x = rearrange(x, 'b c t f -> b t (f c)', c=C, t=T, f=F).contiguous()

        return x
