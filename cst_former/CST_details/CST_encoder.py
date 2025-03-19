import torch
import torch.nn as nn
from einops import rearrange
import torch.utils.checkpoint as checkpoint

class CST_attention(torch.nn.Module):
    def __init__(self, temp_embed_dim, params):
        super().__init__()
        self.nb_mel_bins = params['nb_mel_bins']

        self.dropout_rate = params['dropout_rate']
        self.temp_embed_dim = temp_embed_dim
        self.nb_ch = params['nb_ch']

        # Channel attention w. Unfolded Local Embedding (ULE) ----------------------------------------------#
        self.patch_size_t = 10
        self.patch_size_f = 4
        self.patch_size = (self.patch_size_t, self.patch_size_f)
        self.freq_dim = int(self.nb_mel_bins / torch.prod(torch.Tensor(params['f_pool_size'])))
        self.temp_dim = 50
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        self.fold = nn.Fold(output_size=(self.temp_dim, self.freq_dim), kernel_size=self.patch_size, stride=self.patch_size)
        self.ch_attn_embed_dim = int(self.patch_size_t * self.patch_size_f)
        self.ch_mhsa = nn.MultiheadAttention(embed_dim=self.ch_attn_embed_dim,
                                             num_heads=params['nb_heads'],
                                             dropout=self.dropout_rate,
                                             batch_first=True)
        self.ch_layer_norm = nn.LayerNorm(self.temp_embed_dim)


        # Spectral attention -------------------------------------------------------------------------------#

        self.sp_attn_embed_dim = params['nb_cnn2d_filt']  # 64
        self.embed_dim_4_freq_attn = params['nb_cnn2d_filt'] # Update the temp embedding if freq attention is applied
        self.sp_mhsa = nn.MultiheadAttention(embed_dim=self.sp_attn_embed_dim,
                                             num_heads=params['nb_heads'],
                                             dropout=self.dropout_rate, batch_first=True)
        self.sp_layer_norm = nn.LayerNorm(self.temp_embed_dim)


        # Temporal Attention -----------------------------------------------------------------------------------#
        self.temp_mhsa = nn.MultiheadAttention(embed_dim=self.embed_dim_4_freq_attn,
                                               num_heads=params['nb_heads'],
                                               dropout=self.dropout_rate, batch_first=True)
        self.temp_layer_norm = nn.LayerNorm(self.temp_embed_dim)


        # Activations and Dropout
        self.activation = nn.GELU()
        self.drop_out = nn.Dropout(self.dropout_rate if self.dropout_rate > 0. else nn.Identity())

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

    def forward(self,x, M, C, T, F):
        # channel attention unfold
        B = x.size(0)
        x_init = x.clone() # X shape : B T (F C)

        x_unfold_in = rearrange(x_init, ' b t (f c) -> b c t f', c=C, t=T, f=F).contiguous() # X shape : B C T F
        x_unfold = self.unfold(x_unfold_in) # unfold for additional embedding for channel attention, unfolded shape : (B , C x kernel height x kernel width , number of patches)
        x_unfold = rearrange(x_unfold, 'b (c u) tf -> (b tf) c u', c=C).contiguous() # U : patch size dimension (4 * 10)

        xc, _ = self.ch_mhsa(x_unfold, x_unfold, x_unfold)


        xc = rearrange(xc, '(b tf) c u -> b (c u) tf', b=B).contiguous()
        xc = self.fold(xc)  # fold to rearrange. Resultant shape : (B, C * U, T', F') where T' = temp_dim = 50, F' = freq_dim
        xc = rearrange(xc, 'b c t f -> b t (f c)').contiguous()
        xc = xc + x_init # Add residual connection

        if self.dropout_rate:
            xc = self.drop_out(xc)
        xc = self.ch_layer_norm(xc)

        # spectral attention
        xs = rearrange(xc, ' b t (f c) -> (b t) f c', f=F).contiguous() # Resultant shape : ((B * T), F, C)
        xs, _ = self.sp_mhsa(xs, xs, xs) # Resultant shape : ((B * T), F, C)
        xs = rearrange(xs, ' (b t) f c -> b t (f c)', t=T).contiguous()

        xs = xs + xc
        if self.dropout_rate:
            xs = self.drop_out(xs)
        xs = self.sp_layer_norm(xs)


        # temporal attention
        xt = rearrange(xs, ' b t (f c) -> (b f) t c', f=F).contiguous()
        xt, _ = self.temp_mhsa(xt, xt, xt)
        xt = rearrange(xt, ' (b f) t c -> b t (f c)', f=F).contiguous()

        xt = xt + xs
        if self.dropout_rate:
            xt = self.drop_out(xt)
        x = self.temp_layer_norm(xt)


        return x

class CST_encoder(torch.nn.Module):
    def __init__(self, temp_embed_dim, params):
        super().__init__()
        self.freq_atten = params['FreqAtten']
        self.ch_atten_dca = params['ChAtten_DCA']
        self.ch_atten_ule = params['ChAtten_ULE']
        self.nb_ch = params['nb_ch']
        n_layers = params['nb_self_attn_layers']

        self.block_list = nn.ModuleList([CST_attention(
            temp_embed_dim = temp_embed_dim,
            params=params
        ) for _ in range(n_layers)]
        )

    def forward(self, x):
        B, C, T, F = x.size()
        M = self.nb_ch # Number of Microphone Channels

        for block in self.block_list:
            x = block(x, M, C, T, F)

        return x
