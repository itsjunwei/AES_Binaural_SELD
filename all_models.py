"""Main Python script to hold all the models that I want to test out"""

from conformer.encoder import ConformerBlock
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile, clever_format
from utils import *
from parameters import get_params

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size=(3, 3), stride=(1, 1), 
                 padding=(1, 1), add_bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                              kernel_size=kernel_size, stride=stride, 
                              padding=padding, bias=add_bias)
        self.bn = nn.BatchNorm2d(out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize nn.Linear and nn.LayerNorm
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
        x = F.relu_(self.bn(self.conv(x)))
        return x

class ConvBlockTwo(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize nn.Linear and nn.LayerNorm
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
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return x

class ResLayer(nn.Module):
    """Initialize a ResNet layer"""
    def __init__(self, in_channels, out_channels):
        super(ResLayer, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3,3), stride=(1,1), 
                               padding=(1,1), bias=False)

        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3,3), stride=(1,1),
                               padding=(1,1), bias=False)

        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=(1,1), bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize nn.Linear and nn.LayerNorm
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

        identity = x.clone()
        out = F.relu_(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None: 
            out += self.shortcut(identity)
        else:
            out += identity
        out = F.relu_(out)
        return out

class ResNetConf18(nn.Module):
    """Create a ResNet-18 system with 2 x Conformer blocks. Option to include
    Squeeze Excite layers"""

    def __init__(self, in_feat_shape, out_feat_shape, normd=True,
                 num_classes=13, use_se=False, use_stemse=False,
                 use_finalse = False, use_cbam=False,
                 device = None, verbose = False):
        super().__init__()

        self.nb_classes = num_classes   # Number of classes
        self.distance_normed = normd    # Normalize distances
        self.use_se = use_se            # Include all the SqEx layers after each ResBlk
        self.use_stemse = use_stemse    # Include the Spatial SqEx in the stem convolution
        self.use_finalse = use_finalse  # Include the final SqEx in the final convolution
        self.use_cbam = use_cbam        # Include the CBAM after the final convolution
        self.device = device            # Device used. CUDA if possible
        self.verbose = verbose          # True if verbose

        # conv1
        self.conv1 = ConvBlock(in_channels=in_feat_shape[1],
                               out_channels=24,
                               kernel_size=(5,5), padding="same")

        if self.use_stemse:
            self.stem_spatialsqex = SpatialSELayer(num_channels=24)
        self.stempool = AvgMaxPool((1,2))
        
        # resnet layer 1
        self.ResNet_1 = ResLayer(in_channels=24, out_channels=24)
        if self.use_se:
            self.se_1 = ChannelSpatialSELayer(num_channels=24, reduction_ratio=4)
        self.ResNet_2 = ResLayer(in_channels=24, out_channels=24)
        if self.use_se:
            self.se_2 = ChannelSpatialSELayer(num_channels=24, reduction_ratio=4)
        self.pooling1 = AvgMaxPool((2,2))
        # resnet layer 2
        self.ResNet_3 = ResLayer(in_channels=24, out_channels=48)
        if self.use_se:
            self.se_3 = ChannelSpatialSELayer(num_channels=48, reduction_ratio=4)
        self.ResNet_4 = ResLayer(in_channels=48, out_channels=48)
        if self.use_se:
            self.se_4 = ChannelSpatialSELayer(num_channels=48, reduction_ratio=4)
        self.pooling2 = AvgMaxPool((2,2))
        # resnet layer 3
        self.ResNet_5 = ResLayer(in_channels=48, out_channels=96)
        if self.use_se:
            self.se_5 = ChannelSpatialSELayer(num_channels=96, reduction_ratio=4)
        self.ResNet_6 = ResLayer(in_channels=96, out_channels=96)
        if self.use_se:
            self.se_6 = ChannelSpatialSELayer(num_channels=96, reduction_ratio=4)
        self.pooling3 = AvgMaxPool((2,2))
        # resnet layer 4
        self.ResNet_7 = ResLayer(in_channels=96, out_channels=192)
        if self.use_se:
            self.se_7 = ChannelSpatialSELayer(num_channels=192, reduction_ratio=4)
        self.ResNet_8 = ResLayer(in_channels=192, out_channels=192)
        if self.use_se:
            self.se_8 = ChannelSpatialSELayer(num_channels=192, reduction_ratio=4)
        self.pooling4 = AvgMaxPool((1,2))

        # conv2
        self.conv2 = ConvBlock(in_channels=192,
                               out_channels=256)

        if self.use_finalse:
            self.finalse = ChannelSpatialSELayer(num_channels=256, reduction_ratio=4)  
        elif self.use_cbam:
            self.ca = ChannelAttention(in_planes=256, ratio=16)
            self.sa = SpatialAttention(kernel_size=3)

        # conformer
        self.conformer1 = ConformerBlock(256)
        self.conformer2 = ConformerBlock(256)

        # Time downsampling to match label rate
        self.timepooling = nn.AvgPool2d((4,1))

        # FC1
        self.dropout1 = nn.Dropout(p=0.05)
        self.fc1 = nn.Linear(256, 256)
        self.leaky = nn.LeakyReLU()
        # FC2
        self.dropout2 = nn.Dropout(p=0.05)
        self.fc2 = nn.Linear(256, out_feat_shape[-1])

        self.init_params()   

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.conv1(x)
        if self.use_stemse:
            x = self.stem_spatialsqex(x)
        x = self.stempool(x)
        if self.verbose: print("After stem : {}".format(x.shape))

        x = self.ResNet_1(x)
        if self.use_se:
            x = self.se_1(x)
        x = self.ResNet_2(x)
        if self.use_se:
            x = self.se_2(x)
        x = self.pooling1(x)
        if self.verbose: print("After ResBlk 1 : {}".format(x.shape))

        x = self.ResNet_3(x)
        if self.use_se:
            x = self.se_3(x)
        x = self.ResNet_4(x)
        if self.use_se:
            x = self.se_4(x)
        x = self.pooling2(x)
        if self.verbose: print("After ResBlk 2 : {}".format(x.shape))

        x = self.ResNet_5(x)
        if self.use_se:
            x = self.se_5(x)
        x = self.ResNet_6(x)
        if self.use_se:
            x = self.se_6(x)
        x = self.pooling3(x)
        if self.verbose: print("After ResBlk 3 : {}".format(x.shape))

        x = self.ResNet_7(x)
        if self.use_se:
            x = self.se_7(x)
        x = self.ResNet_8(x)
        if self.use_se:
            x = self.se_8(x)
        x = self.pooling4(x)
        if self.verbose: print("After ResBlk 4 : {}".format(x.shape))

        x = self.conv2(x)
        
        if self.use_finalse:
            x = self.finalse(x)
        elif self.use_cbam:
            x = self.ca(x) * x
            x = self.sa(x) * x

        # x = self.timepooling(x) # RC Middle Pooling
        x_mean = torch.mean(x, dim=3)
        x_max, _ = torch.max(x, dim=3)
        x = x_mean + x_max
        x = x.permute(0, 2, 1)

        x = self.conformer1(x)
        x = self.conformer2(x)
        if self.verbose: print("After Conformer : {}".format(x.shape))

        x = self.leaky(self.fc1(self.dropout1(x)))
        x = self.fc2(self.dropout2(x))
        if self.distance_normed:
            x = torch.tanh(x)

        return x

class AvgMaxPool(nn.Module):
    def __init__(self, poolsize):
        super().__init__()
        self.poolsize = poolsize

    def forward(self, x):
        x1 = F.avg_pool2d(x, kernel_size = self.poolsize)
        x2 = F.max_pool2d(x, kernel_size = self.poolsize)
        x_out = x1 + x2

        return x_out


class ResNet(nn.Module):
    def __init__(self, in_feat_shape, out_feat_shape, 
                 normd=True,
                 gru_size = 256, verbose=False,
                 niu_resnet = False,
                 res_layers = [64, 64, 128, 256, 512]):
        super().__init__()

        self.niu_resnet = niu_resnet
        if self.niu_resnet:
            res_layers = [24, 24, 48, 96, 192]

        self.res_layers = res_layers
        self.verbose = verbose
        self.normd = normd

        # resnet stem
        if self.niu_resnet:
            self.stem = ConvBlock(in_channels= in_feat_shape[1],
                                  out_channels=res_layers[0])
            self.stempool = AvgMaxPool((1,2))
        else:
            self.stem = ConvBlockTwo(in_channels= in_feat_shape[1],
                                  out_channels=res_layers[0])
            self.stempool = AvgMaxPool((2,2))

        # resnet layer 1
        self.ResNet_1 = ResLayer(in_channels=self.res_layers[0], out_channels=self.res_layers[1])
        self.ResNet_2 = ResLayer(in_channels=self.res_layers[1], out_channels=self.res_layers[1])
        self.pooling1 = nn.AvgPool2d((2,2))

        # resnet layer 2
        self.ResNet_3 = ResLayer(in_channels=self.res_layers[1], out_channels=self.res_layers[2])
        self.ResNet_4 = ResLayer(in_channels=self.res_layers[2], out_channels=self.res_layers[2])
        self.pooling2 = nn.AvgPool2d((2,2))

        # resnet layer 3
        self.ResNet_5 = ResLayer(in_channels=self.res_layers[2], out_channels=self.res_layers[3])
        self.ResNet_6 = ResLayer(in_channels=self.res_layers[3], out_channels=self.res_layers[3])
        if self.niu_resnet:
            self.pooling3 = nn.AvgPool2d((2,2))
        else:
            self.pooling3 = nn.AvgPool2d((1,2))

        # resnet layer 4
        self.ResNet_7 = ResLayer(in_channels=self.res_layers[3], out_channels=self.res_layers[4])
        self.ResNet_8 = ResLayer(in_channels=self.res_layers[4], out_channels=self.res_layers[4])
        self.pooling4 = nn.AvgPool2d((1,2))

        # determining the bigru size
        if self.niu_resnet:
            self.tail_conv = ConvBlock(in_channels=self.res_layers[-1],
                                       out_channels=gru_size)
            gru_in = gru_size
        else:
            gru_in = self.res_layers[-1]

        self.bigru = nn.GRU(input_size = gru_in, hidden_size = gru_size,
                            num_layers = 2, batch_first=True, bidirectional=True, dropout=0.05)

        # decoding layers
        self.fc1 = nn.Linear(in_features=gru_size * 2,
                             out_features=gru_size, bias=True)
        self.dropout1 = nn.Dropout(p=0.05)
        self.leaky = nn.LeakyReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=gru_size,
                             out_features=out_feat_shape[-1], bias=True)
        self.dropout2 = nn.Dropout(p=0.05)

        # Normalized Distance Activation function
        if self.normd:
            self.final_out = nn.Tanh()
        else:
            self.final_out = None

    def forward(self, x):

        x = self.stem(x)
        x = self.stempool(x)

        x = self.ResNet_1(x)
        x = self.ResNet_2(x)
        x = self.pooling1(x)
        if self.verbose:
            print("After R1 : {}".format(x.shape))

        x = self.ResNet_3(x)
        x = self.ResNet_4(x)
        x = self.pooling2(x)
        if self.verbose:
            print("After R2 : {}".format(x.shape))

        x = self.ResNet_5(x)
        x = self.ResNet_6(x)
        x = self.pooling3(x)
        if self.verbose:
            print("After R3 : {}".format(x.shape))

        x = self.ResNet_7(x)
        x = self.ResNet_8(x)
        x = self.pooling4(x)
        if self.verbose:
            print("After R4 : {}".format(x.shape))

        if self.niu_resnet:
            x = self.tail_conv(x)
            if self.verbose:
                print("After ResNet : {}".format(x.shape))

        # Preparing for biGRU layers
        x = torch.mean(x, dim=3)
        x = x.transpose(1,2).contiguous()
        x , _ = self.bigru(x)
        x = torch.tanh(x)

        # Fully connected decoding layers
        x = self.leaky(self.fc1(self.dropout1(x)))
        x = self.fc2(self.dropout2(x))

        if self.final_out is not None:
            x = self.final_out(x)

        return x



if __name__ == "__main__":
    # input_feature_shape = (1, 8, 400, 200) # SALSA (FOA) input shape
    input_feature_shape = (1, 8, 400, 200) # SALSA-Lite (MIC) input shape
    # input_feature_shape = (1, 7, 400, 128) # MelSpecIV input shape 
    output_feature_shape = (1, 50, 156)

    # 8.797G MACS and 6.177M Params (128 melbins -- 5.928G MACs)
    # model = DCASE_Model(in_feat_shape=input_feature_shape,
    #                     out_feat_shape=output_feature_shape,
    #                     normd=True, use_se=True, use_stemse=True,
    #                     res_layers=[32, 32, 64, 128, 256])

    """
    SELDNet+ Complexity
    ----------------------------------------------------
    Feature         | 2x2 Pooling | 512 and 444 Pooling 
    ----------------------------------------------------
    MelSpecIV 128   | 0.932G MACs | 1.177G MACs
    DRRIV 128       | 0.946G MACs | 1.207G MACs
    ----------------------------------------------------
    LinSpec 200     | 1.395G MACs | 1.746G MACs
    LinSpecDRR 200  | 1.418G MACs | 1.792G MACs
    ----------------------------------------------------
    SALSA-Lite 191  | 1.317G MACs | 1.597G MACs
    SALSA-Dlite 191 | 1.339G MACs | 1.641G MACs
    ----------------------------------------------------
    """
    # model = CRNN4(in_feat_shape=input_feature_shape,
    #               out_feat_shape=output_feature_shape,
    #               conv_filters=[32, 64, 128, 256],
    #               norm_distance=True,
    #               verbose=True)
    
    # model = Seld_ResNet(in_feat_shape=input_feature_shape, out_feat_shape=output_feature_shape,
    #                     n_classes=13)

    """ 5.383M Params (MelSpecIV 1024 NFFT, 128 MelBins)
        AvgPool 2x2 at stem --> 6.801G MACs (2.924 MACs)
        ResNet256 + TA Conf --> 6.674G MACs 
        Freq only stempool  --> 12.933G MACs
        No stempool at all  --> 18.245G MACs
    """
    # model = ResNetConf18(in_feat_shape=input_feature_shape,
    #                      out_feat_shape=output_feature_shape,
    #                      normd=True, num_classes=13, use_se=True, use_stemse=True,
    #                      use_finalse=True, verbose=True)

    """4.501M Params
        With 32-256 filters --> 6.702G MACs"""
    # model = ResNetConf8(in_feat_shape=input_feature_shape, out_feat_shape=output_feature_shape,
    #                     normd=True, use_cbam=False, use_se=False, use_finalse=False, use_taconf=False)

    """
    Niu ResNet Complexity
    ----------------------------
    SALSA-Lite | 7 X 400 X 191  | 1.624G MACs | 4.165M Params
    SALSA-DRR  | 8 X 400 X 191  | 1.641G MACs | 4.166M Params
    MelGCC     | 10 X 400 X 128 | 1.223G MACs | 4.166M Params
    LinGCC     | 10 X 400 X 200 | 1.826G MACs | 4.166M Params
    SALSA-Mel  | 7 X 400 X 128  | 1.190G MACs | 4.165M Params

    MelSpecIV 128  | 1.769G MACs
    DRRIV 128      | 1.780G MACs
    LinSpec 200    | 2.669G MACs
    LinSpecDRR 200 | 2.686G MACs
    """
    model = ResNet(in_feat_shape=input_feature_shape,
                   out_feat_shape=output_feature_shape,
                   normd=False,
                   verbose=True, niu_resnet=True)
    
    # from cst_former.CST_former_model import CST_former
    # model = CST_former(in_feat_shape=input_feature_shape,
    #                    out_shape=output_feature_shape,
    #                    params=get_params(argv='drrsl3f', verbose=False))

    x = torch.rand((input_feature_shape), device=torch.device("cpu"), requires_grad=True)
    y = model(x)

    print("Total Params : {:.3f}M".format(count_parameters(model)/(10**6)))

    macs, params = profile(model, inputs=(torch.randn(input_feature_shape), ))
    macs, params = clever_format([macs, params], "%.3f")
    print("{} MACS and {} Params".format(macs, params))
    print("Output shape : {}".format(y.shape))
    
    import torchinfo
    model_profile = torchinfo.summary(model, input_size=input_feature_shape)
    print('MACC:\t \t %.3f' %  (model_profile.total_mult_adds/1e9), 'G')
    print('Memory:\t \t %.3f' %  (model_profile.total_params/1e6), 'M\n')
    
    del x, y, model
