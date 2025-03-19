import numpy as np
from scipy.signal import lfilter
import torch
import torch.nn as nn
import torch.nn.functional as F

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_psd(X, lambda_val):
    """
    Estimate the Power Spectral Density (PSD) using a first-order IIR filter.
    
    Parameters:
    X : numpy.ndarray
        The input signal, typically the STFT of the microphone signals.
    lambda_val : float
        The smoothing factor.
    
    Returns:
    Sxx : numpy.ndarray
        The estimated Power Spectral Density.
    """
    # Compute the squared magnitude of X
    abs_X_squared = np.abs(X) ** 2
    
    # Define the filter coefficients
    b = [1 - lambda_val]
    a = [1, -lambda_val]
    
    # Apply the IIR filter along the second dimension (axis 1)
    Sxx = lfilter(b, a, abs_X_squared, axis=1)
    
    return Sxx


def estimate_cpsd(X1, X2, lambda_val):
    """
    Estimate the Power Spectral Density (PSD) using a first-order IIR filter.
    
    Parameters:
    X : numpy.ndarray
        The input signal, typically the STFT of the microphone signals.
    lambda_val : float
        The smoothing factor.
    
    Returns:
    Sxx : numpy.ndarray
        The estimated Power Spectral Density.
    """
    # Compute the squared magnitude of X
    X1_X2j = X1 * np.conj(X2)
    
    # Define the filter coefficients
    b = [1 - lambda_val]
    a = [1, -lambda_val]
    
    # Apply the IIR filter along the second dimension (axis 1)
    Sxx = lfilter(b, a, X1_X2j, axis=1)
    
    return Sxx

def estimate_cdr_nodoa(Cxx, Cnn):
    """
    Estimate the Coherence-to-Diffuse Ratio (CDR) without direction of arrival (DOA).
    
    Parameters:
    Cxx : numpy.ndarray
        The auto Power Spectral Density (PSD).
    Cnn : numpy.ndarray
        The noise Power Spectral Density (PSD).
    
    Returns:
    CDR : numpy.ndarray
        The estimated Coherence-to-Diffuse Ratio.
    """
    # Extend Cnn to the dimensions of Cxx
    Cnn = np.ones_like(Cxx) * Cnn

    # Limit the magnitude of Cxx to prevent numerical problems
    magnitude_threshold = 1 - 1e-10
    magnitudes = np.abs(Cxx)
    indices = np.where(magnitudes > magnitude_threshold)
    Cxx[indices] = magnitude_threshold * Cxx[indices] / np.abs(Cxx[indices])
    
    cdr =  (-(np.abs(Cxx)**2 + Cnn**2 * np.real(Cxx) ** 2 - Cnn ** 2 * np.abs(Cxx) ** 2 - 2 * Cnn * np.real(Cxx) + Cnn ** 2) ** (1/2) - np.abs(Cxx) ** 2 + Cnn * np.real(Cxx))/(np.abs(Cxx) ** 2-1)

    cdr = np.maximum(np.real(cdr) , 0)

    return cdr

def init_layer(layer, method='kaiming_uniform'):
    """Initialize a Linear or Convolutional layer. """
    if method == 'xavier_uniform':  ## default
        nn.init.xavier_uniform_(layer.weight)
    elif method == 'xavier_normal':
        nn.init.xavier_normal_(layer.weight)
    elif method == 'kaiming_uniform':  ## to try
        nn.init.kaiming_uniform_(layer.weight)
    elif method == 'kaiming_normal':
        nn.init.kaiming_normal_(layer.weight)
    elif method == 'orthogonal':
        nn.init.orthogonal_(layer.weight)
    else:
        raise NotImplementedError('init method {} is not implemented'.format(method))

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)
    
class AFF(nn.Module):
    
    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        
        xr = x + residual # X + Y (residual)
        xr_l = self.local_att(xr) # Local Attention of MS-CAM
        xr_g = self.global_att(xr) # Global Attention of MS-CAM
        
        xr_out = xr_l + xr_g
        xr_out = self.sigmoid(xr_out) # Output of MS-CAM
        
        out = xr_out * x + (1 - xr_out) * residual
        
        return out 
    
    
"""To use the CBAM

self.ca = ChannelAttention(in_planes = n_conv_filter_out)
self.sa = SpatialAttention()

x = F.relu_(self.bn1(self.conv1(x)))
x = self.bn2(self.conv2(x))
x = self.ca(x) * x
x = self.sa(x) * x
x += residual
x = F.relu_(x)

return x
"""
class ChannelAttention(nn.Module):
    """Channel Attention as proposed in the paper 'Convolutional Block Attention Module'"""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """Spatial Attention as proposed in the paper 'Convolutional Block Attention Module'"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ChannelSETorchLayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*

    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSETorchLayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.avgpool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(num_channels, num_channels_reduced, 1)
        self.fc2 = nn.Conv2d(num_channels_reduced, num_channels, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # Create the recalibration vector from SqueezeExcite
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)

        # Recalibrate the input
        x = scale * x
        return x
    

class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*

    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """

        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class SpatialSELayer(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """

        :param num_channels: No of input channels
        """
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """

        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, a, b = input_tensor.size()

        if weights is not None:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        squeeze_tensor = squeeze_tensor.view(batch_size, 1, a, b)
        output_tensor = torch.mul(input_tensor, squeeze_tensor)
        return output_tensor


class ChannelSpatialSELayer(nn.Module):
    """
    Re-implementation of concurrent spatial and channel squeeze & excitation:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018, arXiv:1803.02579*
    """

    def __init__(self, num_channels, reduction_ratio=4):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)

    def forward(self, input_tensor):
        """

        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        output_tensor = torch.add(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor


class stemConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        elif pool_type is None:
            pass
        else:
            raise Exception('Incorrect argument!')

        return x

class preAttn(nn.Module):
    def __init__(self, in_channels, n_attn1, n_attn2=None):
        super().__init__()

        if n_attn2 is None:
            n_attn2 = n_attn1 * 4

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=n_attn1,
                               kernel_size=(3,3), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(n_attn1)
        self.conv2 = nn.Conv2d(in_channels=n_attn1, out_channels=n_attn2,
                               kernel_size=(3,3), padding="same", bias=False)
        self.bn2 = nn.BatchNorm2d(n_attn2)
        self.conv3 = nn.Conv2d(in_channels=n_attn2, out_channels=in_channels, kernel_size=1, padding="same", bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.conv3(x)
        x = torch.sigmoid(x)

        return x