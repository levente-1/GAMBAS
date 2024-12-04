# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from scipy import ndimage
from . import transformer_configs as configs
from path_generate import *

from mamba_ssm import Mamba


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias,dim2=None):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        #use_dropout= use_dropo
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad3d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad3d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Mamba version
class BottleneckCNN(nn.Module):
    def __init__(self, config):
        super(BottleneckCNN, self).__init__()
        self.config = config
        use_bias = False
        norm_layer = nn.InstanceNorm3d
        padding_type = "reflect"
        
        # Residual CNN
        model = [ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False,
                             use_bias=use_bias)]
        setattr(self, "residual_cnn", nn.Sequential(*model))

    def forward(self, x):
        x = self.residual_cnn(x)
        return x

class MambaLayer(nn.Module):
    """ Mamba layer for state-space sequence modeling

    Args:
        dim (int): Model dimension.
        d_state (int): SSM state expansion factor.
        d_conv (int): Local convolution width.
        expand (int): Block expansion factor.
    
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
    
    def forward(self, x):
        B, C, D, H, W = x.shape

        # Check model dimension
        assert C == self.dim
        
        # Convert input from (B, C, H, W) to (B, H*W, C)
        x = x.float().view(B, C, -1).permute(0, 2, 1)
        
        # Normalize and pass through Mamba layer
        norm_out = self.norm(x)
        mamba_out = self.mamba(norm_out)

        # Convert output from (B, H*W, C) to (B, C, H, W)
        out = mamba_out.permute(0, 2, 1).view(B, C, D, H, W)

        return out

class cmMambaWithCNN(nn.Module):
    """ Channel-mixed Mamba (cmMamba) block with residual CNN block

    Args:
        config (dict): Model configuration.
        in_channels (int): Number of input channels.
        d_state (int): SSM state expansion factor.
        d_conv (int): Local convolution width.
        expand (int): Block expansion factor.
        ngf (int): Number of generator filters.
        norm_layer (nn.Module): Normalization layer.
        use_dropout (bool): Use dropout.
        use_bias (bool): Use bias.
        img_size (int): Image size.
    
    """
    def __init__(self, config, in_channels, d_state=16, d_conv=4, expand=2, ngf=64, norm_layer=nn.BatchNorm2d, use_bias=False):
        super().__init__()
        # Mamba block
        self.mamba_layer = MambaLayer(
            dim=in_channels, d_state=d_state, d_conv=d_conv, expand=expand
        )

        self.config = config
        ngf = 64
        padding_type = "reflect"
        use_bias = False
        norm_layer = nn.InstanceNorm3d

        # Channel compression block
        self.cc = channel_compression(ngf*8, ngf*4)

        # Residual CNN block
        model = [ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, 
                             use_bias=use_bias)]
        setattr(self, "residual_cnn", nn.Sequential(*model))

    def forward(self, x):
        # Pass input through Mamba block
        mamba_out = self.mamba_layer(x)
        x = torch.cat([x, mamba_out], dim=1)

        # Pass Mamba block output through channel compression
        x = self.cc(x)
        
        # Pass channel compressed output through residual CNN block
        x = self.residual_cnn(x)

        return x

class I2IMamba(nn.Module):
    def __init__(self, config, input_dim, img_size=224, output_dim=3):
        super(I2IMamba, self).__init__()
        self.config = config
        output_nc = output_dim
        ngf = 64
        use_bias = False
        norm_layer = nn.InstanceNorm3d
        padding_type = "reflect"
        mult = 4

        ############################################################################################
        # Layer1-Encoder1
        model = [nn.ReflectionPad3d(3),
                 nn.Conv3d(input_dim, ngf, kernel_size=7, padding=0, 
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
      
        setattr(self, "encoder_1", nn.Sequential(*model))
        ############################################################################################
        # Layer2-Encoder2
        n_downsampling = 2
        model = []
        i = 0
        mult = 2**i
        model = [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, 
                 stride=2, padding=1, bias=use_bias),
                 norm_layer(ngf * mult * 2),
                 nn.ReLU(True)]

        setattr(self, "encoder_2", nn.Sequential(*model))
        ############################################################################################
        # Layer3-Encoder3
        model = []
        i = 1
        mult = 2**i
        model = [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, 
                 stride=2, padding=1, bias=use_bias),
                 norm_layer(ngf * mult * 2),
                 nn.ReLU(True)]
        
        setattr(self, "encoder_3", nn.Sequential(*model))
        ####################################ART Blocks##############################################
        mult = 4
        img_size = 256 
        input_dim = 256 # Adjust this according to new input dimension

        # Episodic bottleneck
        # cmMamba block with residual CNN block
        self.bottleneck_1 = cmMambaWithCNN(self.config, input_dim)
        # self.bottleneck_1 = BottleneckCNN(self.config)
        
        self.bottleneck_2 = BottleneckCNN(self.config)
        self.bottleneck_3 = BottleneckCNN(self.config)
        self.bottleneck_4 = BottleneckCNN(self.config)

        # cmMamba block with residual CNN block
        self.bottleneck_5 = cmMambaWithCNN(self.config, input_dim)
        # self.bottleneck_5 = BottleneckCNN(self.config)
        
        self.bottleneck_6 = BottleneckCNN(self.config)
        self.bottleneck_7 = BottleneckCNN(self.config)
        self.bottleneck_8 = BottleneckCNN(self.config)

        # cmMamba block with residual CNN block
        self.bottleneck_9 = cmMambaWithCNN(self.config, input_dim)
        # self.bottleneck_9 = BottleneckCNN(self.config)

        ############################################################################################
        # Layer13-Decoder1 - currently removed the additional in_channels (removed * 2 for first argument), taking away skip connection to here
        n_downsampling = 2
        i = 0
        mult = 2 ** (n_downsampling - i)
        model = []
        model = [nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2), 
                                    kernel_size=3, stride=2, 
                                    padding=1, output_padding=1, 
                                    bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)]
        setattr(self, "decoder_1", nn.Sequential(*model))
        ############################################################################################
        # Layer14-Decoder2
        i = 1
        mult = 2 ** (n_downsampling - i)
        model = []
        model = [nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=2,
                                    padding=1, output_padding=1,
                                    bias=use_bias),
                 norm_layer(int(ngf * mult / 2)),
                 nn.ReLU(True)]
        setattr(self, "decoder_2", nn.Sequential(*model))
        ############################################################################################
        # Layer15-Decoder3
        model = []
        model = [nn.ReflectionPad3d(3)]
        model += [nn.Conv3d(ngf, output_dim, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        setattr(self, "decoder_3", nn.Sequential(*model))

    def forward(self, x):
        # Encoder
        x1 = self.encoder_1(x)
        x2 = self.encoder_2(x1)
        x3 = self.encoder_3(x2)

        # Episodic bottleneck
        x = self.bottleneck_1(x3)
        x = self.bottleneck_2(x)
        x = self.bottleneck_3(x)
        x = self.bottleneck_4(x)
        x = self.bottleneck_5(x)
        x = self.bottleneck_6(x)
        x = self.bottleneck_7(x)
        x = self.bottleneck_8(x)
        x = self.bottleneck_9(x)

        # Decoder
        # x = self.decoder_1(torch.cat([x, x3], dim=1))
        # x = self.decoder_2(torch.cat([x, x2], dim=1))
        # x = self.decoder_3(torch.cat([x, x1], dim=1))
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        return x

class channel_compression(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
          in_channels (int):  Number of input channels.
          out_channels (int): Number of output channels.
          stride (int):       Controls the stride.
        """
        super(channel_compression, self).__init__()

        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
          self.skip = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
            nn.InstanceNorm3d(out_channels))
        else:
          self.skip = None

        self.block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.InstanceNorm3d(out_channels))

    def forward(self, x):
        out = self.block(x)
        out += (x if self.skip is None else self.skip(x))
        out = F.relu(out)
        return out


