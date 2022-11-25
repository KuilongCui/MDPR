# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import os
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from fastreid.layers.weight_init import weights_init_kaiming

__all__ = [
    'Identity',
    'Flatten',
    'GlobalAvgPool',
    'GlobalMaxPool',
    'GeneralizedMeanPooling',
    'GeneralizedMeanPoolingP',
    'FastGlobalAvgPool',
    'AdaptiveAvgMaxPool',
    'ClipGlobalAvgPool',
    'BilinearAttnPool'
]


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        return input


class Flatten(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        return input.view(input.size(0), -1, 1, 1)


class GlobalAvgPool(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size=1, *args, **kwargs):
        super().__init__(output_size)


class GlobalMaxPool(nn.AdaptiveMaxPool2d):
    def __init__(self, output_size=1, *args, **kwargs):
        super().__init__(output_size)


class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm=3, output_size=(1, 1), eps=1e-6, *args, **kwargs):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'


class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """ Same, but norm is trainable
    """

    def __init__(self, norm=3, output_size=(1, 1), eps=1e-6, *args, **kwargs):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = nn.Parameter(torch.ones(1) * norm)


class AdaptiveAvgMaxPool(nn.Module):
    def __init__(self, output_size=1, *args, **kwargs):
        super().__init__()
        self.gap = FastGlobalAvgPool()
        self.gmp = GlobalMaxPool(output_size)

    def forward(self, x):
        avg_feat = self.gap(x)
        max_feat = self.gmp(x)
        feat = avg_feat + max_feat
        return feat


class FastGlobalAvgPool(nn.Module):
    def __init__(self, flatten=False, *args, **kwargs):
        super().__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)


class ClipGlobalAvgPool(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.avgpool = FastGlobalAvgPool()

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.clamp(x, min=0., max=1.)
        return x

# Bilinear Attention Pooling
class BilinearAttnPool(nn.Module):
    EPSILON = 1e-12

    def __init__(self, *args, **kwargs):
        super(BilinearAttnPool, self).__init__()

        self.pool = GeneralizedMeanPooling()
        self.attention = BasicConv2d(kwargs['feat_dim'], kwargs['attn_num'])
        self.attn_output = kwargs['attn_output']
        self.dir = kwargs['output_dir'] + "/attn/"
        self.image_dir = self.dir+"raw_image/"
        self.attn_dir = self.dir+"raw_attention/"

        if self.attn_output:
            if not os.path.isdir(self.dir):
                os.mkdir(self.dir)
            if not os.path.isdir(self.image_dir):
                os.mkdir(self.image_dir)
            if not os.path.isdir(self.attn_dir):
                os.mkdir(self.attn_dir)

    def forward(self, features, batch_inputs=None):
        attentions = self.attention(features)

        if self.attn_output:
            img_paths = batch_inputs['img_paths']
            for i in range(len(img_paths)):
                (path, filename) = os.path.split(img_paths[i])
                np.save(self.attn_dir+filename.split(".")[0]+".npy", attentions[i].detach().cpu().numpy())
                np.save(self.image_dir+filename.split(".")[0]+".npy", batch_inputs['images'][i].detach().cpu().numpy())

        B, _, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)

        feature_matrix = []
        for i in range(M):
            AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
            feature_matrix.append(AiF)
        feature_matrix = torch.cat(feature_matrix, dim=-1)
        
        # sign-sqrt
        feature_matrix = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + self.EPSILON)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix, dim=-1)

        return feature_matrix.view(B, -1, 1, 1)

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

        self.conv.apply(weights_init_kaiming)
        self.bn.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
