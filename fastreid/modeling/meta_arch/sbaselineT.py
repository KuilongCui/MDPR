# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import copy
import random

from matplotlib.artist import get

from fastreid.layers.batch_norm import IBN, get_norm
from fastreid.layers.pooling import GeneralizedMeanPoolingP, AdaptiveAvgMaxPool
from fastreid.layers.weight_init import weights_init_kaiming
import torch
from torch import nn
import torch.nn.functional as F
from fastreid.config import configurable
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY

@META_ARCH_REGISTRY.register()  # type: ignore
class SbaselineT(nn.Module):
    """
    Baseline architecture. Any models that contains the following two components:
    1. Per-image feature extraction (aka backbone)
    2. Per-image feature aggregation and loss computation
    """

    @configurable
    def __init__(
            self,
            *,
            last_embedding,
            last_head,
            last_only,
            num_class,
            attach_activation,
            attnShaped_activation,
            detach_attn,
            detach_all,
            drop,
            drop_H,
            drop_W,
            backbone1,
            b1_head,
            attn_head,
            last_stride,
            backbone_embedding_dim,
            feat_dim,
            embedding_dim,
            num_attn,
            backbone,
            heads,
            pixel_mean,
            pixel_std,
            with_ibn,
            norm_type,
            loss_kwargs
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone:
            heads:
            pixel_mean:
            pixel_std:
        """
        super().__init__()
        # backbone
        self.backbone = backbone
        # head
        self.heads = heads

        self.detach_attn = detach_attn
        self.detach_all = detach_all

        self.loss_kwargs = loss_kwargs

        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)

        self.num_attn = num_attn
        self.feat_dim = feat_dim

        if last_stride == 1:
            self.upSamper3 = Embedding(self.num_attn, self.num_attn, with_ibn=with_ibn, norm_type=norm_type, bias=True)
        else:
            self.upSamper3 = UpSamper(in_dim=self.num_attn, output_dim=self.num_attn, with_ibn=with_ibn, norm_type=norm_type)
        
        self.upSamper2 = UpSamper(in_dim=self.num_attn, output_dim=self.num_attn, with_ibn=with_ibn, norm_type=norm_type)
        self.upSamper1 = UpSamper(in_dim=self.num_attn, output_dim=self.num_attn, with_ibn=with_ibn, norm_type=norm_type)
        
        embedding_dim = backbone_embedding_dim
        
        self.attention_maker = BasicConv2d(self.feat_dim, num_attn, with_ibn=with_ibn, norm_type=norm_type)
        self.attach = Attach(self.feat_dim, self.num_attn, with_ibn=with_ibn, norm_type=norm_type, embedding_dim=embedding_dim, attach_activation=attach_activation)
        self.attnShaped3 = AttnShaped(self.feat_dim//2, with_ibn=with_ibn, norm_type=norm_type, attnShaped_activation=attnShaped_activation)
        self.attnShaped2 = AttnShaped(self.feat_dim//4, with_ibn=with_ibn, norm_type=norm_type, attnShaped_activation=attnShaped_activation)
        self.attnShaped1 = AttnShaped(self.feat_dim//8, with_ibn=with_ibn, norm_type=norm_type, attnShaped_activation=attnShaped_activation)

        self.backbone1 = backbone1
        self.head1 = b1_head
        self.attn_head = attn_head

        self.drop = drop
        if drop:
            self.dropblock = BatchDrop(drop_H, drop_W)

        self.embedding4 = Embedding(feat_dim, embedding_dim, with_ibn=with_ibn, norm_type=norm_type)
        self.embedding3 = Embedding(feat_dim, embedding_dim, with_ibn=with_ibn, norm_type=norm_type)
        self.embedding2 = Embedding(feat_dim, embedding_dim, with_ibn=with_ibn, norm_type=norm_type)
        self.embedding1 = Embedding(feat_dim, embedding_dim, with_ibn=with_ibn, norm_type=norm_type)

        if 'ProxyAnchorLoss' in self.loss_kwargs['loss_names']:
            self.proxy_anchor = Proxy_Anchor(num_class, embedding_dim)

        if 'MultiSimilarityLoss' in self.loss_kwargs['loss_names']:
            self.ms = MultiSimilarityLoss()

        self.addtion_embedding = last_embedding
        self.last_only = last_only

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        heads = build_heads(cfg)

        all_blocks = backbone

        backbone1 = nn.ModuleList([
            copy.deepcopy(all_blocks.layer2),
            copy.deepcopy(all_blocks.layer3),
            copy.deepcopy(all_blocks.layer4)]
        )

        b1_head = build_heads(cfg)
        attn_head = build_heads(cfg, True)

        if cfg.MODEL.LAST_EMBEDDING:
            last_head = build_heads(cfg, last_feat=cfg.MODEL.BACKBONE.FEAT_DIM)
        else:
            last_head = None

        dropblock = cfg.MODEL.DROP.ENABLED 
        return {
            'last_embedding': cfg.MODEL.LAST_EMBEDDING,
            'last_head': last_head,
            'last_only': cfg.MODEL.LAST_ONLY,
            'num_class': cfg.MODEL.HEADS.NUM_CLASSES,
            'attach_activation': cfg.MODEL.BACKBONE.ATTACH_ACTIVATION,
            'attnShaped_activation': cfg.MODEL.BACKBONE.ATTNSHAPED_ACTIVATION,
            'detach_attn': cfg.MODEL.DETACH_ATTN,
            'detach_all': cfg.MODEL.DETACH_ALL,
            'drop': dropblock,
            'drop_H': cfg.MODEL.DROP.H_RATIO,
            'drop_W': cfg.MODEL.DROP.W_RATIO,
            'backbone1': backbone1,
            'b1_head': b1_head,
            'attn_head': attn_head,
            'last_stride': cfg.MODEL.BACKBONE.LAST_STRIDE,
            'backbone_embedding_dim': cfg.MODEL.BACKBONE.EMBEDDING_DIM,
            'feat_dim': cfg.MODEL.BACKBONE.FEAT_DIM,
            'embedding_dim': cfg.MODEL.HEADS.EMBEDDING_DIM,
            'num_attn': cfg.MODEL.HEADS.ATTN,
            'backbone': backbone,
            'heads': heads,
            'pixel_mean': cfg.MODEL.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.PIXEL_STD,
            'with_ibn': cfg.MODEL.BACKBONE.WITH_IBN,
            'norm_type': cfg.MODEL.BACKBONE.NORM,
            'loss_kwargs':
                {
                    # loss name
                    'loss_names': cfg.MODEL.LOSSES.NAME,

                    # loss hyperparameters
                    'ce': {
                        'eps': cfg.MODEL.LOSSES.CE.EPSILON,
                        'alpha': cfg.MODEL.LOSSES.CE.ALPHA,
                        'scale': cfg.MODEL.LOSSES.CE.SCALE,
                        'attn_scale': cfg.MODEL.LOSSES.CE.ATTN_SCALE,
                        'main_scale': cfg.MODEL.LOSSES.CE.MAIN_SCALE
                    },
                    'tri': {
                        'margin': cfg.MODEL.LOSSES.TRI.MARGIN,
                        'norm_feat': cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                        'hard_mining': cfg.MODEL.LOSSES.TRI.HARD_MINING,
                        'scale': cfg.MODEL.LOSSES.TRI.SCALE,
                        'attn': cfg.MODEL.LOSSES.TRI.ATTN
                    },
                    'circle': {
                        'margin': cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.CIRCLE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.CIRCLE.SCALE
                    },
                    'cosface': {
                        'margin': cfg.MODEL.LOSSES.COSFACE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.COSFACE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.COSFACE.SCALE
                    }
                }
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)

        x1,x2,x3,feature4 = self.backbone(images)

        a4 = self.attention_maker(feature4)
        attn_feat = self.attach(a4, feature4)

        if self.detach_attn:
            a3 = self.upSamper3(a4.detach())
        else:
            a3 = self.upSamper3(a4)

        a2 = self.upSamper2(a3)
        a1 = self.upSamper1(a2)

        if self.detach_all:
            f3 = self.attnShaped3(a3, x3.detach())
            f2 = self.attnShaped2(a2, x2.detach())
            f1 = self.attnShaped1(a1, x1.detach())
        else:
            f3 = self.attnShaped3(a3, x3)
            f2 = self.attnShaped2(a2, x2)
            f1 = self.attnShaped1(a1, x1)

        feature1 = self.backbone1[0](f1)

        feature1 = feature1 * 0.5 + f2 * 0.5

        feature1 = self.backbone1[1](feature1)

        feature1 = feature1 * 0.5 + f3 * 0.5

        feature1 = self.backbone1[2](feature1)


        feature4 = self.embedding4(feature4)
        feature1 = self.embedding1(feature1)
        
        if self.drop:
            feature1 = self.dropblock(feature1)

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"]

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            outputs4 = self.heads(feature4, targets, batched_inputs=batched_inputs)
            outputs1 = self.head1(feature1, targets, batched_inputs=batched_inputs)
            attn_output = self.attn_head(attn_feat, targets, batched_inputs=batched_inputs)

            losses = self.losses(attn_output, outputs4, outputs1, targets)
          
            return losses
        else:
            # attn_output = self.attn_head(attn_feat, batched_inputs=batched_inputs)
            outputs4 = self.heads(feature4, batched_inputs=batched_inputs) # 88.88
            outputs1 = self.head1(feature1, batched_inputs=batched_inputs) # 87.79

            eval_feat = torch.cat((outputs4, outputs1), dim=1)

            return eval_feat

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs['images']
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        images.sub_(self.pixel_mean).div_(self.pixel_std)  # type: ignore
        return images
    
    def addtion_loss(self, outputs, gt_labels, prefix):
        b1_logits         = outputs['cls_outputs']
        b1_pool_feat      = outputs['features']

        loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']

        assert len(prefix) != 0

        prefix = 'loss_' + prefix + "_"

        
        if 'CrossEntropyLoss' in loss_names:
            ce_kwargs = self.loss_kwargs.get('ce')

            loss_dict[prefix + 'cls'] = cross_entropy_loss(
                b1_logits,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('main_scale')

        if 'TripletLoss' in loss_names:
            tri_kwargs = self.loss_kwargs.get('tri')
            loss_dict[prefix + 'triplet'] = triplet_loss(
                b1_pool_feat,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale')

        return loss_dict

    def losses(self, attn_output, b4_outputs, b1_outputs, gt_labels, prefix=""):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        # fmt: off
        pred_class_logits = attn_output['pred_class_logits'].detach()

        b1_logits         = b1_outputs['cls_outputs']
        b4_logits         = b4_outputs['cls_outputs']
        attn_logits       = attn_output['cls_outputs']

        b1_pool_feat      = b1_outputs['features']
        b4_pool_feat      = b4_outputs['features']

        # fmt: on

        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']

        origin_prefix = prefix

        if len(prefix) == 0:
            prefix = 'loss_'
        else:
            prefix = 'loss_' + prefix + "_"

        if 'CrossEntropyLoss' in loss_names:
            ce_kwargs = self.loss_kwargs.get('ce')

            loss_dict[prefix + 'cls_b1'] = cross_entropy_loss(
                b1_logits,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('main_scale')

            loss_dict[prefix + 'cls_b4'] = cross_entropy_loss(
                b4_logits,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale')
            
            if len(origin_prefix) == 0:
                loss_dict[prefix + 'cls_attn'] = cross_entropy_loss(
                    attn_logits,
                    gt_labels,
                    ce_kwargs.get('eps'),
                    ce_kwargs.get('alpha')
                ) * ce_kwargs.get('attn_scale')

        if 'TripletLoss' in loss_names:
            tri_kwargs = self.loss_kwargs.get('tri')
            loss_dict[prefix + 'triplet_b1'] = triplet_loss(
                b1_pool_feat,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale')

            loss_dict[prefix + 'triplet_b4'] = triplet_loss(
                b4_pool_feat,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale')

        if 'ProxyAnchorLoss' in loss_names:
            loss_dict[prefix + 'proxy_anchor_b1'] = self.proxy_anchor(
                b1_pool_feat,
                gt_labels,
            )

            loss_dict[prefix + 'proxy_anchor_b4'] = self.proxy_anchor(
                b4_pool_feat,
                gt_labels
            )
        
        if 'MultiSimilarityLoss' in loss_names:
            loss_dict[prefix + 'proxy_ms_b1'] = self.ms(
                b1_pool_feat,
                gt_labels,
            )

            loss_dict[prefix + 'proxy_ms_b4'] = self.ms(
                b4_pool_feat,
                gt_labels
            )
            
        return loss_dict

class UpSamper(nn.Module):
    def __init__(self, in_dim, with_ibn, norm_type, output_dim=None, scale=2):
        super(UpSamper, self).__init__()

        if output_dim is None:
            self.up2=nn.Sequential(
                nn.ConvTranspose2d(in_dim , in_dim, kernel_size=scale, stride=scale),
                nn.Conv2d(in_dim, in_dim, 1),
            )
        else:
            self.up2=nn.Sequential(
                nn.ConvTranspose2d(in_dim , in_dim, kernel_size=scale, stride=scale),
                nn.Conv2d(in_dim, output_dim, 1),
            )
        self.up2.apply(weights_init_kaiming)

        if with_ibn:
            self.bn = IBN(in_dim, norm_type)
        else:
            self.bn = get_norm(norm_type, in_dim)

        weights_init_kaiming(self.bn)

        self.relu = nn.ReLU()

    def forward(self, x):
        back = self.up2(x)
        back = self.bn(back)
        back = self.relu(back)

        return back

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, with_ibn, norm_type):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        # self.bn = nn.InstanceNorm2d(out_channels, eps=0.001)
        # self.bn = get_norm("syncBN", out_channels)
        if with_ibn:
            self.bn = IBN(out_channels, norm_type)
        else:
            self.bn = get_norm(norm_type, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        weights_init_kaiming(self.conv)
        weights_init_kaiming(self.bn)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class Attach(nn.Module):
    EPSILON = 1e-12

    def __init__(self, input_dim, attn_num, with_ibn, norm_type, attach_activation, embedding_dim=0):
        super(Attach, self).__init__()
        self.pool = GeneralizedMeanPoolingP()
        # self.pool = AdaptiveAvgMaxPool()
        self.embedding_dim = embedding_dim
        self.attach_activation = attach_activation

        if embedding_dim != 0:
            self.embedding_layer = nn.Conv2d(input_dim, embedding_dim, 1, bias=False)
            self.embedding_layer.apply(weights_init_kaiming)

            if self.attach_activation:
                self.bn = nn.ModuleList()

                for i in range(attn_num):
                    if with_ibn:
                        bn = IBN(embedding_dim, norm_type)
                    else:
                        bn = get_norm(norm_type, embedding_dim)
                    weights_init_kaiming(bn)
                    self.bn.append(bn)

                self.relu = nn.ReLU()

    def forward(self, attentions, features):
        feature_matrix = []
        for i in range(attentions.shape[1]):
            AiF = features * attentions[:, i:i + 1, ...]
            if self.embedding_dim !=0:
                AiF = self.embedding_layer(AiF)
                if self.attach_activation:
                    AiF = self.bn[i](AiF)
                    AiF = self.relu(AiF)
            AiF = self.pool(AiF).view(features.shape[0], -1)
            feature_matrix.append(AiF)
        feature_matrix = torch.cat(feature_matrix, dim=1)

        # sign-sqrt
        feature_matrix = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + self.EPSILON)
        # feature_matrix = F.normalize(feature_matrix, dim=-1)
        return feature_matrix.view(feature_matrix.shape[0], feature_matrix.shape[1], 1, 1) # B * (M*C)

class AttnShaped(nn.Module):
    EPSILON = 1e-12

    def __init__(self, input_dim, with_ibn, norm_type, attnShaped_activation):
        super(AttnShaped, self).__init__()

        self.attnShaped_activation = attnShaped_activation

        if self.attnShaped_activation:
            if with_ibn:
                self.bn = IBN(input_dim, norm_type)
            else:
                self.bn = get_norm(norm_type, input_dim)
            self.bn.apply(weights_init_kaiming)

            self.relu = nn.ReLU()

    def forward(self, attentions, features):

        # attentions = torch.mean(attentions, dim=1, keepdim=True)

        attentions = torch.max(attentions, dim=1, keepdim=True).values
        attentions = F.normalize(attentions, dim=(2,3))  # type: ignore
        AiF = features * attentions

        if self.attnShaped_activation:
            AiF = self.bn(AiF)
            AiF = self.relu(AiF)

        return AiF

class Embedding(nn.Module):
    def __init__(self, input_dim, out_dim, with_ibn, norm_type, bias= False):
        super(Embedding, self).__init__()

        self.embedding = nn.Conv2d(input_dim, out_dim, 1, 1, bias=bias)

        if with_ibn:
            self.bn = IBN(out_dim, norm_type)
        else:
            self.bn = get_norm(norm_type, out_dim)

        self.relu = nn.ReLU()

        self.bn.apply(weights_init_kaiming)
        self.embedding.apply(weights_init_kaiming)

    def forward(self, features):
        f = self.embedding(features)
        f = self.bn(f)
        f = self.relu(f)

        return f

class BatchDrop(nn.Module):
    def __init__(self, h_ratio=0.15, w_ratio=1.0):
        super(BatchDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio
    
    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(self.h_ratio * h)
            rw = round(self.w_ratio * w)
            sx = random.randint(0, h-rh)
            sy = random.randint(0, w-rw)
            mask = x.new_ones(x.size())
            mask[:, :, sx:sx+rh, sy:sy+rw] = 0
            x = x * mask
        return x
