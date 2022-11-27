# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy

import torch
from torch import nn
import torch.nn.functional as F
from fastreid.config import configurable
from fastreid.layers import get_norm
from fastreid.layers.batch_norm import IBN
from fastreid.layers.pooling import GlobalAvgPool
from fastreid.layers.weight_init import weights_init_kaiming
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.backbones.resnet import Bottleneck
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY
import numpy as np
import os

@META_ARCH_REGISTRY.register()
class A2MGN(nn.Module):
    """
    Multiple Granularities Network architecture, which contains the following two components:
    1. Per-image feature extraction (aka backbone)
    2. Multi-branch feature aggregation
    """

    @configurable
    def __init__(
            self,
            *,
            output_all,
            output_dir,
            backbone,
            neck1,
            neck2,
            neck3,
            neck4,
            b1_head,
            b2_head,
            b21_head,
            b22_head,
            b3_head,
            b31_head,
            b32_head,
            b33_head,
            b4_head,
            feat_dim,
            attn_num,
            pixel_mean,
            pixel_std,
            loss_kwargs=None
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone:
            neck1:
            neck2:
            neck3:
            b1_head:
            b2_head:
            b21_head:
            b22_head:
            b3_head:
            b31_head:
            b32_head:
            b33_head:
            pixel_mean:
            pixel_std:
            loss_kwargs:
        """

        super().__init__()

        self.backbone = backbone

        # branch1
        self.b1 = neck1
        self.b1_head = b1_head

        # branch2
        self.b2 = neck2
        self.b2_head = b2_head
        self.b21_head = b21_head
        self.b22_head = b22_head

        # branch3
        self.b3 = neck3
        self.b3_head = b3_head
        self.b31_head = b31_head
        self.b32_head = b32_head
        self.b33_head = b33_head

        # branch4
        self.b4 = neck4
        self.attention_maker = BasicConv2d(feat_dim, attn_num)
        self.attach = Attach()
 
        self.b4_head = b4_head

        self.loss_kwargs = loss_kwargs
        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)

        self.output_all = output_all
        self.output_dir = output_dir

        if self.output_all:
            self.dir = self.output_dir + "/attn_" + str(attn_num) + "/"
            self.image_dir = self.dir + "raw_image/"
            self.attn_dir = self.dir + "raw_attention/"
            self.feat_dir = self.dir + "raw_feat/"

            if not os.path.isdir(self.dir):
                os.mkdir(self.dir)
            if not os.path.isdir(self.image_dir):
                os.mkdir(self.image_dir)
            if not os.path.isdir(self.attn_dir):
                os.mkdir(self.attn_dir)
            if not os.path.isdir(self.feat_dir):
                os.mkdir(self.feat_dir)

    @classmethod
    def from_config(cls, cfg):
        feat_dim = cfg.MODEL.BACKBONE.FEAT_DIM
        bn_norm = cfg.MODEL.BACKBONE.NORM
        with_se = cfg.MODEL.BACKBONE.WITH_SE
        attn_num      = cfg.MODEL.HEADS.ATTN
        embedding_dim = cfg.MODEL.HEADS.EMBEDDING_DIM
        all_blocks = build_backbone(cfg)

        # backbone
        backbone = nn.Sequential(
            all_blocks.conv1,
            all_blocks.bn1,
            all_blocks.relu,
            all_blocks.maxpool,
            all_blocks.layer1,
            all_blocks.layer2,
            all_blocks.layer3[0]
        )
        res_conv4 = nn.Sequential(*all_blocks.layer3[1:])
        res_g_conv5 = all_blocks.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, bn_norm, False, with_se, downsample=nn.Sequential(
                nn.Conv2d(1024, 2048, 1, bias=False), get_norm(bn_norm, 2048))),
            Bottleneck(2048, 512, bn_norm, False, with_se),
            Bottleneck(2048, 512, bn_norm, False, with_se))
        res_p_conv5.load_state_dict(all_blocks.layer4.state_dict())

        # branch
        neck1 = nn.Sequential(
            copy.deepcopy(res_conv4),
            copy.deepcopy(res_g_conv5)
        )
        b1_head = build_heads(cfg)

        # branch2
        neck2 = nn.Sequential(
            copy.deepcopy(res_conv4),
            copy.deepcopy(res_p_conv5)
        )
        b2_head = build_heads(cfg)
        b21_head = build_heads(cfg)
        b22_head = build_heads(cfg)

        # branch3
        neck3 = nn.Sequential(
            copy.deepcopy(res_conv4),
            copy.deepcopy(res_p_conv5)
        )
        b3_head = build_heads(cfg)
        b31_head = build_heads(cfg)
        b32_head = build_heads(cfg)
        b33_head = build_heads(cfg)

        # branch4
        neck4 = nn.Sequential(
            copy.deepcopy(res_conv4),
            copy.deepcopy(res_p_conv5)
        )
        b4_head = build_heads(cfg, dim_spec=feat_dim*attn_num, embedding_spec=embedding_dim*attn_num,
                            pool_spec="Identity")

        return {
            'output_all': cfg.OUTPUT_ALL,
            'output_dir': cfg.OUTPUT_DIR,
            'backbone': backbone,
            'neck1': neck1,
            'neck2': neck2,
            'neck3': neck3,
            'neck4': neck4,
            'b1_head': b1_head,
            'b2_head': b2_head,
            'b21_head': b21_head,
            'b22_head': b22_head,
            'b3_head': b3_head,
            'b31_head': b31_head,
            'b32_head': b32_head,
            'b33_head': b33_head,
            'b4_head': b4_head,
            'feat_dim': cfg.MODEL.BACKBONE.FEAT_DIM,
            "attn_num": attn_num,
            'pixel_mean': cfg.MODEL.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.PIXEL_STD,
            'loss_kwargs':
                {
                    # loss name
                    'loss_names': cfg.MODEL.LOSSES.NAME,

                    # loss hyperparameters
                    'ce': {
                        'eps': cfg.MODEL.LOSSES.CE.EPSILON,
                        'alpha': cfg.MODEL.LOSSES.CE.ALPHA,
                        'scale': cfg.MODEL.LOSSES.CE.SCALE
                    },
                    'tri': {
                        'margin': cfg.MODEL.LOSSES.TRI.MARGIN,
                        'norm_feat': cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                        'hard_mining': cfg.MODEL.LOSSES.TRI.HARD_MINING,
                        'scale': cfg.MODEL.LOSSES.TRI.SCALE
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
        features = self.backbone(images)  # (bs, 2048, 16, 8)

        # branch1
        b1_feat = self.b1(features)

        # branch2
        b2_feat = self.b2(features)
        b21_feat, b22_feat = torch.chunk(b2_feat, 2, dim=2)

        # branch3
        b3_feat = self.b3(features)
        b31_feat, b32_feat, b33_feat = torch.chunk(b3_feat, 3, dim=2)

        # branch4
        b4_feat = self.b4(features)
        attention = self.attention_maker(b4_feat)
        b4_feat = self.attach(attention, b4_feat)
        #b4_feat = features

        assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
        targets = batched_inputs["targets"]
        
        if self.training:
            if targets.sum() < 0: targets.zero_()

            b1_outputs = self.b1_head(b1_feat, targets)
            b2_outputs = self.b2_head(b2_feat, targets)
            b21_outputs = self.b21_head(b21_feat, targets)
            b22_outputs = self.b22_head(b22_feat, targets)
            b3_outputs = self.b3_head(b3_feat, targets)
            b31_outputs = self.b31_head(b31_feat, targets)
            b32_outputs = self.b32_head(b32_feat, targets)
            b33_outputs = self.b33_head(b33_feat, targets)
            
            b4_outputs = self.b4_head(b4_feat, targets)

            losses = self.losses(b1_outputs,
                                 b2_outputs, b21_outputs, b22_outputs,
                                 b3_outputs, b31_outputs, b32_outputs, b33_outputs,
                                 b4_outputs,
                                 targets)
            return losses
        else:
            if self.output_all:
                img_paths = batched_inputs['img_paths']

                for i in range(len(img_paths)):
                    (path, filename) = os.path.split(img_paths[i])
                    np.save(self.attn_dir+filename.split(".")[0]+"_a4.npy", attention[i].detach().cpu().numpy())
                    np.save(self.attn_dir+filename.split(".")[0]+"_a3.npy", attention[i].detach().cpu().numpy())
                    np.save(self.image_dir+filename.split(".")[0]+".npy", batched_inputs['images'][i].detach().cpu().numpy())

            b1_pool_feat = self.b1_head(b1_feat)
            b2_pool_feat = self.b2_head(b2_feat)
            b21_pool_feat = self.b21_head(b21_feat)
            b22_pool_feat = self.b22_head(b22_feat)
            b3_pool_feat = self.b3_head(b3_feat)
            b31_pool_feat = self.b31_head(b31_feat)
            b32_pool_feat = self.b32_head(b32_feat)
            b33_pool_feat = self.b33_head(b33_feat)
            b4_pool_feat = self.b4_head(b4_feat)

            pred_feat = torch.cat([b1_pool_feat, b2_pool_feat, b3_pool_feat, b21_pool_feat,
                                   b22_pool_feat, b31_pool_feat, b32_pool_feat, b33_pool_feat,b4_pool_feat], dim=1)

            return pred_feat

    def preprocess_image(self, batched_inputs):
        r"""
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs["images"].to(self.device)
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs.to(self.device)
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self,
               b1_outputs,
               b2_outputs, b21_outputs, b22_outputs,
               b3_outputs, b31_outputs, b32_outputs, b33_outputs,
               b4_outputs,
               gt_labels):
        # model predictions
        # fmt: off
        pred_class_logits = b1_outputs['pred_class_logits'].detach()
        b1_logits         = b1_outputs['cls_outputs']
        b2_logits         = b2_outputs['cls_outputs']
        b21_logits        = b21_outputs['cls_outputs']
        b22_logits        = b22_outputs['cls_outputs']
        b3_logits         = b3_outputs['cls_outputs']
        b31_logits        = b31_outputs['cls_outputs']
        b32_logits        = b32_outputs['cls_outputs']
        b33_logits        = b33_outputs['cls_outputs']
        b4_logits         = b4_outputs['cls_outputs']


        b1_pool_feat      = b1_outputs['features']
        b2_pool_feat      = b2_outputs['features']
        b3_pool_feat      = b3_outputs['features']
        b21_pool_feat     = b21_outputs['features']
        b22_pool_feat     = b22_outputs['features']
        b31_pool_feat     = b31_outputs['features']
        b32_pool_feat     = b32_outputs['features']
        b33_pool_feat     = b33_outputs['features']
        b4_pool_feat      = b4_outputs['features']
        # fmt: on

        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        b22_pool_feat = torch.cat((b21_pool_feat, b22_pool_feat), dim=1)
        b33_pool_feat = torch.cat((b31_pool_feat, b32_pool_feat, b33_pool_feat), dim=1)

        loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']

        if "CrossEntropyLoss" in loss_names:
            ce_kwargs = self.loss_kwargs.get('ce')
            loss_dict['loss_cls_b1'] = cross_entropy_loss(
                b1_logits,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale') / 9.0

            loss_dict['loss_cls_b2'] = cross_entropy_loss(
                b2_logits,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale') / 9.0

            loss_dict['loss_cls_b21'] = cross_entropy_loss(
                b21_logits,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale') / 9.0

            loss_dict['loss_cls_b22'] = cross_entropy_loss(
                b22_logits,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale') / 9.0

            loss_dict['loss_cls_b3'] = cross_entropy_loss(
                b3_logits,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale') / 9.0

            loss_dict['loss_cls_b31'] = cross_entropy_loss(
                b31_logits,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale') / 9.0

            loss_dict['loss_cls_b32'] = cross_entropy_loss(
                b32_logits,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale') / 9.0

            loss_dict['loss_cls_b33'] = cross_entropy_loss(
                b33_logits,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale') / 9.0

            loss_dict['loss_cls_b4'] = cross_entropy_loss(
                b4_logits,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale') / 9.0

        if "TripletLoss" in loss_names:
            tri_kwargs = self.loss_kwargs.get('tri')
            loss_dict['loss_triplet_b1'] = triplet_loss(
                b1_pool_feat,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale') / 6.0

            loss_dict['loss_triplet_b2'] = triplet_loss(
                b2_pool_feat,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale') / 6.0

            loss_dict['loss_triplet_b3'] = triplet_loss(
                b3_pool_feat,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale') / 6.0

            loss_dict['loss_triplet_b4'] = triplet_loss(
                b4_pool_feat,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale') / 6.0

            loss_dict['loss_triplet_b22'] = triplet_loss(
                b22_pool_feat,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale') / 6.0

            loss_dict['loss_triplet_b33'] = triplet_loss(
                b33_pool_feat,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale') / 6.0

        return loss_dict

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False,)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class Attach(nn.Module):
    EPSILON = 1e-12

    def __init__(self):
        super(Attach, self).__init__()
        self.pool = GlobalAvgPool()

    def forward(self, attentions, features):
        feature_matrix = []
        for i in range(attentions.shape[1]):
            AiF = features * attentions[:, i:i + 1, ...]
            AiF = self.pool(AiF).view(features.shape[0], -1)
            feature_matrix.append(AiF)
        feature_matrix = torch.cat(feature_matrix, dim=1)

        # sign-sqrt
        feature_matrix = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + self.EPSILON)

        return feature_matrix.view(feature_matrix.shape[0], feature_matrix.shape[1], 1, 1) # B * (M*C)
