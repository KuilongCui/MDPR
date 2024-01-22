# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import copy
import os
import random
import numpy as np
from fastreid.layers import pooling

from fastreid.layers.batch_norm import get_norm
from fastreid.layers.pooling import GeneralizedMeanPoolingP
from fastreid.layers.weight_init import weights_init_kaiming
import torch
from torch import nn
import torch.nn.functional as F
from fastreid.config import configurable
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.losses import *
from fastreid.modeling.meta_arch.build import META_ARCH_REGISTRY
from fastreid.modeling.heads import build_heads

@META_ARCH_REGISTRY.register()  # type: ignore
class MDPREX(nn.Module):
    """
    Baseline architecture. Any models that contains the following two components:
    1. Per-image feature extraction (aka backbone)
    2. Per-image feature aggregation and loss computation
    """

    @configurable
    def __init__(
            self, *, output_all, output_dir, attn_guide, pixel_mean, pixel_std, distillation_able, attn_diversity_loss_enable, \
            feat_dim, backbone_embedding_dim, attn_num, backbone, attn_backbone, attn_feature4_head, attn_feature3_head, \
            attn_bap_feature4_head, attn_bap_feature3_head, part_backbone, hard_global_head, part_num, hard_part_head, \
            fusion_enable, fusion_feat_head, loss_kwargs
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

        self.attn_guide = attn_guide
        self.distillation_able = distillation_able
        self.fusion_enable = fusion_enable
        self.attn_diversity_loss_enable = attn_diversity_loss_enable

        # ------------- pre-process -------------
        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)

        # ------------- features and attn generator -------------
        self.generate = Generate(feat_dim, backbone_embedding_dim, attn_num, part_num, attn_guide, \
                                backbone, part_backbone, attn_backbone)

        # ------------- hard content branch -----------------------
        self.hard_global_head = hard_global_head
        self.part_num = part_num
        self.hard_part_head = hard_part_head

        # ------------- soft content branch -----------------------
        self.attn_feature4_head = attn_feature4_head
        self.attn_feature3_head = attn_feature3_head

        self.attn_bap_feature4_head = attn_bap_feature4_head
        self.attn_bap_feature3_head = attn_bap_feature3_head

        # ------------- fusion -----------------------
        if self.fusion_enable:
            self.fusion = Fusion(backbone_embedding_dim)
            self.fusion_feat_head = fusion_feat_head

        # ------------- mutual distilltion mlp -----------------------
        if self.distillation_able:
            self.predictor1 = DomainTransfer(backbone_embedding_dim, backbone_embedding_dim)
            self.predictor2 = DomainTransfer(backbone_embedding_dim, backbone_embedding_dim)

        # ------------- loss -----------------------
        self.loss_kwargs = loss_kwargs

        # ------------- output attn heatmap -------------
        self.output_all = output_all
        self.output_dir = output_dir

    @classmethod
    def from_config(cls, cfg):
        backbone_embedding_dim = cfg.MODEL.BACKBONE.EMBEDDING_DIM

        # ------------- backbone -----------------------
        all_blocks = build_backbone(cfg)
        backbone = nn.Sequential(
            all_blocks.conv1,
            all_blocks.bn1,
            all_blocks.relu,
            all_blocks.maxpool,
            all_blocks.layer1,
            all_blocks.layer2,
        )
        res_conv4 = all_blocks.layer3
        res_conv5 = all_blocks.layer4

        # ------------- soft content branch -----------------------
        num_attn = cfg.ATTN_NUM

        attn_backbone = nn.Sequential(
            copy.deepcopy(res_conv4),
            copy.deepcopy(res_conv5)
        )
        
        attn_feature4_head = build_heads(cfg)
        attn_feature3_head = build_heads(cfg)
        attn_bap_feature4_head = build_heads(cfg, dim_spec=backbone_embedding_dim*num_attn, \
                                embedding_spec=0, pool_spec="Identity")
        attn_bap_feature3_head = build_heads(cfg, dim_spec=backbone_embedding_dim*num_attn,\
                                        embedding_spec=0, pool_spec="Identity")

        # ------------- hard content branch -----------------------
        part_backbone = nn.Sequential(
            copy.deepcopy(res_conv4),
            copy.deepcopy(res_conv5)
        )
        hard_global_head = build_heads(cfg)

        hard_part_head = nn.ModuleList()
        for i in range(cfg.PART_NUM):
            hard_part_head.append(build_heads(cfg))

        # ------------- fusion -----------------------
        fusion_feat_head = build_heads(cfg, dim_spec=backbone_embedding_dim * 4)

        return {
            # --------- parameters ---------
            'output_all': cfg.OUTPUT_ALL,
            'output_dir': cfg.OUTPUT_DIR,
            'pixel_mean': cfg.MODEL.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.PIXEL_STD,
            'distillation_able': cfg.DISTILLATION_ENABLE,
            'attn_diversity_loss_enable': cfg.ATTN_DIVERSITY_LOSS_ENABLE,
            
            # --------- config ---------
            'attn_guide': cfg.ATTN_GUIDED,
            'feat_dim': cfg.MODEL.BACKBONE.FEAT_DIM,
            'backbone_embedding_dim': cfg.MODEL.BACKBONE.EMBEDDING_DIM,
            'attn_num': cfg.ATTN_NUM,
            'backbone': backbone,

            # --------- soft content branch ---------
            'attn_backbone': attn_backbone,
            'attn_feature4_head': attn_feature4_head,
            'attn_feature3_head': attn_feature3_head,
            'attn_bap_feature4_head': attn_bap_feature4_head,
            'attn_bap_feature3_head': attn_bap_feature3_head,

            # --------- hard content branch ---------
            'part_backbone': part_backbone,
            'hard_global_head': hard_global_head,
            'part_num': cfg.PART_NUM,
            'hard_part_head': hard_part_head,
            
            # --------- fusion ---------
            'fusion_enable': cfg.FUSION_ENABLE,
            'fusion_feat_head': fusion_feat_head,

            # --------- loss ---------
            'loss_kwargs':
                {
                    'loss_names': cfg.MODEL.LOSSES.NAME,
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
                    'attn_diversity': {
                        'scale': cfg.MODEL.LOSSES.ATTN_DIVERSITY.SCALE,
                    }
                }
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)

        hard_global_feat, hard_part_feat, attn_feature4, attn_feature3, \
            attn_bap_feature3, attn_bap_pool_feature3, attn_bap_feature4, attn_bap_pool_feature4, a4, a3 = self.generate(images)
        
        if self.fusion_enable:
            fusion_feat = self.fusion(hard_global_feat, hard_part_feat, attn_feature4, attn_feature3)

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"]

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            losses = {}

            # ------------- hard content branch -------------
            outputs_hard_global = self.hard_global_head(hard_global_feat, targets, batched_inputs=batched_inputs)
            self.losses(outputs_hard_global, targets, self.loss_kwargs['loss_names'], "hard_global", losses)

            for i in range(self.part_num):
                outputs_hard_part = self.hard_part_head[i](hard_part_feat[i], targets, batched_inputs=batched_inputs)
                self.losses(outputs_hard_part, targets, self.loss_kwargs['loss_names'], "hard_part_"+str(i), losses)

            # ------------- soft content branch -------------
            attn_feature4_output = self.attn_feature4_head(attn_feature4, targets, batched_inputs=batched_inputs)
            self.losses(attn_feature4_output, targets, self.loss_kwargs['loss_names'], "attn_feature4", losses)

            attn_bap_feature4_output = self.attn_bap_feature4_head(attn_bap_feature4, targets, batched_inputs=batched_inputs)
            self.losses(attn_bap_feature4_output, targets, 'CrossEntropyLoss', "attn_bap4", losses, log=True)

            if self.attn_diversity_loss_enable:
                losses['loss_diversity_loss_4'] = cross_catgory_loss(attn_bap_pool_feature4)
                losses['loss_diversity_loss_reg_4'] = torch.sum(a4) / a4.shape[0] * self.loss_kwargs.get('attn_diversity').get('scale')

            attn_feature3_output = self.attn_feature3_head(attn_feature3, targets, batched_inputs=batched_inputs)
            self.losses(attn_feature3_output, targets, self.loss_kwargs['loss_names'], "attn_feature3", losses)
            
            attn_bap_feature3_output = self.attn_bap_feature3_head(attn_bap_feature3, targets, batched_inputs=batched_inputs)
            self.losses(attn_bap_feature3_output, targets, 'CrossEntropyLoss', "attn_bap3", losses)
            
            if self.attn_diversity_loss_enable:
                losses['loss_diversity_loss_3'] = cross_catgory_loss(attn_bap_pool_feature3)
                losses['loss_diversity_loss_reg_3'] = torch.sum(a3) / a3.shape[0] * self.loss_kwargs.get('attn_diversity').get('scale')

            # ------------- fusion -------------
            if self.fusion_enable:
                fusion_feat_ouput = self.fusion_feat_head(fusion_feat, targets, batched_inputs=batched_inputs)
                self.losses(fusion_feat_ouput, targets, self.loss_kwargs['loss_names'], "fusion", losses)
            
            # ------------- mututal distillation loss -------------
            if self.distillation_able:
                z1, z2 = outputs_hard_global['features'], attn_feature4_output['features']
                p1, p2 = self.predictor1(z1), self.predictor2(z2)

                losses['loss_mutual_distillation'] = -(torch.cosine_similarity(p1, z2.detach()).mean() + \
                                                    torch.cosine_similarity(p2, z1.detach()).mean()) * 0.5

            return losses
        else:
            self.may_visualize(batched_inputs, a4, a3)
                        
            eval_feat = []

            # ------------- hard content branch -------------
            outputs2 = self.hard_global_head(hard_global_feat, batched_inputs=batched_inputs)
            eval_feat.append(outputs2)

            for i in range(self.part_num):
                outputs_part = self.hard_part_head[i](hard_part_feat[i], batched_inputs=batched_inputs)
                eval_feat.append(outputs_part)

            # ------------- soft content branch -------------
            attn_feature4_output = self.attn_feature4_head(attn_feature4, batched_inputs=batched_inputs)
            eval_feat.append(attn_feature4_output)
            attn_feature3_ouput = self.attn_feature3_head(attn_feature3, batched_inputs=batched_inputs)
            eval_feat.append(attn_feature3_ouput)

            # ------------- fusion -------------
            if self.fusion_enable:
                fusion_feat_ouput = self.fusion_feat_head(fusion_feat, batched_inputs=batched_inputs)
                eval_feat.append(fusion_feat_ouput)

            eval_feat = torch.cat(eval_feat, dim=1)

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
    
    def may_visualize(self, batched_inputs, a4, a3):
        if self.output_all:
            # ------------- legality check -------------
            dir = os.path.join(self.output_dir,"heatmap_attn")
            image_dir = os.path.join(dir, "raw_image")
            attn_dir = os.path.join(dir, "raw_attention")

            if not os.path.isdir(dir):
                os.mkdir(dir)
            if not os.path.isdir(image_dir):
                os.mkdir(image_dir)
            if not os.path.isdir(attn_dir):
                os.mkdir(attn_dir)

            # ------------- output attn heatmap -------------
            img_paths = batched_inputs['img_paths']

            for i in range(len(img_paths)):
                (path, filename) = os.path.split(img_paths[i])
                np.save(os.path.join(attn_dir, filename.split(".")[0]+"_a4.npy"), a4[i].detach().cpu().numpy())
                np.save(os.path.join(attn_dir, filename.split(".")[0]+"_a3.npy"), a3[i].detach().cpu().numpy())
                np.save(os.path.join(image_dir, filename.split(".")[0]+".npy"), batched_inputs['images'][i].detach().cpu().numpy())

    def losses(self, outputs, gt_labels, loss_names, prefix="", loss=None, log=False, scale=1.):
            """
            Compute loss from modeling's outputs, the loss function input arguments
            must be the same as the outputs of the model forwarding.
            """
            # model predictions
            # fmt: off      
            cls_outputs       = outputs['cls_outputs']
            pred_features     = outputs['features']
            # fmt: on

            # Log prediction accuracy
            if log:
                pred_class_logits = outputs['pred_class_logits'].detach()
                log_accuracy(pred_class_logits, gt_labels)

            loss_dict = {}
            if loss is not None:
                loss_dict = loss

            if 'CrossEntropyLoss' in loss_names:
                ce_kwargs = self.loss_kwargs.get('ce')
                loss_dict['loss_'+prefix+'_cls'] = cross_entropy_loss(
                    cls_outputs,
                    gt_labels,
                    ce_kwargs.get('eps'),
                    ce_kwargs.get('alpha')
                ) * ce_kwargs.get('scale') * scale

            if 'TripletLoss' in loss_names:
                tri_kwargs = self.loss_kwargs.get('tri')
                loss_dict['loss_'+prefix+'_triplet'] = triplet_loss(
                    pred_features,
                    gt_labels,
                    tri_kwargs.get('margin'),
                    tri_kwargs.get('norm_feat'),
                    tri_kwargs.get('hard_mining')
                ) * tri_kwargs.get('scale') * scale

class Generate(nn.Module):
    def __init__(self, feat_dim, backbone_embedding_dim, attn_num, part_num, \
                    attn_guide, backbone, backbone_part, backbone_attn):
        super(Generate, self).__init__()

        self.valid_attn = attn_num + 1
        self.attn_num = attn_num
        self.part_num = part_num
        self.attn_guide = attn_guide

        # ------------- backbone -----------------------
        self.backbone = backbone

        # ------------- hard content branch -----------------------
        self.backbone_part = backbone_part

        self.embedding_hard_global = Embedding(feat_dim, backbone_embedding_dim)
        self.embedding_hard_part = nn.ModuleList()
        for i in range(part_num):
            self.embedding_hard_part.append(Embedding(feat_dim, backbone_embedding_dim))

        # ------------- soft content branch -----------------------
        self.backbone_attn = backbone_attn

        self.embedding_attn_feature4 = Embedding(feat_dim, backbone_embedding_dim)

        self.em3_1024 = Embedding(feat_dim//2, feat_dim//2)
        self.embedding_attn_feature3 = Embedding(feat_dim//2, backbone_embedding_dim) 

        self.attention_maker4 = BasicConv2d(feat_dim, self.valid_attn)
        
        if attn_guide:
            self.up_attn_4_3 = PamUpSamper(self.valid_attn, self.valid_attn, bias=False, scale=1.)
            self.attention_maker3 = BasicConv2d(feat_dim//2 + self.valid_attn, self.valid_attn)
        else:
            self.attention_maker3 = BasicConv2d(feat_dim//2, self.valid_attn)

        self.attn_bap3 = Bap(feat_dim//2, self.attn_num, embedding_dim=backbone_embedding_dim)
        self.attn_bap4 = Bap(feat_dim, self.attn_num, embedding_dim=backbone_embedding_dim)

    def forward(self, images):
        features = self.backbone(images)

        # ------------- hard content branch -------------
        hard_global_feat = self.backbone_part(features)
        hard_chunk_feat = torch.chunk(hard_global_feat, self.part_num, dim=2)

        hard_global_feat = self.embedding_hard_global(hard_global_feat)
        hard_part_feat = []
        for i in range(self.part_num):
            hard_part_feat.append(self.embedding_hard_part[i](hard_chunk_feat[i]))

        # ------------- soft content branch -------------
        x3 = self.backbone_attn[0](features)
        attn_g_feat = self.backbone_attn[1](x3)
        embedding_attn_feature4 = self.embedding_attn_feature4(attn_g_feat)
        
        a4 = self.attention_maker4(attn_g_feat)

        if self.attn_guide:
            new_a3 = self.up_attn_4_3(a4)
            new_x3 = torch.cat((x3, new_a3), dim=1)
            a3 = self.attention_maker3(new_x3)
        else:
            a3 = self.attention_maker3(x3)

        x3 = self.em3_1024(x3)
        a3 = self.softmax_attn(a3)
        attn_bap_feature3, attn_bap_pool_feature3 = self.attn_bap3(a3, x3)
        embedding_attn_feature3 = self.embedding_attn_feature3(x3)

        a4 = self.softmax_attn(a4)
        attn_bap_feature4, attn_bap_pool_feature4 = self.attn_bap4(a4, attn_g_feat)
        
        return hard_global_feat, hard_part_feat, embedding_attn_feature4, embedding_attn_feature3, \
            attn_bap_feature3, attn_bap_pool_feature3, attn_bap_feature4, attn_bap_pool_feature4, a4, a3

    def softmax_attn(self, attn):
        assert attn.shape[1] == self.valid_attn
        attn = torch.softmax(attn, dim=1)
        return attn[:, :self.attn_num]

class Fusion(nn.Module):
    def __init__(self, feat_dim) -> None:
        super(Fusion, self).__init__()
        
        self.merge = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 1, 1, bias=False),
            nn.Conv2d(feat_dim, feat_dim, 3, 1, 1, bias=False),
            
            get_norm("BN", feat_dim),
            nn.ReLU()
        )
        self.merge.apply(weights_init_kaiming)
        
        self.output_part = nn.Sequential(
            nn.Conv2d(feat_dim * 2, feat_dim * 2, 1, 1, bias=False),
            nn.Conv2d(feat_dim * 2, feat_dim * 2, 3, 1, 1, bias=False),
            get_norm("BN", feat_dim * 2),
            nn.ReLU()
        )
        self.output_part.apply(weights_init_kaiming)

        self.output_attn = nn.Sequential(
            nn.Conv2d(feat_dim * 2, feat_dim * 2, 1, 1, bias=False),
            nn.Conv2d(feat_dim * 2, feat_dim * 2, 3, 1, 1, bias=False),
            get_norm("BN", feat_dim * 2),
            nn.ReLU()
        )
        self.output_attn.apply(weights_init_kaiming)

        self.output = nn.Sequential(
            nn.Conv2d(feat_dim * 4, feat_dim * 4, 1, 1, bias=False),
            nn.Conv2d(feat_dim * 4, feat_dim * 4, 3, 1, 1, bias=False),
            get_norm("BN", feat_dim * 4),
            nn.ReLU()
        )
        self.output.apply(weights_init_kaiming)

    def forward(self, hard_global, hard_part, attn_feature4, attn_feature3):
        hard_part_merge = self.merge(torch.cat(hard_part, dim=2))
        part_merge =  self.output_part(torch.cat([hard_part_merge, hard_global], dim=1))
        attn_merge = self.output_attn(torch.cat([attn_feature4, attn_feature3], dim=1))
        return self.output(torch.cat([part_merge, attn_merge], dim=1))

class PamUpSamper(nn.Module):
    def __init__(self, in_dim, output_dim, bias=False, scale=2):
        super(PamUpSamper, self).__init__()

        self.scale = scale
        self.out_dim = output_dim
        
        self.upsapmle = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_dim, output_dim, 1, bias=bias)
        self.conv.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.upsapmle(x)
        x = self.conv(x)
        return F.relu(x, inplace=True)
        
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicConv2d, self).__init__()
        self.attn_num = out_channels

        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv.apply(weights_init_kaiming)

        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.conv3.apply(weights_init_kaiming)

        self.bn = get_norm("BN", out_channels)
        self.bn.apply(weights_init_kaiming)

    def forward(self, x):
        x1 = self.conv(x)
        x3 = self.conv3(x)
        x = self.bn(x1 + x3)
        return F.relu(x, inplace=True)
    
class Bap(nn.Module):
    EPSILON = 1e-12

    def __init__(self, input_dim, attn_num, embedding_dim=0):
        super(Bap, self).__init__()
        self.pool = GeneralizedMeanPoolingP()

        self.embedding_dim = embedding_dim

        if embedding_dim != 0:
            self.embedding_layer = nn.ModuleList()
            for i in range(attn_num):
                self.embedding_layer.append(Embedding(input_dim, embedding_dim))
            self.embedding_layer.apply(weights_init_kaiming)
            
    def forward(self, attentions, features):
        feature_matrix = []
        pool_feat = []

        for i in range(attentions.shape[1]):
            AiF = features * attentions[:, i:i + 1, ...]

            if self.embedding_dim !=0:
                AiF = self.embedding_layer[i](AiF)

            pool_AiF = self.pool(AiF)
            pool_feat.append(pool_AiF)

            AiF = pool_AiF.view(features.shape[0], -1)
            feature_matrix.append(AiF)

        feature_matrix = torch.cat(feature_matrix, dim=1)

        feature_matrix = F.relu(feature_matrix)

        return feature_matrix.view(feature_matrix.shape[0], feature_matrix.shape[1], 1, 1), pool_feat # B * (M*C)

class Embedding(nn.Module):
    def __init__(self, input_dim, out_dim, bias=False, bias_freeze=False):
        super(Embedding, self).__init__()

        self.embedding = nn.Conv2d(input_dim, out_dim, 1, 1, bias=bias)
        self.embedding.apply(weights_init_kaiming)

        self.bn = get_norm("BN", out_dim, bias_freeze=bias_freeze)
        self.bn.apply(weights_init_kaiming)

    def forward(self, features):
        f = self.embedding(features)
        f = self.bn(f)
        return F.relu(f, inplace=True)

class DomainTransfer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(DomainTransfer, self).__init__()

        self.conv = nn.Sequential(nn.Linear(in_dim, out_dim, bias=False),
                                    nn.BatchNorm1d(out_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(out_dim, out_dim, bias=bias))
        self.conv.apply(weights_init_kaiming)
    
    def forward(self, x):
        return self.conv(x)

def cross_catgory_loss(x, gamma=1.):
    nparts = len(x)
    corr_matrix = torch.zeros(nparts, nparts)

    for i in range(nparts):
        x[i] = x[i].squeeze()
        x[i] = torch.div(x[i], x[i].norm(dim=1, keepdim=True))

    for i in range(nparts):
        for j in range(nparts):
            corr_matrix[i, j] = torch.mean(torch.sum(x[i] * x[j],  dim=-1))
            if i == j:
                corr_matrix[i, j] = 0.

    loss = torch.mul(torch.sum(torch.triu(corr_matrix)), gamma)

    return loss.cuda()

def diversity_loss(x, gamma=1.):
    loss = torch.zeros([1]).cuda()
    for i in range(len(x)):
        x[i] = x[i].squeeze()
        x[i] = torch.div(x[i], x[i].norm(dim=1, keepdim=True))

    for i in range(len(x)):
        for j in range(i+1, len(x)):
            loss = loss + torch.mean(torch.sum(x[i]*x[j], dim=1))

    loss = torch.mul(loss, gamma)

    return loss.cuda()
