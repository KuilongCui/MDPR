# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import copy

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

@META_ARCH_REGISTRY.register()
class OsNetSbaseline(nn.Module):
    """
    Baseline architecture. Any models that contains the following two components:
    1. Per-image feature extraction (aka backbone)
    2. Per-image feature aggregation and loss computation
    """

    @configurable
    def __init__(
            self,
            *,
            backbone1,
            b1_head,
            backbone2,
            b2_head,
            backbone3,
            b3_head,
            attn_head,
            last_stride,
            backbone_embedding_dim,
            feat_dim,
            embedding_dim,
            num_class,
            num_attn,
            backbone,
            heads,
            pixel_mean,
            pixel_std,
            loss_kwargs=None
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

        self.loss_kwargs = loss_kwargs

        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)

        self.num_attn = num_attn
        self.feat_dim = feat_dim

        if last_stride == 1:
            self.upSamper3 = nn.Sequential(
                nn.Conv2d(self.num_attn, self.num_attn, 1),
                # nn.BatchNorm2d(self.num_attn),
                IBN(self.num_attn, "syncBN"),
                nn.ReLU()
            )
            self.upSamper3.apply(weights_init_kaiming)
        else:
            self.upSamper3 = UpSamper(in_dim=self.num_attn, output_dim=self.num_attn)
        
        self.upSamper2 = UpSamper(in_dim=self.num_attn, output_dim=self.num_attn, scale=1)
        self.upSamper1 = UpSamper(in_dim=self.num_attn, output_dim=self.num_attn)
        # self.upSamper0 = nn.Sequential(
        #     UpSamper(in_dim=self.num_attn, output_dim=self.num_attn),
        #     UpSamper(in_dim=self.in_planes//16, output_dim=64),
        # )
        
        embedding_dim = backbone_embedding_dim
        self.attention_maker = BasicConv2d(self.feat_dim, num_attn)
        self.attach = Attach(self.feat_dim, embedding_dim=embedding_dim)
        self.attnShaped3 = AttnShaped(self.feat_dim)
        self.attnShaped2 = AttnShaped(384)
        self.attnShaped1 = AttnShaped(self.feat_dim//2)

        self.backbone1 = backbone1
        self.head1 = b1_head
        self.backbone2 = backbone2
        self.head2 = b2_head
        self.backbone3 = backbone3
        self.head3 = b3_head
        self.attn_head = attn_head

        
        self.embedding4 = nn.Sequential(
            nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False),
            # get_norm("syncBN", embedding_dim),
            IBN(embedding_dim, "syncBN"),
            nn.ReLU()
        )
        self.embedding3 = nn.Sequential(
            nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False),
            # get_norm("syncBN", embedding_dim),
            IBN(embedding_dim, "syncBN"),
            nn.ReLU()
        )
        self.embedding2 = nn.Sequential(
            nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False),
            # get_norm("syncBN", embedding_dim),
            IBN(embedding_dim, "syncBN"),
            nn.ReLU()
        )
        self.embedding1 = nn.Sequential(
            nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False),
            IBN(embedding_dim, "syncBN"),
            # get_norm("syncBN", embedding_dim),
            nn.ReLU()
        )

        if 'CenterLoss' in self.loss_kwargs['loss_names']:
            dim = feat_dim if embedding_dim == 0 else embedding_dim
            self.center_loss = CenterLoss(num_classes=num_class, feat_dim=dim)

        if 'KlLoss' in self.loss_kwargs['loss_names']:
            self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        heads = build_heads(cfg)

        all_blocks = backbone

        if cfg.MODEL.BACKBONE.NAME == 'build_resnet_backbone':
            backbone1 = nn.Sequential(
                copy.deepcopy(all_blocks.layer2),
                copy.deepcopy(all_blocks.layer3),
                copy.deepcopy(all_blocks.layer4)
            )
            backbone2 = nn.Sequential(
                copy.deepcopy(all_blocks.layer3),
                copy.deepcopy(all_blocks.layer4)
            )
            backbone3 = nn.Sequential(
                copy.deepcopy(all_blocks.layer4)
            )
        elif cfg.MODEL.BACKBONE.NAME == 'build_osnet_backbone':
            backbone1 = nn.Sequential(
                copy.deepcopy(all_blocks.conv3),
                copy.deepcopy(all_blocks.conv4),
                copy.deepcopy(all_blocks.conv5)
            )
            backbone2 = nn.Sequential(
                copy.deepcopy(all_blocks.conv4),
                copy.deepcopy(all_blocks.conv5)
            )
            backbone3 = nn.Sequential(
                copy.deepcopy(all_blocks.conv5)
            )

        b1_head = build_heads(cfg)
        b2_head = build_heads(cfg)
        b3_head = build_heads(cfg)
        attn_head = build_heads(cfg, True)

        return {
            'backbone1': backbone1,
            'b1_head': b1_head,
            'backbone2': backbone2,
            'b2_head': b2_head,
            'backbone3': backbone3,
            'b3_head': b3_head,
            'attn_head': attn_head,
            'last_stride': cfg.MODEL.BACKBONE.LAST_STRIDE,
            'backbone_embedding_dim': cfg.MODEL.BACKBONE.EMBEDDING_DIM,
            'feat_dim': cfg.MODEL.BACKBONE.FEAT_DIM,
            'embedding_dim': cfg.MODEL.HEADS.EMBEDDING_DIM,
            'num_class': cfg.MODEL.HEADS.NUM_CLASSES,
            'num_attn': cfg.MODEL.HEADS.ATTN,
            'backbone': backbone,
            'heads': heads,
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
                    },
                    'center': {
                        'scale': cfg.MODEL.LOSSES.CENTER.SCALE
                    },
                    'kl': {
                        'margin': cfg.MODEL.LOSSES.KL.MARGIN,
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

        a3 = self.upSamper3(a4)      
        a2 = self.upSamper2(a3)
        a1 = self.upSamper1(a2)

        f3 = self.attnShaped3(a3, x3)
        f2 = self.attnShaped2(a2, x2)
        f1 = self.attnShaped1(a1, x1)


        feature3 = self.backbone3(f3)
        feature2 = self.backbone2(f2)
        feature1 = self.backbone1(f1)

        feature4 = self.embedding4(feature4)
        feature3 = self.embedding3(feature3)
        feature2 = self.embedding2(feature2)
        feature1 = self.embedding1(feature1)

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"]

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            outputs4 = self.heads(feature4, targets, batched_inputs=batched_inputs)
            outputs3 = self.head3(feature3, targets, batched_inputs=batched_inputs)
            outputs2 = self.head2(feature2, targets, batched_inputs=batched_inputs)
            outputs1 = self.head1(feature1, targets, batched_inputs=batched_inputs)
            attn_output = self.attn_head(attn_feat, targets, batched_inputs=batched_inputs)

            losses = self.losses(attn_output, outputs4, outputs3, outputs2, outputs1, targets, attn_map=a4)
                
            return losses
        else:
            # attn_output = self.attn_head(attn_feat, batched_inputs=batched_inputs)
            outputs4 = self.heads(feature4, batched_inputs=batched_inputs)
            outputs3 = self.head3(feature3, batched_inputs=batched_inputs)
            outputs2 = self.head2(feature2, batched_inputs=batched_inputs)
            outputs1 = self.head1(feature1, batched_inputs=batched_inputs)
            return torch.cat((outputs4, outputs3, outputs2, outputs1), dim=1)

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

        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self, attn_output, b4_outputs, b3_outputs, b2_outputs, b1_outputs, gt_labels, attn_map=None):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        # fmt: off
        pred_class_logits = attn_output['pred_class_logits'].detach()

        b1_logits         = b1_outputs['cls_outputs']
        b2_logits         = b2_outputs['cls_outputs']
        b3_logits         = b3_outputs['cls_outputs']
        b4_logits         = b4_outputs['cls_outputs']
        attn_logits       = attn_output['cls_outputs']

        b1_pool_feat      = b1_outputs['features']
        b2_pool_feat      = b2_outputs['features']
        b3_pool_feat      = b3_outputs['features']
        b4_pool_feat      = b4_outputs['features']
        attn_pool_feat    = attn_output['features']

        # fmt: on

        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']

        if 'CrossEntropyLoss' in loss_names:
            ce_kwargs = self.loss_kwargs.get('ce')

            loss_dict['loss_cls_b1'] = cross_entropy_loss(
                b1_logits,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale')

            loss_dict['loss_cls_b2'] = cross_entropy_loss(
                b2_logits,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale')

            loss_dict['loss_cls_b3'] = cross_entropy_loss(
                b3_logits,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale')

            loss_dict['loss_cls_b4'] = cross_entropy_loss(
                b4_logits,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale')

            loss_dict['loss_cls_attn'] = cross_entropy_loss(
                attn_logits,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale')

        if 'TripletLoss' in loss_names:
            tri_kwargs = self.loss_kwargs.get('tri')
            loss_dict['loss_triplet_b1'] = triplet_loss(
                b1_pool_feat,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale')

            loss_dict['loss_triplet_b2'] = triplet_loss(
                b2_pool_feat,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale')

            loss_dict['loss_triplet_b3'] = triplet_loss(
                b3_pool_feat,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale')

            loss_dict['loss_triplet_b4'] = triplet_loss(
                b4_pool_feat,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale')

            # loss_dict['loss_triplet_attn'] = triplet_loss(
            #     attn_pool_feat,
            #     gt_labels,
            #     tri_kwargs.get('margin'),
            #     tri_kwargs.get('norm_feat'),
            #     tri_kwargs.get('hard_mining')
            # ) * tri_kwargs.get('scale')
     
        if 'KlLoss' in loss_names:
            kl_kwargs = self.loss_kwargs.get('kl')
            margin = kl_kwargs.get('margin')
            loss_dict['loss_kl_12'] = torch.abs(-self.kl_loss(F.log_softmax(b1_pool_feat), F.softmax(b2_pool_feat.detach()))+margin)
            loss_dict['loss_kl_13'] = torch.abs(-self.kl_loss(F.log_softmax(b1_pool_feat), F.softmax(b3_pool_feat.detach()))+margin)
            loss_dict['loss_kl_14'] = torch.abs(-self.kl_loss(F.log_softmax(b1_pool_feat), F.softmax(b4_pool_feat.detach()))+margin)
            loss_dict['loss_kl_23'] = torch.abs(-self.kl_loss(F.log_softmax(b2_pool_feat), F.softmax(b3_pool_feat.detach()))+margin)
            loss_dict['loss_kl_24'] = torch.abs(-self.kl_loss(F.log_softmax(b2_pool_feat), F.softmax(b4_pool_feat.detach()))+margin)
            loss_dict['loss_kl_34'] = torch.abs(-self.kl_loss(F.log_softmax(b3_pool_feat), F.softmax(b4_pool_feat.detach()))+margin)

            loss_dict['loss_kl_21'] = torch.abs(-self.kl_loss(F.log_softmax(b2_pool_feat), F.softmax(b1_pool_feat.detach()))+margin)
            loss_dict['loss_kl_31'] = torch.abs(-self.kl_loss(F.log_softmax(b3_pool_feat), F.softmax(b1_pool_feat.detach()))+margin)
            loss_dict['loss_kl_41'] = torch.abs(-self.kl_loss(F.log_softmax(b4_pool_feat), F.softmax(b1_pool_feat.detach()))+margin)
            loss_dict['loss_kl_32'] = torch.abs(-self.kl_loss(F.log_softmax(b3_pool_feat), F.softmax(b2_pool_feat.detach()))+margin)
            loss_dict['loss_kl_42'] = torch.abs(-self.kl_loss(F.log_softmax(b4_pool_feat), F.softmax(b2_pool_feat.detach()))+margin)
            loss_dict['loss_kl_43'] = torch.abs(-self.kl_loss(F.log_softmax(b3_pool_feat), F.softmax(b4_pool_feat.detach()))+margin)

        if 'OrthogonalLoss' in loss_names:
            all_feat = torch.stack((b1_pool_feat, b2_pool_feat, b3_pool_feat, b4_pool_feat), dim=1)
            B, C, HW = all_feat.shape
            all_feat = F.normalize(all_feat, dim=2)
            loss_dict['loss_orthogonal'] = torch.norm(torch.bmm(all_feat, all_feat.view(B, -1, C))-torch.eye(C).cuda()) / B
        
        if 'CrossOrthogonalLoss' in loss_names:
            feat = torch.stack((b1_pool_feat, b2_pool_feat, b3_pool_feat, b4_pool_feat), dim=1)
            B, _ = b1_pool_feat.shape

            loss_co = {}
            for i in range(B):
                dist = torch.mm(feat[i], feat[i].T)
                v = torch.eig(dist, eigenvectors=False).eigenvalues[:, :1]
                print(v)
                v_max = torch.max(v).values
                v_min = torch.min(v).values
                loss_co[str(i)] = torch.norm((v_max-v_min) * dist)

            loss_dict["loss_co"] = sum(loss_co.values())

        return loss_dict

class UpSamper(nn.Module):
    def __init__(self, in_dim, output_dim=None, scale=2):
        super(UpSamper, self).__init__()

        if output_dim is None:
            self.up2=nn.Sequential(
                nn.ConvTranspose2d(in_dim , in_dim, kernel_size=scale, stride=scale),
                nn.Conv2d(in_dim, in_dim, 1),
                IBN(in_dim, "syncBN"),
                # get_norm("syncBN", in_dim),
                nn.ReLU()
            )
            
        else:
            self.up2=nn.Sequential(
                nn.ConvTranspose2d(in_dim , in_dim, kernel_size=scale, stride=scale),
                nn.Conv2d(in_dim, output_dim, 1),
                # get_norm("syncBN", output_dim),
                IBN(output_dim, "syncBN"),
                nn.ReLU()
            )
            
        self.up2.apply(weights_init_kaiming)


    def forward(self, x):

        back2 = self.up2(x)

        return back2

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        # self.bn = nn.InstanceNorm2d(out_channels, eps=0.001)
        # self.bn = get_norm("syncBN", out_channels)
        self.bn = IBN(out_channels, "syncBN")
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

    def __init__(self, input_dim, embedding_dim=0):
        super(Attach, self).__init__()
        self.pool = GeneralizedMeanPoolingP()
        # self.pool = AdaptiveAvgMaxPool()
        self.embedding_dim = embedding_dim
        if embedding_dim != 0:
            self.embedding_layer = nn.Conv2d(input_dim, embedding_dim, 1, bias=False)
            self.embedding_layer.apply(weights_init_kaiming)

    def forward(self, attentions, features):
        feature_matrix = []
        for i in range(attentions.shape[1]):
            AiF = features * attentions[:, i:i + 1, ...]
            if self.embedding_dim !=0:
                AiF = self.embedding_layer(AiF)
            AiF = self.pool(AiF).view(features.shape[0], -1)
            feature_matrix.append(AiF)
        feature_matrix = torch.cat(feature_matrix, dim=1)

        # sign-sqrt
        feature_matrix = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + self.EPSILON)
        # feature_matrix = F.normalize(feature_matrix, dim=-1)
        return feature_matrix.view(feature_matrix.shape[0], feature_matrix.shape[1], 1, 1) # B * (M*C)

class AttnShaped(nn.Module):
    EPSILON = 1e-12

    def __init__(self, input_dim):
        super(AttnShaped, self).__init__()

        # self.bn = get_norm("syncBN", input_dim)
        self.bn = IBN(input_dim, "syncBN")
        
        self.relu = nn.ReLU()
        self.bn.apply(weights_init_kaiming)

    def forward(self, attentions, features):

        # attentions = torch.mean(attentions, dim=1, keepdim=True)

        attentions = torch.max(attentions, dim=1, keepdim=True).values
        # t_max = torch.max(torch.max(attentions, dim=2, keepdim=True).values, dim=3, keepdim=True).values.expand(attentions.shape)
        # t_min = torch.min(torch.min(attentions, dim=2, keepdim=True).values, dim=3, keepdim=True).values.expand(attentions.shape)
        # attentions = (attentions - t_min) / (t_max - t_min)
        # attentions = torch.where(attentions>0.1, attentions, torch.zeros_like(attentions).cuda())
        attentions = F.normalize(attentions, dim=(2,3))
        AiF = features * attentions
        AiF = self.bn(AiF)
        AiF = self.relu(AiF)

        return AiF
