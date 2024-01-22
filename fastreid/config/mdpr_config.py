from .config import CfgNode as CN

def add_mdpr_cfg(cfg):
    _C = cfg

    _C.MODEL.BACKBONE.EMBEDDING_DIM = 512

    _C.ATTN_NUM = 2

    _C.PART_NUM = 2
    
    _C.ATTN_GUIDED = True

    _C.MODEL.LOSSES.ATTN_DIVERSITY = CN()
    _C.MODEL.LOSSES.ATTN_DIVERSITY.SCALE = 0.001

    _C.FUSION_ENABLE = True

    _C.DISTILLATION_ENABLE = True

    _C.OUTPUT_ALL = False

    _C.ATTN_DIVERSITY_LOSS_ENABLE = True
