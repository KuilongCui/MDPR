# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from ...utils.registry import Registry

REID_HEADS_REGISTRY = Registry("HEADS")
REID_HEADS_REGISTRY.__doc__ = """
Registry for reid heads in a baseline model.

ROIHeads take feature maps and region proposals, and
perform per-region computation.
The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

def build_heads(cfg, dim_spec=None, embedding_spec=None, pool_spec=None, num_classes_spec=None, with_bnneck_spec=None):
    """
    Build REIDHeads defined by `cfg.MODEL.REID_HEADS.NAME`.
    """
    head = cfg.MODEL.HEADS.NAME
    return REID_HEADS_REGISTRY.get(head)(cfg, dim_spec, embedding_spec, pool_spec, num_classes_spec, with_bnneck_spec)
