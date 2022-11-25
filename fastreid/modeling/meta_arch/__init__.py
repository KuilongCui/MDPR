# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from .build import META_ARCH_REGISTRY, build_model


# import all the meta_arch, so they will be registered
from .baseline import Baseline
from .mgn import MGN
from .moco import MoCo
from .distiller import Distiller
from .sbaseline import Sbaseline
from .OsNetSbaseline import OsNetSbaseline
from .GanSbaseline import GanSbaseline
from .PartSbaseline import PartSbaseline
from .RepVggSbaseline import RepVggSbaseline
from .sbaselineT import SbaselineT
from .sbaselineCam import SbaselineCam
