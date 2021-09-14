from superbert.utils.registry import Registry, build_from_cfg
import numpy as np


PIPELINES = Registry('pipeline')

def build_transforms(cfg, default_args=None):
    # for data in cfg.datasets:
    transforms = build_from_cfg(cfg, PIPELINES, default_args)
    return transforms
