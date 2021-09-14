from superbert.utils.registry import Registry, build_from_cfg


MODEL = Registry('model')

def build_model(cfg, default_args=None):

    model = build_from_cfg(cfg.model, MODEL, {"cfg":cfg})
    return model