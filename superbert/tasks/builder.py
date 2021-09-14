
from superbert.utils.registry import Registry, build_from_cfg



TASKS = Registry('tasks')

def build_tasks(cfg, default_args=None):
    # for data in cfg.datasets:
    tasks = []
    for data in cfg:
        task = build_from_cfg(data, TASKS, default_args)
        tasks.append(task)
    
    return tasks