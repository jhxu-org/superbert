from PIL.Image import NONE
from superbert.datasets import transforms
from torch.utils.data import DataLoader
from superbert.utils.registry import Registry, build_from_cfg
from .samplers import DistributedGroupSampler, DistributedSampler, GroupSampler
from functools import partial
from superbert.utils.dist_utils import get_dist_info
import numpy as np
import random
from .transforms import build_transforms
from torch.utils.data import ConcatDataset


DATASETS = Registry('dataset')

def build_dataset(cfg, default_args=None):
    # for data in cfg.datasets:
    datasets = []
    for data in cfg:
        if hasattr(data, "pipeline") and data.pipeline != None:
            transforms = build_transforms(data.pipeline)
            data.update({"transforms":transforms})
        dataset = build_from_cfg(data, DATASETS, default_args)
        datasets.append(dataset)
    
    return ConcatDataset(datasets)

# def build_transform(cfg, default_args=None):
#     # for data in cfg.datasets:
#     dataset = build_from_cfg(cfg[0], PIPELINES = Registry('pipeline')
# , default_args)
#     return dataset


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    if dist:
        # DistributedGroupSampler will definitely shuffle the data to satisfy
        # that images on each GPU are in the same group
        # if shuffle:
        #     sampler = DistributedSampler(
        #         dataset, samples_per_gpu, world_size, rank, seed=seed)
        # else:
        sampler = DistributedSampler(
                dataset, world_size, rank, shuffle=shuffle, seed=seed)
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = None#GroupSampler(dataset, samples_per_gpu) if shuffle else None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=dataset.collate,
        pin_memory=False,
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)