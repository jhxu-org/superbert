from __future__ import absolute_import, division, print_function
from superbert.modeling.builder import build_model
from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass
import argparse
import datetime
import json
import logging
import os
import random
from superbert.trainer import train
import sys
import time
import math
import shutil
from superbert.utils.config import Config
from superbert.utils.misc import set_seed
import numpy as np
import torch
from superbert.modeling import build_model
from transformers.models.bert import (BertConfig,
                                  BertTokenizer)
from transformers.file_utils import WEIGHTS_NAME

from superbert.datasets.builder import build_dataset, build_dataloader

from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from superbert.utils.misc import mkdir, get_rank
from superbert.utils.metric_logger import TensorboardLogger

logger = logging.getLogger(__name__)
from superbert.trainer import train

""" ****** Pretraining ****** """


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="./config/default.py", type=str, required=False,
                        help="config file.")
   
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")

    parser.add_argument("--pretrain", default=None, type=str,
                        help="The pretrain model path.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
   
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--from_scratch", action='store_true',
                        help="train from scratch")
    # distributed
    parser.add_argument('--gpu_ids', type=str, default='-1')

    # logging
    parser.add_argument('--ckpt_period', type=int, default=10000,
                        help="Period for saving checkpoint")
    parser.add_argument('--log_period', type=int, default=100,
                        help="Period for saving logging info")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)

    # cfg.merge_from_dict(args)

    if args.gpu_ids != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    args.num_gpus = int(
        os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = args.num_gpus > 1

    if args.gpu_ids != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        logger.info("Output Directory Exists.")

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method="env://"
        )
        args.n_gpu = 1
    args.device = device
    cfg.gpu_ids = args.gpu_ids
    cfg.local_rank = args.local_rank
    cfg.n_gpu = args.n_gpu
    cfg.device = args.device
    cfg.output_dir = args.output_dir
    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1)
    )

    if cfg.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                cfg.gradient_accumulation_steps))


    set_seed(args.seed, args.n_gpu)  # Added here for reproductibility (even between python 2 and 3)

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train:
        raise ValueError(
            "Training is currently the only implemented execution option. Please set `do_train`.")

    if not os.path.exists(args.output_dir):
        mkdir(args.output_dir)

    last_checkpoint_dir = None
    arguments = {"iteration": 0}
    if os.path.exists(args.output_dir):
        save_file = os.path.join(args.output_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        if last_saved:
            folder_name = os.path.splitext(last_saved.split('/')[0])[0] # in the form of checkpoint-00001 or checkpoint-00001/pytorch_model.bin
            last_checkpoint_dir = os.path.join(args.output_dir, folder_name)
            arguments["iteration"] = int(folder_name.split('-')[-1])
            assert os.path.isfile(os.path.join(last_checkpoint_dir, WEIGHTS_NAME)), "Last_checkpoint detected, but file not found!"

    model  = build_model(cfg).cuda()


    cfg.resume_path = None
    if last_checkpoint_dir is not None:  # recovery
        cfg.resume_path = last_checkpoint_dir
        logger.info(" -> Recovering model from {}".format(last_checkpoint_dir))
        model.load_state_dict(torch.load(os.path.join(cfg.resume_path, 'superbert.pth'), map_location='cpu'))
    meters = TensorboardLogger(
        log_dir=os.path.join(args.output_dir,"train_logs"),
        delimiter="  ",
    )
    # cfg.model.update(cfg.config)


    # train from scratch
    if args.from_scratch:
        if last_checkpoint_dir is None:
            logger.info("Training from scratch ... ")
            model.apply(model.init_weights)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        'Total Parameters: {}'.format(total_params))
    if args.pretrain:
        model.load_pretrain(args.pretrain)

    if get_rank() == 0:
        BertTokenizer.from_pretrained(
                cfg.model_name_or_path, cache_dir=cfg.cache_dir, do_lower_case="uncased" in cfg.model_name_or_path
            )
    if args.local_rank != -1:
        torch.distributed.barrier()
    tokenizer = BertTokenizer.from_pretrained(
            cfg.model_name_or_path, do_lower_case="uncased" in cfg.model_name_or_path
        )
    # if get_rank() == 0 and args.local_rank != -1:
    #     torch.distributed.barrier()
    train_dataset = build_dataset(cfg.train,{"tokenizer":tokenizer})
    val_dataset = build_dataset(cfg.val, {"tokenizer":tokenizer})
    
    train(cfg, train_dataset, val_dataset, model=model,  meters=meters)
    


if __name__ == "__main__":
    main()
