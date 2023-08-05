# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch import nn, optim
import torch.distributed as dist
import torchvision.datasets as datasets
from torch.nn import BatchNorm1d, Linear, ReLU, CrossEntropyLoss, BCEWithLogitsLoss, CosineSimilarity
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler, ReduceLROnPlateau
from torch.utils.data import Subset, ConcatDataset, DataLoader
from torchlars import LARS
from torchvision.datasets import CIFAR10
from torch.nn.functional import normalize
from torchvision.models import resnet18
from tqdm import tqdm
import pickle as pk

import augmentations as aug
from GradualWarmupScheduler import GradualWarmupScheduler
from analysis import analysis
from analysis_result import Result
from dataset2 import OneClassDataset2, AugmentedDataset2
from datasets import OneClassDataset
from distributed import init_distributed_mode

import resnet
from model import Model
from vicreg_loss import vicreg_loss


def get_arguments():
    parser = argparse.ArgumentParser(description="Pretrain a resnet model with VICReg", add_help=False)

    # Data
    # parser.add_argument("--data-dir", type=Path, default="/path/to/imagenet", required=True,
    #                     help='Path to the image net dataset')

    # Checkpoints
    parser.add_argument("--exp-dir", type=Path, default="./exp",
                        help='Path to the experiment folder, where all logs/checkpoints will be stored')
    parser.add_argument("--log-freq-time", type=int, default=60,
                        help='Print logs to the stats.txt file every [log-freq-time] seconds')

    # Model
    parser.add_argument("--arch", type=str, default="resnet18",
                        help='Architecture of the backbone encoder network')
    parser.add_argument("--mlp", default="1024-1024-1024",
                        help='Size and number of layers of the MLP expander head')
    parser.add_argument("--encodingdim", type=int, default="16",
                        help='Size of Y(representation)')

    # Optim
    parser.add_argument("--epochs", type=int, default=512,
                        help='Number of epochs')
    parser.add_argument("--batch-size", type=int, default=384,
                        help='Effective batch size (per worker batch size is [batch-size] / world-size)')
    parser.add_argument("--base-lr", type=float, default=0.2,
                        help='Base learning rate, effective learning after warmup is [base-lr] * [batch-size] / 256')
    parser.add_argument("--wd", type=float, default=1e-6,
                        help='Weight decay')

    # Loss
    parser.add_argument("--sim-coeff", type=float, default=25.0,
                        help='Invariance regularization loss coefficient')
    parser.add_argument("--std-coeff", type=float, default=25.0,
                        help='Variance regularization loss coefficient')
    parser.add_argument("--cov-coeff", type=float, default=1.0,
                        help='Covariance regularization loss coefficient')
    parser.add_argument("--rotation-pred", default=True, action=argparse.BooleanOptionalAction,
                        help='with rotation prediction')
    parser.add_argument("--use-rotated-data", default=True, action=argparse.BooleanOptionalAction,
                        help='with rotation data')

    # Running
    parser.add_argument("--num-workers", type=int, default=20)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # Distributed
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--_class', default=0, type=int,
                        help='normal class')

    return parser

def main(args, result):
    torch.backends.cudnn.benchmark = True

    if args.rotation_pred and not args.use_rotated_data:
        print('rotated data is required for rotation pred')
        exit(-1)

    args.exp_dir = Path(f'exp_{args.batch_size}_{args.epochs}_{args.encodingdim}_{args.mlp}_{args.rotation_pred}_{args.use_rotated_data}')

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
    print(" ".join(sys.argv))
    print(" ".join(sys.argv), file=stats_file)

    model = Model(args.encodingdim, args.mlp).cuda()
    model.train()

    base_optimizer = SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-6)
    optim = LARS(base_optimizer, eps=1e-8, trust_coef=0.001)
    scheduler = CosineAnnealingLR(optim, args.epochs)
    scheduler_warmup = GradualWarmupScheduler(optim, multiplier=10.0, total_epoch=10, after_scheduler=scheduler)

    optim.zero_grad()
    optim.step()

    # model.load_state_dict(torch.load(f'exp_384_512_16_1024-1024-1024_True_True/resnet50_{args._class}.pth'))
    # roc = analysis(model, args, result)
    # return roc

    inlier = [args._class]
    outlier = list(range(10))
    outlier.remove(args._class)

    cifar10_train = CIFAR10(root='.', train=True, download=True)
    train_dataset = OneClassDataset(cifar10_train, one_class_labels=inlier, transform=None, augmentation=False, with_rotation=False)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    with torch.no_grad():
        xs = []
        for x, _ in loader:
            xs.append(x)
        mean = torch.mean(torch.cat(xs), dim=[0, 2, 3])
        std = torch.std(torch.cat(xs), dim=[0, 2, 3])

    transforms = aug.TrainTransform(mean, std)

    transform = transforms

    cifar10_train = CIFAR10(root='.', train=True, download=True)
    train_dataset = OneClassDataset(cifar10_train, one_class_labels=inlier, transform=transform, with_rotation=args.use_rotated_data)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)

    start_epoch = 1

    bar = tqdm(range(start_epoch, args.epochs+1))
    loss_ = vicreg_loss(args.rotation_pred, int(args.mlp.split("-")[-1]))

    for epoch in bar:
        total_loss = 0
        epoch_loss = None
        for step, ((x, y), l) in enumerate(loader):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            optim.zero_grad()
            with torch.cuda.amp.autocast():
                _, _, x, c_x = model(x)
                _, _, y, c_y = model(y)
                loss_info, loss = loss_(x, y, c_x, c_y, l)

            loss.backward()
            if epoch_loss is None:
                epoch_loss = torch.tensor(loss_info)
            else:
                epoch_loss += torch.tensor(loss_info)

            optim.step()
            scheduler_warmup.step(epoch - 1 + step / len(loader))

            if torch.any(torch.isnan(loss)).item():
                raise Exception('nan loss')
            total_loss += loss.item()

        bar.set_description(f'{epoch_loss/len(loader)}, loss: {total_loss/len(loader): .2f}')

        if args.rank == 0:
            state = dict(
                epoch=epoch + 1,
                model=model.state_dict(),
            )
            torch.save(state, args.exp_dir / "model.pth")

    if args.rank == 0:
        torch.save(model.state_dict(), args.exp_dir / f"resnet50_{args._class}.pth")

    roc = analysis(model, args, result)
    print(f'class: {cifar10_train.classes[args._class]}, roc: {roc}')
    return roc

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    parser = argparse.ArgumentParser('VICReg training script', parents=[get_arguments()])

    args = parser.parse_args()
    result = Result(f'exp_{args.batch_size}_{args.epochs}_{args.encodingdim}_{args.mlp}_{args.rotation_pred}_{args.use_rotated_data}/result_01')

    sum = 0
    for i in range(10):
        args = parser.parse_args()
        args.rank = 0
        args._class = i
        try:
            sum += main(args, result)
        except Exception as e:
            print(e)

    print(f'avg roc: {sum/10.}')

    result.save()