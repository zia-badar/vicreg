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

import augmentations as aug
from analysis import analysis
from analysis_result import Result
from dataset2 import OneClassDataset2, AugmentedDataset2
from datasets import OneClassDataset
from distributed import init_distributed_mode

import resnet
from model import Model


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


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)




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

    transforms = aug.TrainTransform()

    model = VICReg(args).cuda()
    model.train()
    # model.load_state_dict(torch.load(f'fix_network'))
    # torch.save(model.state_dict(), 'fix_network')

    base_optimizer = SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-6)
    optim = LARS(base_optimizer, eps=1e-8, trust_coef=0.001)
    scheduler = CosineAnnealingLR(optim, args.epochs)
    scheduler_warmup = GradualWarmupScheduler(optim, multiplier=10.0, total_epoch=10, after_scheduler=scheduler)

    # model.load_state_dict(torch.load(f'exp/resnet50_{args._class}.pth'))
    # roc = analysis(model, args, result)
    # return roc

    inlier = [args._class]
    outlier = list(range(10))
    outlier.remove(args._class)
    transform = transforms

    cifar10_train = CIFAR10(root='.', train=True, download=True)
    train_dataset = OneClassDataset(cifar10_train, one_class_labels=inlier, transform=transform, with_rotation=args.use_rotated_data)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)

    start_epoch = 1

    # start_time = last_logging = time.time()
    # scaler = torch.cuda.amp.GradScaler()
    bar = tqdm(range(start_epoch, args.epochs+1))
    for epoch in bar:
        # sampler.set_epoch(epoch)
        total_loss = 0
        epoch_loss = None
        for step, ((x, y), l) in enumerate(loader):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            optim.zero_grad()
            with torch.cuda.amp.autocast():
                loss_info, loss = model.forward(x, y, l, args)

            # loss_info, loss = model.forward(x, y, l, args)

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
                # optimizer=optimizer.state_dict(),
            )
            torch.save(state, args.exp_dir / "model.pth")

    if args.rank == 0:
        # torch.save(model.module.backbone.state_dict(), args.exp_dir / f"resnet50_{args._class}.pth")
        torch.save(model.state_dict(), args.exp_dir / f"resnet50_{args._class}.pth")

    roc = analysis(model, args, result)
    print(f'class: {cifar10_train.classes[args._class]}, roc: {roc}')
    return roc

class VICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])
        self.backbone_1 = Model(args.encodingdim, args.mlp).cuda()

        #
        layers = []
        proj_dim = (int)(args.mlp.split('-')[-1])
        for _ in range(4):
            layers += [nn.Linear(proj_dim, proj_dim), nn.BatchNorm1d(proj_dim), nn.ReLU(inplace=True)]
        layers += [nn.Linear(proj_dim, 4)]
        self.classifier = nn.Sequential(*layers)
        #
        self.cross_entropy_loss = CrossEntropyLoss()

    def classifying_aug_loss(self, a, b):
        batch_size = a.shape[0]
        labels = torch.tensor([1, 0]).repeat(batch_size).type(torch.float).cuda()
        b_sft = torch.cat([b[1:], b[0][None]])
        tmp_1 = torch.reshape(torch.permute(torch.stack((a, b, a, b_sft)), (1, 0, 2)), (batch_size, 2, 1024))
        tmp_2 = self.classifier_aug(tmp_1)
        return self.bce_loss(tmp_2.reshape(batch_size*2, 1), labels[:, None])

    def forward(self, x, y, l, args):
        _, _, x = self.backbone_1(x)
        _, _, y = self.backbone_1(y)

        l = F.one_hot(l-1, num_classes = 4).cuda().to(torch.float)
        rot_loss = 0
        if args.rotation_pred:
            rot_loss = (self.cross_entropy_loss(self.classifier(x), l) + self.cross_entropy_loss(self.classifier(y), l))/2

        repr_loss = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)
        # # print(f'{self.args.sim_coeff * repr_loss} , {self.args.std_coeff * std_loss} , {self.args.cov_coeff * cov_loss}, {class_aug_loss}')

        loss = (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
            + rot_loss
        )
        return [self.args.sim_coeff * repr_loss.item(), self.args.std_coeff * std_loss.item(), self.args.cov_coeff * cov_loss.item(), rot_loss.item()], loss


# def Projector(args, embedding):
#     mlp_spec = f"{embedding}-{args.mlp}"
#     layers = []
#     f = list(map(int, mlp_spec.split("-")))
#     for i in range(len(f) - 2):
#         layers.append(nn.Linear(f[i], f[i + 1]))
#         layers.append(nn.BatchNorm1d(f[i + 1]))
#         layers.append(nn.ReLU(True))
#     layers.append(nn.Linear(f[-2], f[-1], bias=False))
#     return nn.Sequential(*layers)


def exclude_bias_and_norm(p):
    return p.ndim == 1


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    parser = argparse.ArgumentParser('VICReg training script', parents=[get_arguments()])

    result = Result('result_01')

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