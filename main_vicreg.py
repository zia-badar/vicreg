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
from dataset2 import OneClassDataset2, AugmentedDataset2
from datasets import OneClassDataset
from distributed import init_distributed_mode

import resnet


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
    parser.add_argument("--mlp", default="512-512-512",
                        help='Size and number of layers of the MLP expander head')

    # Optim
    parser.add_argument("--epochs", type=int, default=512,
                        help='Number of epochs')
    parser.add_argument("--batch-size", type=int, default=256,
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


class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        # self.g = nn.Sequential(
        #                        nn.Linear(512, 512, bias=False),
        #                        nn.BatchNorm1d(512),
        #                        nn.ReLU(inplace=True),
        #                        nn.Linear(512, feature_dim),
        #                        )

        layers = []
        # for _ in range(8):
        #     layers += [nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True)]
        layers.append(nn.Linear(512, 16))
        self.g = nn.Sequential(*layers)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return feature, out


def main(args):
    torch.backends.cudnn.benchmark = True
    # init_distributed_mode(args)
    # print(args)
    # gpu = torch.device(args.device)

    # if args.rank == 0:
    #     args.exp_dir.mkdir(parents=True, exist_ok=True)
    #     stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
    #     print(" ".join(sys.argv))
    #     print(" ".join(sys.argv), file=stats_file)


    args.exp_dir.mkdir(parents=True, exist_ok=True)
    stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
    print(" ".join(sys.argv))
    print(" ".join(sys.argv), file=stats_file)

    transforms = aug.TrainTransform()

    # dataset = datasets.ImageFolder(args.data_dir / "train", transforms)
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    # assert args.batch_size % args.world_size == 0
    # per_device_batch_size = args.batch_size // args.world_size
    # loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=per_device_batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     sampler=sampler,
    # )




    # model = VICReg(args).cuda(gpu)
    model = VICReg(args).cuda()
    model.train()
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    # optimizer = LARS(
    #     model.parameters(),
    #     lr=0,
    #     weight_decay=args.wd,
    #     weight_decay_filter=exclude_bias_and_norm,
    #     lars_adaptation_filter=exclude_bias_and_norm,
    # )

    base_optimizer = SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-6)
    optim = LARS(base_optimizer, eps=1e-8, trust_coef=0.001)
    scheduler = CosineAnnealingLR(optim, args.epochs)
    scheduler_warmup = GradualWarmupScheduler(optim, multiplier=10.0, total_epoch=10, after_scheduler=scheduler)

    # model.load_state_dict(torch.load(f'exp/resnet50_{args._class}.pth'))
    # roc = analysis(model, args)
    # return roc

    # if (args.exp_dir / "model.pth").is_file():
    #     if args.rank == 0:
    #         print("resuming from checkpoint")
    #     ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
    #     start_epoch = ckpt["epoch"]
    #     model.load_state_dict(ckpt["model"])
    #     optimizer.load_state_dict(ckpt["optimizer"])
    # else:
    #     start_epoch = 0

    inlier = [args._class]
    outlier = list(range(10))
    outlier.remove(args._class)
    # dataset = CIFAR10(root='../', train=True, download=True)
    transform = transforms
    # inlier_dataset = OneClassDataset(dataset, one_class_labels=inlier, transform=transform)
    # outlier_dataset = OneClassDataset(dataset, zero_class_labels=outlier, transform=transform)
    # train_inlier_dataset = Subset(inlier_dataset, range(0, (int)(.7 * len(inlier_dataset))))
    # train_dataset = train_inlier_dataset
    # validation_inlier_dataset = Subset(inlier_dataset, range((int)(.7 * len(inlier_dataset)), len(inlier_dataset)))
    # validation_dataset = ConcatDataset([validation_inlier_dataset, outlier_dataset])

    cifar10_train = CIFAR10(root='.', train=True, download=True)
    # cifar10_test = CIFAR10(root='.', train=False, download=True)
    train_dataset = OneClassDataset(cifar10_train, one_class_labels=inlier, transform=transform)
    # test_dataset = ConcatDataset([OneClassDataset(cifar10_train, zero_class_labels=outlier, transform=transform),
    #                               OneClassDataset(cifar10_test, one_class_labels= inlier, zero_class_labels=outlier, transform=transform)])

    loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)

    start_epoch = 1

    # start_time = last_logging = time.time()
    # scaler = torch.cuda.amp.GradScaler()
    for epoch in tqdm(range(start_epoch, args.epochs+1)):
        # sampler.set_epoch(epoch)
        total_loss = 0
        for step, ((x, y), l) in enumerate(loader):
            # x = x.cuda(gpu, non_blocking=True)
            # y = y.cuda(gpu, non_blocking=True)
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            # lr = adjust_learning_rate(args, optimizer, loader, step)

            optim.zero_grad()
            # with torch.cuda.amp.autocast():
            #     loss = model.forward(x, y, l)

            with torch.cuda.amp.autocast():
                loss = model.forward(x, y, l)

            loss.backward()

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            optim.step()
            scheduler_warmup.step(epoch - 1 + step / len(loader))

            current_time = time.time()
            # if args.rank == 0 and current_time - last_logging > args.log_freq_time:
            #     stats = dict(
            #         epoch=epoch,
            #         step=step,
            #         loss=loss.item(),
            #         # time=int(current_time - start_time),
            #         lr=1,
            #     )
            #     # print(json.dumps(stats))
            #     # print(json.dumps(stats), file=stats_file)
            #     last_logging = current_time
            total_loss += loss.item()

        print(f'loss: {total_loss/len(loader): .2f}, epoch: {epoch}')
        if args.rank == 0:
            state = dict(
                epoch=epoch + 1,
                model=model.state_dict(),
                # optimizer=optimizer.state_dict(),
            )
            torch.save(state, args.exp_dir / "model.pth")

        # if epoch % 50 == 0:
        #     roc = analysis(model, args, False)
        #     print(f'class: {cifar10_train.classes[args._class]}, roc: {roc}')
        #     model.train()

    if args.rank == 0:
        # torch.save(model.module.backbone.state_dict(), args.exp_dir / f"resnet50_{args._class}.pth")
        torch.save(model.state_dict(), args.exp_dir / f"resnet50_{args._class}.pth")

    roc = analysis(model, args)
    print(f'class: {cifar10_train.classes[args._class]}, roc: {roc}')
    return roc


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.base_lr * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

def contrastive_loss(z, z_aug):
    z = normalize(z, dim=1)
    z_aug = normalize(z_aug, dim=1)

    batch_size = z.shape[0]
    temperature = 0.5

    # simclr
    pos = torch.exp((z[:, None, :] @ z_aug[:, :, None]).squeeze()/temperature)
    mask = torch.cat([torch.logical_not(torch.eye(batch_size).cuda()), torch.logical_not(torch.eye(batch_size).cuda())])
    neg = torch.sum(torch.masked_select(torch.exp((z @ torch.cat([z, z_aug]).T) / temperature), mask.T).view(batch_size, 2*batch_size-2), dim=1)
    l = -torch.mean(torch.log(pos/(pos + neg)))

    return l

class VICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])
        self.backbone_1 = Model().cuda()
        # self.backbone_1, self.embedding = resnet.__dict__[args.arch](
        #     zero_init_residual=True
        # )
        # self.backbone_2, self.embedding = resnet.__dict__[args.arch](
        #     zero_init_residual=True
        # )
        # self.embedding = 32
        # self.backbone_1.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1)
        # self.backbone_1.maxpool = nn.Identity()
        # self.backbone_1 = nn.Sequential(self.backbone_1, BatchNorm1d(512), Linear(512, 32))

        # self.backbone_2.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1)
        # self.backbone_2.maxpool = nn.Identity()
        # self.backbone_2 = nn.Sequential(self.backbone_2, BatchNorm1d(512), Linear(512, 32))

        self.projector_1 = Projector(args, 16)
        #
        layers = []
        for _ in range(4):
            layers += [nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True)]
        layers += [nn.Linear(512, 4)]
        self.classifier = nn.Sequential(*layers)
        #
        self.cross_entropy_loss = CrossEntropyLoss()
        # #
        # self.classifier_aug = nn.Sequential(Linear(1024, 1))
        # #
        # self.labels = torch.tensor([1, 0]).repeat(256).type(torch.float).cuda()
        # #
        # self.bce_loss = BCEWithLogitsLoss()
        #
        # self.projector_2 = Projector(args, self.embedding)

    def classifying_aug_loss(self, a, b):
        batch_size = a.shape[0]
        labels = torch.tensor([1, 0]).repeat(batch_size).type(torch.float).cuda()
        b_sft = torch.cat([b[1:], b[0][None]])
        tmp_1 = torch.reshape(torch.permute(torch.stack((a, b, a, b_sft)), (1, 0, 2)), (batch_size, 2, 1024))
        tmp_2 = self.classifier_aug(tmp_1)
        return self.bce_loss(tmp_2.reshape(batch_size*2, 1), labels[:, None])

    def forward(self, x, y, l):
        _, repr_x = self.backbone_1(x)
        _, repr_y = self.backbone_1(y)
        x = self.projector_1(repr_x)
        y = self.projector_1(repr_y)
        l = F.one_hot(l-1, num_classes = 4).cuda().to(torch.float)
        rot_loss = (self.cross_entropy_loss(self.classifier(x), l) + self.cross_entropy_loss(self.classifier(y), l))/2

        # class_aug_loss = self.classifying_aug_loss(x, y)

        # contras_loss = contrastive_loss(repr_x, repr_y)

        repr_loss = F.mse_loss(x, y)

        # dist = torch.cdist(x, y) ** 2
        # num = torch.sum(torch.diag(dist)) / (torch.prod(torch.tensor(x.shape)))     # same as mse_loss
        # den = torch.sum(torch.sum(dist, dim=1)) / (torch.prod(torch.tensor(x.shape)) * x.shape[0])
        # repr_loss = num / den

        # num2 = torch.diag(dist) / x.shape[1]
        # den2 = torch.sum(dist, dim=1) / torch.prod(torch.tensor(x.shape))
        # repr_loss = torch.mean(num2 / den2)


        # x = torch.cat(FullGatherLayer.apply(x), dim=0)
        # y = torch.cat(FullGatherLayer.apply(y), dim=0)
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
            # + class_aug_loss
            + rot_loss
            # contras_loss
        )
        return loss


def Projector(args, embedding):
    mlp_spec = f"{embedding}-{args.mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


def exclude_bias_and_norm(p):
    return p.ndim == 1


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# class LARS(optim.Optimizer):
#     def __init__(
#         self,
#         params,
#         lr,
#         weight_decay=0,
#         momentum=0.9,
#         eta=0.001,
#         weight_decay_filter=None,
#         lars_adaptation_filter=None,
#     ):
#         defaults = dict(
#             lr=lr,
#             weight_decay=weight_decay,
#             momentum=momentum,
#             eta=eta,
#             weight_decay_filter=weight_decay_filter,
#             lars_adaptation_filter=lars_adaptation_filter,
#         )
#         super().__init__(params, defaults)
#
#     @torch.no_grad()
#     def step(self):
#         for g in self.param_groups:
#             for p in g["params"]:
#                 dp = p.grad
#
#                 if dp is None:
#                     continue
#
#                 if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
#                     dp = dp.add(p, alpha=g["weight_decay"])
#
#                 if g["lars_adaptation_filter"] is None or not g[
#                     "lars_adaptation_filter"
#                 ](p):
#                     param_norm = torch.norm(p)
#                     update_norm = torch.norm(dp)
#                     one = torch.ones_like(param_norm)
#                     q = torch.where(
#                         param_norm > 0.0,
#                         torch.where(
#                             update_norm > 0, (g["eta"] * param_norm / update_norm), one
#                         ),
#                         one,
#                     )
#                     dp = dp.mul(q)
#
#                 param_state = self.state[p]
#                 if "mu" not in param_state:
#                     param_state["mu"] = torch.zeros_like(p)
#                 mu = param_state["mu"]
#                 mu.mul_(g["momentum"]).add_(dp)
#
#                 p.add_(mu, alpha=-g["lr"])


# def batch_all_gather(x):
#     x_list = FullGatherLayer.apply(x)
#     return torch.cat(x_list, dim=0)
#
#
# class FullGatherLayer(torch.autograd.Function):
#     """
#     Gather tensors from all process and support backward propagation
#     for the gradients across processes.
#     """
#
#     @staticmethod
#     def forward(ctx, x):
#         output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
#         dist.all_gather(output, x)
#         return tuple(output)
#
#     @staticmethod
#     def backward(ctx, *grads):
#         all_gradients = torch.stack(grads)
#         dist.all_reduce(all_gradients)
#         return all_gradients[dist.get_rank()]
#
#
# def handle_sigusr1(signum, frame):
#     os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
#     exit()
#
#
# def handle_sigterm(signum, frame):
#     pass
#

if __name__ == "__main__":
    parser = argparse.ArgumentParser('VICReg training script', parents=[get_arguments()])

    sum = 0
    for i in range(10):
        args = parser.parse_args()
        args.rank = 0
        args._class = i
        try:
            sum += main(args)
        except Exception as e:
            print(e)

    print(f'avg roc: {sum/10.}')