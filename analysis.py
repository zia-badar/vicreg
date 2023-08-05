import random

import torch
import tsne_torch
from PIL import ImageOps, ImageFilter, Image
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM
from torch import cdist
from torch.nn import CosineSimilarity
from torch.nn.functional import normalize
from torch.utils.data import Subset, ConcatDataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms, ToTensor
from tsne_torch import TorchTSNE

import augmentations
from dataset2 import OneClassDataset2
from datasets import OneClassDataset


fig, axis = plt.subplots(10, 1)
fig.set_figwidth(10)
fig.set_figheight(100)

def analysis(model, args, result, showTSNE=True):
    aug = 10
    inlier = [args._class]
    outlier = list(range(10))
    outlier.remove(args._class)
    transform = None

    cifar10_train = CIFAR10(root='.', train=True, download=True)
    train_dataset = OneClassDataset(cifar10_train, one_class_labels=inlier, transform=None, augmentation=False,
                                    with_rotation=False)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    with torch.no_grad():
        xs = []
        for x, _ in loader:
            xs.append(x)
        mean = torch.mean(torch.cat(xs), dim=[0, 2, 3])
        std = torch.std(torch.cat(xs), dim=[0, 2, 3])
    normalization_transform = transforms.Normalize( mean=mean, std=std )

    score_sum = None
    for rotation in range(4):
        cifar10_train = CIFAR10(root='.', train=True, download=True)
        cifar10_test = CIFAR10(root='.', train=False, download=True)
        train_dataset = OneClassDataset(cifar10_train, one_class_labels=inlier, transform=transform, with_rotation=False, augmentation=False, rotation=rotation, normalization_transform=normalization_transform)
        test_dataset = ConcatDataset([OneClassDataset(cifar10_train, zero_class_labels=outlier, transform=transform, with_rotation=False, augmentation=False, rotation=rotation, normalization_transform=normalization_transform),
                                      OneClassDataset(cifar10_test, one_class_labels= inlier, zero_class_labels=outlier, transform=transform, with_rotation=False, augmentation=False, rotation=rotation, normalization_transform=normalization_transform)])

        with torch.no_grad():
            model.eval()

            training_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
            test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

            train_x = []
            with torch.no_grad():
                for x, l in training_dataloader:
                    x = x.cuda()
                    x, _, _, _ = model(x)
                    train_x.append(normalize(x, dim=1))
            train_x = torch.cat(train_x)
            gamma = (10 / (torch.var(train_x).item() * train_x.shape[1]))
            svm = OneClassSVM(kernel='rbf', gamma=gamma).fit(train_x.cpu().numpy())

            val_x = []
            labels = []
            with torch.no_grad():
                for x, l in test_dataset:
                    x = x.cuda()
                    # x, _, _, _ = model.backbone_1(x)
                    x, _, _, _ = model(x)
                    val_x.append(normalize(x, dim=1))
                    # val_x.append(x)
                    labels.append(l)
            val_x = torch.cat(val_x).cpu().numpy()
            labels = torch.cat(labels).cpu().numpy()

            score = svm.score_samples(val_x)

            roc = roc_auc_score(labels, score)
            print(f'class {args._class}: roc: {roc}')

            if score_sum is None:
                score_sum = score
            else:
                score_sum += score

            roc = roc_auc_score(labels, score_sum)
            print(f'class {args._class}: roc: {roc}\n\n')

            if not hasattr(result, 'class_to_rotation_roc'):
                result.class_to_rotation_roc = {}

            if result.class_to_rotation_roc.get(args._class) == None:
                result.class_to_rotation_roc[args._class] = {}

            result.class_to_rotation_roc[args._class][rotation] = roc

    roc = roc_auc_score(labels, score_sum)
    print(f'class {args._class}: roc: {roc}')

    if showTSNE:
        visual_tsne(model, args, roc, result, normalization_transform)
    return roc

def visual_tsne(model, args, roc, result, normalization_transform):
    aug = 10
    inlier = [args._class]
    outlier = list(range(10))
    outlier.remove(args._class)
    transform = None

    cifar10_test = CIFAR10(root='.', train=False, download=True)
    inlier_dataset = Subset(OneClassDataset(cifar10_test, one_class_labels=inlier, transform=transform, with_rotation=True, augmentation=False, normalization_transform=normalization_transform), range(0, 1000*4))
    outlier_dataset = Subset(OneClassDataset(cifar10_test, zero_class_labels=outlier, transform=transform, with_rotation=True, augmentation=False, normalization_transform=normalization_transform), range(0, (int)(len(inlier_dataset))))
    validation_dataset = ConcatDataset([inlier_dataset, outlier_dataset])

    with torch.no_grad():
        model.eval()

        validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size,
                                                            num_workers=args.num_workers, pin_memory=True, shuffle=True)

        samples = []
        labels = []
        for x, l in validation_dataloader:
            x = x.cuda()
            samples.append(model(x)[0])
            labels.append(l)

        samples = torch.cat(samples).cpu()
        labels = torch.cat(labels)

        limit = 12000
        labels = labels[:limit].cpu().numpy()

        torch.cuda.empty_cache()

        emb = TSNE(n_components=2, learning_rate='auto', init = 'random', perplexity = 30).fit_transform(samples[:limit].numpy())

        nominal_labels = labels == 1
        anomaly_labels = labels == 0

        ax = axis[args._class]

        ax.scatter(emb[anomaly_labels, 0], emb[anomaly_labels, 1], label='anomaly', c='r', marker='.')
        rot_0_labels = labels == 1
        ax.scatter(emb[rot_0_labels, 0], emb[rot_0_labels, 1], label='rot 0', c='g', marker='.')
        rot_90_labels = labels == 2
        ax.scatter(emb[rot_90_labels, 0], emb[rot_90_labels, 1], label='rot 90', c='k', marker='.')
        rot_180_labels = labels == 3
        ax.scatter(emb[rot_180_labels, 0], emb[rot_180_labels, 1], label='rot 180', c='m', marker='.')
        rot_270_labels = labels == 4
        ax.scatter(emb[rot_270_labels, 0], emb[rot_270_labels, 1], label='rot 270', c='y', marker='.')

        ax.set_title(f'class: {cifar10_test.classes[args._class]}, roc: {roc}')
        ax.legend()

        if not hasattr(result, 'tsne_plots'):
            result.tsne_plots = {}

        if result.tsne_plots.get(args._class) == None:
            result.tsne_plots[args._class] = {}

        if not hasattr(result.tsne_plots[args._class], 'emb'):
            result.tsne_plots[args._class]['emb'] = emb

        if not hasattr(result.tsne_plots[args._class], 'anomaly_labels'):
            result.tsne_plots[args._class]['anomaly_labels'] = anomaly_labels

        if not hasattr(result.tsne_plots[args._class], 'rot_0_labels'):
            result.tsne_plots[args._class]['rot_0_labels'] = rot_0_labels
        if not hasattr(result.tsne_plots[args._class], 'rot_90_labels'):
            result.tsne_plots[args._class]['rot_90_labels'] = rot_90_labels
        if not hasattr(result.tsne_plots[args._class], 'rot_180_labels'):
            result.tsne_plots[args._class]['rot_180_labels'] = rot_180_labels
        if not hasattr(result.tsne_plots[args._class], 'rot_270_labels'):
            result.tsne_plots[args._class]['rot_270_labels'] = rot_270_labels

        if args._class == 9:
            plt.savefig(f'exp_{args.batch_size}_{args.epochs}_{args.encodingdim}_{args.mlp}_{args.rotation_pred}_{args.use_rotated_data}/plot.png')