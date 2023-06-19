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
    # dataset = CIFAR10(root='.', train=True, download=True)
    transform = augmentations.TrainTransform()
    # # transform = ToTensor()
    # inlier_dataset = OneClassDataset(dataset, one_class_labels=inlier, transform=transform, with_rotation=False, augmentation=False)
    # outlier_dataset = OneClassDataset(dataset, zero_class_labels=outlier, transform=transform, with_rotation=False, augmentation=False)
    # train_inlier_dataset = Subset(inlier_dataset, range(0, (int)(.7 * len(inlier_dataset))))
    # train_dataset = train_inlier_dataset
    # validation_inlier_dataset = Subset(inlier_dataset, range((int)(.7 * len(inlier_dataset)), len(inlier_dataset)))
    # validation_dataset = ConcatDataset([validation_inlier_dataset, outlier_dataset])

    score_sum = None
    for rotation in range(4):
        cifar10_train = CIFAR10(root='.', train=True, download=True)
        cifar10_test = CIFAR10(root='.', train=False, download=True)
        train_dataset = OneClassDataset(cifar10_train, one_class_labels=inlier, transform=transform, with_rotation=False, augmentation=False, rotation=rotation)
        test_dataset = ConcatDataset([OneClassDataset(cifar10_train, zero_class_labels=outlier, transform=transform, with_rotation=False, augmentation=False, rotation=rotation),
                                      OneClassDataset(cifar10_test, one_class_labels= inlier, zero_class_labels=outlier, transform=transform, with_rotation=False, augmentation=False, rotation=rotation)])

        with torch.no_grad():
            model.backbone_1.eval()

            training_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
            # validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
            test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

            train_x = []
            with torch.no_grad():
                for x, l in training_dataloader:
                    x = x.cuda()
                    x, _ = model.backbone_1(x)
                    train_x.append(normalize(x, dim=1))
                    # train_x.append(x)
            train_x = torch.cat(train_x)
            gamma = (10 / (torch.var(train_x).item() * train_x.shape[1]))
            svm = OneClassSVM(kernel='rbf', gamma=gamma).fit(train_x.cpu().numpy())
            # svm = OneClassSVM(kernel='linear').fit(train_x.cpu().numpy())

            val_x = []
            labels = []
            with torch.no_grad():
                for x, l in test_dataset:
                    x = x.cuda()
                    x, _ = model.backbone_1(x)
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

        # train_aug_rot = []
        # for (x, _), l in train_aug_rot_dataloader:
        #     x = x.cuda()
        #     x, _ = model.backbone_1(x)
        #     train_aug_rot.append(x)
        #
        # train_aug_rot = torch.cat(train_aug_rot)
        #
        # mean = torch.mean(train_aug_rot, dim=0)
        # std = torch.std(train_aug_rot, dim=0)
        #
        # train_x = []
        # for x, l in training_dataloader:
        #     x = x.cuda()
        #     x, _ = model.backbone_1(x)
        #     x = (x - mean)/std
        #     train_x.append(x)
        #
        # train_x = torch.cat(train_x)
        #
        # cosine_sim = CosineSimilarity(dim=-1)
        # labels = []
        # cos_sim = []
        # for x, l in test_dataset:
        #     x = x.cuda()
        #     x, _ = model.backbone_1(x)
        #     x = (x - mean)/std
        #     cos_sim.append(torch.sum(cosine_sim(train_x, x.unsqueeze(1)), dim=1).values)
        #     labels.append(l)
        #
        # cos_sim = torch.cat(cos_sim).cpu().numpy()
        # labels = torch.cat(labels).cpu().numpy()
        #
        # roc = roc_auc_score(labels, cos_sim)
        # print(f'class {args._class}: roc: {roc}')

        #
        # train_x = []
        # for (x, _), l in training_dataloader:
        #     train_x.append(model.backbone_1(x.cuda()))
        #
        # train_x = torch.cat(train_x)
        #
        # score = []
        # label = []
        # for (x, _), l in validation_dataloader:
        #     score.append(torch.mean(torch.topk(cdist(train_x, model.backbone_1(x.cuda())), dim=0, k=100, largest=False).values, dim=0))
        #     label.append(l)
        #
        # score = torch.cat(score).cpu().numpy()
        # label = torch.cat(label).cpu().numpy()
        #
        # roc = roc_auc_score(label, score)
        # print(f'class {args._class}: roc: {roc}')


        # score = []
        # score2 = []
        # labels = []
        # # for (x, y), l in validation_dataloader:
        # #     batch_size = x.shape[0]
        # #     x = x.cuda()
        # #     x = x.reshape((batch_size * aug,) + tuple(x.shape[-3:], ))
        # #     x = model.backbone(x)
        # #     x = x.reshape(batch_size, aug, x.shape[-1])
        # #     y = y.cuda()
        # #     y = y.reshape((batch_size * aug,) + tuple(y.shape[-3:], ))
        # #     y = model.backbone(y)
        # #     y = y.reshape(batch_size, aug, y.shape[-1])
        # #
        # #     score.append(torch.norm(x - y, dim=-1).mean(dim=-1))
        # #     labels.append(l)
        #
        # for (x, y), l in validation_dataloader:
        #     x = x.cuda()
        #     x = model.backbone_1(x)
        #     y = y.cuda()
        #     y = model.backbone_2(y)
        #
        #     score.append(torch.norm(x - y, dim=-1))
        #     score2.append(torch.cosine_similarity(x, y))
        #     labels.append(l)
        #
        # score = torch.cat(score).cpu().numpy()
        # score2 = torch.cat(score2).cpu().numpy()
        # labels = torch.cat(labels).cpu().numpy()
        #
        # roc = roc_auc_score(labels, -score)
        # roc2 = roc_auc_score(labels, score2)
        # print(f'class {args._class}: roc: {roc}, roc: {roc2}')
        #
    if showTSNE:
        visual_tsne(model, args, roc, result)
    return roc

def visual_tsne(model, args, roc, result):
    aug = 10
    inlier = [args._class]
    outlier = list(range(10))
    outlier.remove(args._class)
    # dataset = CIFAR10(root='../', train=True, download=True)
    transform = augmentations.TrainTransform()
    # # transform = ToTensor()
    # inlier_dataset = OneClassDataset(dataset, one_class_labels=inlier, transform=transform, with_rotation=True, augmentation=False)
    # outlier_dataset = OneClassDataset(dataset, zero_class_labels=outlier, transform=transform, with_rotation=True, augmentation=False)
    # train_inlier_dataset = Subset(inlier_dataset, range(0, (int)(.7 * len(inlier_dataset))))
    # train_dataset = train_inlier_dataset
    # validation_inlier_dataset = Subset(inlier_dataset, range((int)(.7 * len(inlier_dataset)), len(inlier_dataset)))
    # validation_dataset = ConcatDataset(
    #     [validation_inlier_dataset, Subset(outlier_dataset, range(0, (int)(len(validation_inlier_dataset)/4)))])

    cifar10_test = CIFAR10(root='.', train=False, download=True)
    inlier_dataset = Subset(OneClassDataset(cifar10_test, one_class_labels=inlier, transform=transform, with_rotation=True, augmentation=False), range(0, 1000*4))
    outlier_dataset = Subset(OneClassDataset(cifar10_test, zero_class_labels=outlier, transform=transform, with_rotation=True, augmentation=False), range(0, (int)(len(inlier_dataset))))
    # inlier_dataset = Subset(OneClassDataset(cifar10_test, one_class_labels=inlier, transform=transform, with_rotation=False, augmentation=False), range(0, 1000))
    # outlier_dataset = Subset(OneClassDataset(cifar10_test, zero_class_labels=outlier, transform=transform, with_rotation=False, augmentation=False), range(0, (int)(len(inlier_dataset))))
    validation_dataset = ConcatDataset([inlier_dataset, outlier_dataset])

    with torch.no_grad():
        model.eval()

        validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size,
                                                            num_workers=args.num_workers, pin_memory=True, shuffle=True)

        samples = []
        labels = []
        for x, l in validation_dataloader:
            x = x.cuda()
            samples.append(model.backbone_1(x)[0])
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

        # fig, ax = plt.subplots()
        # fig.set_figwidth(10)
        # fig.set_figheight(10)

        # ax.scatter(emb[nominal_labels, 0], emb[nominal_labels, 1], label='normal', c='g', marker='.')
        # ax.scatter(emb[anomaly_labels, 0], emb[anomaly_labels, 1], label='anomaly', c='r', marker='.')
        # ax.legend()

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

        # produce a legend with the unique colors from the scatter

        if args._class == 9:
           plt.show()

        # plt.show()
