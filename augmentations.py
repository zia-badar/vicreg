# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torchvision
from PIL import ImageOps, ImageFilter
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class TrainTransform(object):
    def __init__(self, mean, std):
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    32, interpolation=InterpolationMode.BICUBIC
                ),
                # transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                # GaussianBlur(p=1.0),
                # Solarization(p=0.0),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=mean, std=std
                ),
            ]
        )
        self.transform_prime = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    32, interpolation=InterpolationMode.BICUBIC
                ),
                # transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                # GaussianBlur(p=0.1),
                # Solarization(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean = mean, std = std
                ),
            ]
        )

        self.hflip = transforms.RandomHorizontalFlip(0.5)

    # def __call__(self, sample, rot):
    #     x1 = self.transform(rotate(self.hflip(sample), rot, interpolation=torchvision.transforms.InterpolationMode.BICUBIC))
    #     x2 = self.transform_prime(rotate(self.hflip(sample), rot, interpolation=torchvision.transforms.InterpolationMode.BICUBIC))
    #     return x1, x2

    def __call__(self, sample1, sample2):
        x1 = self.transform(sample1)
        x2 = self.transform_prime(sample2)

        return x1, x2
