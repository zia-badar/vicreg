import torch
import torchvision.transforms
from torch.distributions import MultivariateNormal
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torchvision.transforms.functional import rotate


class OneClassDataset(Dataset):
    def __init__(self, dataset: Dataset, one_class_labels=[], zero_class_labels=[], transform=None, augmentation=True, with_rotation=True):
        self.dataset = dataset
        self.one_class_labels = one_class_labels
        self.transform = transform
        self.filtered_indexes = []
        self.augmentation = augmentation
        self.with_rotation = with_rotation

        valid_labels = one_class_labels + zero_class_labels
        for i, (x, l) in enumerate(self.dataset):
            if l in valid_labels:
                self.filtered_indexes.append(i)

        # transform = Compose([ToTensor(), Resize((32, 32)), Normalize(mean=(0.5), std=(0.5))])
        to_tensor = ToTensor()
        self.xs = []
        self.ls = []
        for findex in self.filtered_indexes:
            x, l = self.dataset[findex]
            self.xs.append(x)
            self.ls.append(l)

        # self.xs = torch.stack(self.xs)
        self.ls = torch.tensor(self.ls)

        # self.to_tensor = ToTensor()
        self.to_tensor = Compose([ToTensor(), transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )])
        self.rotations = [0, 90, 180, 270]

    def __getitem__(self, item):

        index = (int)(item/4) if self.with_rotation else item
        x = self.xs[index]
        l = 1 if self.ls[index] in self.one_class_labels else 0

        if self.with_rotation:
            x = rotate(x, self.rotations[item%4], interpolation=torchvision.transforms.InterpolationMode.BICUBIC)

        # xs, ys = [], []
        # for _ in range(self.aug_count):
        #     _x, _y = self.transform(x)
        #     xs.append(_x)
        #     ys.append(_y)
        #
        # xs = torch.stack(xs)
        # ys = torch.stack(ys)

        if self.with_rotation:
            if l == 1:
                l = item%4 + 1

        return self.transform(x) if self.augmentation else self.to_tensor(x), l

    def __len__(self):
        return 4*len(self.filtered_indexes) if self.with_rotation else len(self.filtered_indexes)

class ProjectedDataset(Dataset):

    def __init__(self, train, distribution, projection):
        super(ProjectedDataset, self).__init__()

        projection.eval()
        with torch.no_grad():
            if train:
                self.dataset = projection(distribution.sample(sample_shape=(5000,)))
            else:
                self.dataset = projection(distribution.sample(sample_shape=(1500,)))

    def __getitem__(self, item):

        return self.dataset[item], 0

    def __len__(self):
        return len(self.dataset)
