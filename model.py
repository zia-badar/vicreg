import torch
from torch import nn
from torchvision.models import resnet18


class Model(nn.Module):
    def __init__(self, encoding_dim, expander_arch):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)

        self.f = nn.Sequential(*self.f)
        self.e = nn.Sequential(nn.Linear(512, encoding_dim))

        mlp_spec = f"{encoding_dim}-{expander_arch}"
        layers = []
        f = list(map(int, mlp_spec.split("-")))
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            layers.append(nn.BatchNorm1d(f[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1], bias=False))

        self.h = nn.Sequential(*layers)

    def forward(self, x):
        y = torch.squeeze(self.f(x))
        e = self.e(y)
        z = self.h(e)

        return y, e, z