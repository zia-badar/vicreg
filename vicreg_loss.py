import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss


class vicreg_loss(nn.Module):
    def __init__(self, rotation_prediction, embedding_size, sim_coeff=25, std_coeff=25, cov_coeff=1):
        super(vicreg_loss, self).__init__()
        self.rotation_prediction = rotation_prediction
        self.num_features = embedding_size
        self.cross_entropy_loss = CrossEntropyLoss()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, x, y, c_x, c_y, l):
        if not hasattr(self, 'batch_size'):
            self.batch_size = x.shape[0]

        l = F.one_hot(l-1, num_classes = 4).cuda().to(torch.float)
        rot_loss = 0
        if self.rotation_prediction:
            rot_loss = (self.cross_entropy_loss(c_x, l) + self.cross_entropy_loss(c_y, l))/2

        repr_loss = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.batch_size - 1)
        cov_y = (y.T @ y) / (self.batch_size - 1)
        cov_loss = vicreg_loss.off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + vicreg_loss.off_diagonal(cov_y).pow_(2).sum().div(self.num_features)
        # # print(f'{self.args.sim_coeff * repr_loss} , {self.args.std_coeff * std_loss} , {self.args.cov_coeff * cov_loss}, {class_aug_loss}')

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
            + rot_loss
        )
        return [self.sim_coeff * repr_loss.item(), self.std_coeff * std_loss.item(), self.cov_coeff * cov_loss.item(), rot_loss.item()], loss

    def off_diagonal(x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
