import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable
from torch.nn import HuberLoss, SmoothL1Loss

class GLoss(nn.Module):
    def __init__(self, 
                device, 
                downsample_ratios: Iterable=[2, 4, 5],
                avg_pool_coeffs: Iterable=[0., 0., 0.],
                max_pool_coeffs: Iterable=[0., 0., 0.],
                splitting_coeffs: Iterable=[1 / 4., 1 / 16., 0.],
                p: int=1, EPS: float=1e-17,
                delta: float=1.35,
                beta: float=1e-6):
        super(GLoss, self).__init__()
        self.device = device
        self.gradient_kernel = torch.tensor([
                [[[-1, 0, 0], [0, 1, 0], [0, 0, 0]]],
                [[[0, -1, 0], [0, 1, 0], [0, 0, 0]]],
                [[[0, 0, -1], [0, 1, 0], [0, 0, 0]]],
                [[[0, 0, 0], [-1, 1, 0], [0, 0, 0]]],
                [[[0, 0, 0], [0, 1, -1], [0, 0, 0]]],
                [[[0, 0, 0], [0, 1, 0], [-1, 0, 0]]],
                [[[0, 0, 0], [0, 1, 0], [0, -1, 0]]],
                [[[0, 0, 0], [0, 1, 0], [0, 0, -1]]]
        ]).float().to(device)
        self.downsample_ratios = downsample_ratios
        self.avg_pool_coeffs = avg_pool_coeffs
        self.max_pool_coeffs = max_pool_coeffs
        self.splitting_coeffs = splitting_coeffs
        self.p = p
        self.EPS = EPS
        self.delta = delta
        self.beta = beta

    def forward(self, I, I_hat):
        # Calculate pixel loss
        # pixel_loss = torch.abs(I - I_hat).mean()
        pixel_loss = self.p_loss(I, I_hat)

        # Calculate gradient loss
        # gradient_loss = torch.abs(I_grad - I_hat_grad).mean()
        # gradient_loss = self.p_loss(I_grad, I_hat_grad)
        gradient_loss = self.gradient_loss(I, I_hat)

        avg_pool_loss = 0
        max_pool_loss = 0
        split_loss = 0
        # Calculate downsampling losses
        for i, n in enumerate(self.downsample_ratios):
            if self.avg_pool_coeffs[i] > 0:
                avg_pool_loss += self.avg_pool_coeffs[i] * self.gradient_loss(F.avg_pool2d(I, n), F.avg_pool2d(I_hat, n))
            if self.max_pool_coeffs[i] > 0:
                max_pool_loss += self.max_pool_coeffs[i] * self.gradient_loss(F.max_pool2d(I, n), F.max_pool2d(I_hat, n))
            if self.splitting_coeffs[i] > 0:
                split_loss += self.splitting_coeffs[i] * self.gradient_loss(self.splitting(I, n), self.splitting(I_hat, n))

        # Calculate the overall G-Loss
        g_loss = pixel_loss + gradient_loss + avg_pool_loss + max_pool_loss + split_loss
        # print(pixel_loss, gradient_loss, avg_pool_loss, max_pool_loss, split_loss)
        return g_loss

    def gradient_loss(self, I, I_hat):
        # Calculate gradient feature maps
        batch_size, channels, height, width = I.size()
        self.C_G = self.gradient_kernel.expand(-1, channels, -1, -1)
        I_grad = F.conv2d(I, self.C_G, padding=1)
        I_hat_grad = F.conv2d(I_hat, self.C_G, padding=1)

        # Calculate gradient loss
        # gradient_loss = torch.abs(I_grad - I_hat_grad).mean()
        gradient_loss = self.p_loss(I_grad, I_hat_grad)

        return gradient_loss

    def splitting(self, I, n: int=2):
        batch_size, channels, height, width = I.size()
        I_split = I.view(batch_size, channels, height // n, n, width // n, n)
        I_split = I_split.permute(0, 1, 3, 5, 2, 4).contiguous()
        I_split = I_split.view(batch_size, channels * (n ** 2), height // n, width // n)
        return I_split

    def p_loss(self, I, I_hat):
        if self.p == 1:
            p_loss = torch.abs(I - I_hat).mean()
        elif self.p == 2:
            p_loss = torch.sqrt(torch.square(I - I_hat) + self.EPS).mean()
        elif self.p == 'smooth':
            p_loss = SmoothL1Loss(reduction='mean', beta=self.beta)(I, I_hat)
        elif self.p == 'huber':
            p_loss = HuberLoss(reduction='mean', delta=self.delta)(I, I_hat)
        else:
            raise ValueError(f"You can specify p as 1, 2, smooth or huber, not {self.p}")
            
        return p_loss