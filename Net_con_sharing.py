import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AC_net_beta(nn.Module):
    def __init__(self, action_n, state_n, net_width):
        super(AC_net_beta, self).__init__()
        self.A = nn.Sequential(
            layer_init(nn.Linear(state_n, net_width)),
            nn.Tanh(),
            layer_init(nn.Linear(net_width, net_width)),
            nn.Tanh()
        )
        self.alpha = nn.Sequential(*[layer_init(nn.Linear(net_width, action_n)), nn.Identity()])
        self.beta = nn.Sequential(*[layer_init(nn.Linear(net_width, action_n)), nn.Identity()])
        self.C = nn.Sequential(
            layer_init(nn.Linear(state_n, net_width)),
            nn.Tanh(),
            layer_init(nn.Linear(net_width, net_width)),
            nn.Tanh(),
            layer_init(nn.Linear(net_width, 1)),
            nn.Identity()
        )

    def froward(self, s):
        p = self.A(s)
        alpha = F.softplus(self.alpha(p)) + 1.0
        beta = F.softplus(self.beta(p)) + 1.0
        return alpha, beta

    def get_distri(self, s):
        alpha, beta = self.froward(s)
        distri = Beta(alpha, beta)  # Beta Distribution (0-1) for each dim of action
        return distri

    def get_action(self, s):
        alpha, beta = self.froward(s)
        a = alpha / (alpha + beta)  # 0-1
        return a

    def get_critic(self, s):
        Vs = self.C(s)
        return Vs
