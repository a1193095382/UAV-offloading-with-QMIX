import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.io as scio
from pulp import *
from collections import Counter
from tensorboardX import SummaryWriter

N_STATES=24+50+3+20
n_agents=3
N_obs=12

qmix_hidden_dim=256
rnn_hidden_dim=256
n_actions=8*3+1
BATCH_SIZE=256
timelength=40
class QMixNet(nn.Module):
    def __init__(self,):
        super(QMixNet, self).__init__()
        # 因为生成的hyper_w1需要是一个矩阵，而pytorch神经网络只能输出一个向量，
        # 所以就先输出长度为需要的 矩阵行*矩阵列 的向量，然后再转化成矩阵

        # args.n_agents是使用hyper_w1作为参数的网络的输入维度，args.qmix_hidden_dim是网络隐藏层参数个数
        # 从而经过hyper_w1得到(经验条数，args.n_agents * args.qmix_hidden_dim)的矩阵
        # if args.two_hyper_layers:
        self.hyper_w1 = nn.Sequential(nn.Linear(N_STATES, rnn_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(rnn_hidden_dim, n_agents * qmix_hidden_dim))
        # self.hyper_w1.weight.data.normal_(0, 0.1)  # initialization
        #     # 经过hyper_w2得到(经验条数, 1)的矩阵
        self.hyper_w2 = nn.Sequential(nn.Linear(N_STATES, rnn_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(rnn_hidden_dim, qmix_hidden_dim))
        # self.hyper_w1.weight.data.normal_(0, 0.1)  # initialization
        # else:
        # self.hyper_w1 = nn.Linear(N_STATES, n_agents * qmix_hidden_dim)
        #     # 经过hyper_w2得到(经验条数, 1)的矩阵
        # self.hyper_w2 = nn.Linear(N_STATES, qmix_hidden_dim * 1)

        # hyper_w1得到的(经验条数，args.qmix_hidden_dim)矩阵需要同样维度的hyper_b1
        self.hyper_b1 = nn.Linear(N_STATES, qmix_hidden_dim)
        # hyper_w2得到的(经验条数，1)的矩阵需要同样维度的hyper_b1
        self.hyper_b2 =nn.Sequential(nn.Linear(N_STATES, qmix_hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(qmix_hidden_dim, 1)
                                     )

    def forward(self, q_values, states):  # states的shape为(episode_num, max_episode_len， state_shape)
        # 传入的q_values是三维的，shape为(episode_num, max_episode_len， n_agents)
        q_values = q_values.view(-1, 1, n_agents)
        states = states.view(-1, N_STATES)

        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)  # (1920, 32)

        w1 = w1.view(-1, n_agents, qmix_hidden_dim)
        b1 = b1.view(-1, 1, qmix_hidden_dim)

        hidden = F.elu(torch.bmm(q_values, w1) + b1)

        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)

        w2 = w2.view(-1, qmix_hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2
        q_total = q_total.view(BATCH_SIZE*timelength, -1)
        return q_total


class RNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self,):
        super(RNN, self).__init__()
        self.fc1 = nn.Linear(N_obs+1, rnn_hidden_dim)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
        self.fc2 = nn.Linear(rnn_hidden_dim, n_actions)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x, h):
        x = F.relu(self.fc1(x))
        h = h.reshape(-1, rnn_hidden_dim)
        h = self.rnn(x, h)
        q = self.fc2(h)
        return q, h

    def init_hidden_state(self, training=None):

        if training is True:
            return torch.zeros([1, BATCH_SIZE, rnn_hidden_dim]), torch.zeros([1, BATCH_SIZE, rnn_hidden_dim])
        else:
            return torch.zeros([1, 1, rnn_hidden_dim]), torch.zeros([1, 1, rnn_hidden_dim])

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 128)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization二次分布生成数据
        self.fc2 = nn.Linear(128, 256)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization二次分布生成数据
        self.fc3 = nn.Linear(256, 128)
        self.fc3.weight.data.normal_(0, 0.1)  # initialization二次分布生成数据
        self.out = nn.Linear(128,n_actions)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # 激励函数
        x = F.relu(x)
        x = self.fc3(x)  # 激励函数
        x = F.relu(x)
        actions_value = self.out(x)  # Q值
        return actions_value