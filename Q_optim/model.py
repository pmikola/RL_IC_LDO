########################### BELLMAN EQUATiON ##############################
# NewQ(state,action) = Q(state,action) + lr[R(state,action) + gamma*maxQ'(state',action') - Q(state,action(]
# NewQ(state,action) - New Q vale for that state and action
# Q(state,action) - current Q value in state and action
# lr - learning rate
# R(state,action) - reward for taking that action and that state
# gamma- discount rate
# maxQ'(state',action') - Maximum expected future reward for given new state and all possible actions at that new state
########################### BELLMAN EQUATiON ##############################


import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcontrib
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR
from nupic.torch.modules import (
    KWinners2d, KWinners, SparseWeights, SparseWeights2d, Flatten,
    rezero_weights, update_boost_strength
)
import nupic.torch.functions as FF

use_cuda = True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")


class Qnet(nn.Module):
    def __init__(self, input_size, no_bits):
        super().__init__()

        self.MemStruct0 = torch.ones(input_size).to(device)
        self.input_size = input_size  # input_size
        self.hidden_size = int(self.input_size / 4)
        self.num_layers = 2
        self.SPARSITY = 0.2
        self.SPARSITY_CNN = 0.2
        self.PERCENT_ON = 0.33
        self.PERCENT_ON_LIN = 0.33
        self.BOOST_STRENGTH = 2.
        self.dropout = 0.1
        self.bidirectional = False
        self.duty_cycle = None
        self.k = 100
        self.break_ties = True
        self.inplace = True
        self.WSPARSITY = 0.4
        self.relu_on = True
        self.fac = 2
        self.data_flow_counter = 0.
        self.l_min = 0.05
        self.l_max = 10
        self.w_min = 0.05
        self.w_max = 50
        self.w_pass_min = 25000.
        self.w_pass_max = 65000.
        self.C_min = 1.
        self.C_max = 25.
        self.R_min = 5.
        self.R_max = 55.
        self.w_hb = 80.
        self.c_1a = 35
        self.c_1b = 35
        self.c_2a = 13
        self.c_2b = 13
        self.k1 = self.k2 = 3
        self.p = 1
        self.s = 1

        self.no_bits = no_bits
        # LATENT NON-LINEAR SHARED SPACE MAPPING

        if self.bidirectional:
            self.hidden_size /= 2
        self.lstmLatent = nn.LSTM(self.input_size,
                                  self.hidden_size,
                                  self.num_layers,
                                  dropout=self.dropout,
                                  bidirectional=self.bidirectional)

        self.convLatentA = SparseWeights2d(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), stride=(1, 1), padding=1),
            sparsity=self.SPARSITY_CNN)
        self.convLAWinn = KWinners2d(channels=14, percent_on=self.PERCENT_ON, relu=self.relu_on,
                                     boost_strength=self.BOOST_STRENGTH)
        self.convLatentB = SparseWeights2d(
            nn.Conv2d(in_channels=14, out_channels=12, kernel_size=(3, 3), stride=(1, 1), padding=1),
            sparsity=self.SPARSITY_CNN)
        self.convLBWinn = KWinners2d(channels=12, percent_on=self.PERCENT_ON, relu=self.relu_on,
                                     boost_strength=self.BOOST_STRENGTH)

        self.linearLatent = nn.Linear(1512, self.hidden_size)
        # self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        # LATENT NON-LINEAR SHARED SPACE MAPPING

        ############### Multiregressor ################

        # self.conv0a = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_1a, out_channels=self.c_2a, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_2a, percent_on=self.PERCENT_ON, boost_strength=self.BOOST_STRENGTH))
        #
        # self.conv0b = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_2b, out_channels=self.c_1b, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_1b, percent_on=self.PERCENT_ON,
        #                boost_strength=self.BOOST_STRENGTH))
        #
        # self.conv1a = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_1a, out_channels=self.c_2a, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_2a, percent_on=self.PERCENT_ON, boost_strength=self.BOOST_STRENGTH))
        # self.conv1b = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_2b, out_channels=self.c_1b, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_1b, percent_on=self.PERCENT_ON,
        #                boost_strength=self.BOOST_STRENGTH))
        # self.conv2a = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_1a, out_channels=self.c_2a, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_2a, percent_on=self.PERCENT_ON, boost_strength=self.BOOST_STRENGTH))
        # self.conv2b = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_2b, out_channels=self.c_1b, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_1b, percent_on=self.PERCENT_ON,
        #                boost_strength=self.BOOST_STRENGTH))
        # self.conv3a = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_1a, out_channels=self.c_2a, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_2a, percent_on=self.PERCENT_ON, boost_strength=self.BOOST_STRENGTH))
        # self.conv3b = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_2b, out_channels=self.c_1b, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_1b, percent_on=self.PERCENT_ON,
        #                boost_strength=self.BOOST_STRENGTH))
        # self.conv4a = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_1a, out_channels=self.c_2a, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_2a, percent_on=self.PERCENT_ON, boost_strength=self.BOOST_STRENGTH))
        # self.conv4b = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_2b, out_channels=self.c_1b, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_1b, percent_on=self.PERCENT_ON,
        #                boost_strength=self.BOOST_STRENGTH))
        # self.conv5a = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_1a, out_channels=self.c_2a, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_2a, percent_on=self.PERCENT_ON, boost_strength=self.BOOST_STRENGTH))
        # self.conv5b = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_2b, out_channels=self.c_1b, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_1b, percent_on=self.PERCENT_ON,
        #                boost_strength=self.BOOST_STRENGTH))
        # self.conv6a = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_1a, out_channels=self.c_2a, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_2a, percent_on=self.PERCENT_ON, boost_strength=self.BOOST_STRENGTH))
        # self.conv6b = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_2b, out_channels=self.c_1b, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_1b, percent_on=self.PERCENT_ON,
        #                boost_strength=self.BOOST_STRENGTH))
        # self.conv7a = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_1a, out_channels=self.c_2a, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_2a, percent_on=self.PERCENT_ON, boost_strength=self.BOOST_STRENGTH))
        # self.conv7b = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_2b, out_channels=self.c_1b, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_1b, percent_on=self.PERCENT_ON,
        #                boost_strength=self.BOOST_STRENGTH))
        # self.conv8a = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_1a, out_channels=self.c_2a, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_2a, percent_on=self.PERCENT_ON, boost_strength=self.BOOST_STRENGTH))
        # self.conv8b = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_2b, out_channels=self.c_1b, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_1b, percent_on=self.PERCENT_ON,
        #                boost_strength=self.BOOST_STRENGTH))
        # self.conv9a = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_1a, out_channels=self.c_2a, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_2a, percent_on=self.PERCENT_ON, boost_strength=self.BOOST_STRENGTH))
        # self.conv9b = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_2b, out_channels=self.c_1b, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_1b, percent_on=self.PERCENT_ON,
        #                boost_strength=self.BOOST_STRENGTH))
        # self.conv10a = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_1a, out_channels=self.c_2a, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_2a, percent_on=self.PERCENT_ON, boost_strength=self.BOOST_STRENGTH))
        # self.conv10b = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_2b, out_channels=self.c_1b, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_1b, percent_on=self.PERCENT_ON,
        #                boost_strength=self.BOOST_STRENGTH))
        # self.conv11a = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_1a, out_channels=self.c_2a, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_2a, percent_on=self.PERCENT_ON, boost_strength=self.BOOST_STRENGTH))
        # self.conv11b = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_2b, out_channels=self.c_1b, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_1b, percent_on=self.PERCENT_ON,
        #                boost_strength=self.BOOST_STRENGTH))
        # self.conv12a = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_1a, out_channels=self.c_2a, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_2a, percent_on=self.PERCENT_ON, boost_strength=self.BOOST_STRENGTH))
        # self.conv12b = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_2b, out_channels=self.c_1b, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_1b, percent_on=self.PERCENT_ON,
        #                boost_strength=self.BOOST_STRENGTH))
        # self.conv13a = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_1a, out_channels=self.c_2a, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_2a, percent_on=self.PERCENT_ON, boost_strength=self.BOOST_STRENGTH))
        # self.conv13b = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_2b, out_channels=self.c_1b, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_1b, percent_on=self.PERCENT_ON,
        #                boost_strength=self.BOOST_STRENGTH))
        # self.conv14a = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_1a, out_channels=self.c_2a, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_2a, percent_on=self.PERCENT_ON, boost_strength=self.BOOST_STRENGTH))
        # self.conv14b = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_2b, out_channels=self.c_1b, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_1b, percent_on=self.PERCENT_ON,
        #                boost_strength=self.BOOST_STRENGTH))
        # self.conv15a = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_1a, out_channels=self.c_2a, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_2a, percent_on=self.PERCENT_ON, boost_strength=self.BOOST_STRENGTH))
        # self.conv15b = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_2b, out_channels=self.c_1b, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_1b, percent_on=self.PERCENT_ON,
        #                boost_strength=self.BOOST_STRENGTH))
        # self.conv16a = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_1a, out_channels=self.c_2a, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_2a, percent_on=self.PERCENT_ON, boost_strength=self.BOOST_STRENGTH))
        # self.conv16b = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_2b, out_channels=self.c_1b, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_1b, percent_on=self.PERCENT_ON,
        #                boost_strength=self.BOOST_STRENGTH))
        # self.conv17a = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_1a, out_channels=self.c_2a, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_2a, percent_on=self.PERCENT_ON, boost_strength=self.BOOST_STRENGTH))
        # self.conv17b = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_2b, out_channels=self.c_1b, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_1b, percent_on=self.PERCENT_ON,
        #                boost_strength=self.BOOST_STRENGTH))
        # self.conv18a = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_1a, out_channels=self.c_2a, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_2a, percent_on=self.PERCENT_ON, boost_strength=self.BOOST_STRENGTH))
        # self.conv18b = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_2b, out_channels=self.c_1b, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_1b, percent_on=self.PERCENT_ON,
        #                boost_strength=self.BOOST_STRENGTH))
        # self.conv19a = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_1a, out_channels=self.c_2a, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_2a, percent_on=self.PERCENT_ON, boost_strength=self.BOOST_STRENGTH))
        # self.conv19b = nn.Sequential(
        #     SparseWeights2d(nn.Conv2d(in_channels=self.c_2b, out_channels=self.c_1b, kernel_size=(self.k1, self.k2),
        #                               stride=(self.s, self.s), padding=self.p), sparsity=self.SPARSITY_CNN),
        #     KWinners2d(channels=self.c_1b, percent_on=self.PERCENT_ON,
        #                boost_strength=self.BOOST_STRENGTH))
        #
        # self.headX0 = nn.Sequential(
        #     SparseWeights(nn.Linear(self.hidden_size, int(self.hidden_size / self.fac)), sparsity=self.SPARSITY),
        #     KWinners(n=int(self.hidden_size / self.fac), percent_on=self.PERCENT_ON,
        #              boost_strength=self.BOOST_STRENGTH))
        #
        self.headX0 = nn.Sequential(
            SparseWeights(nn.Linear(self.hidden_size, int(self.hidden_size / self.fac)), sparsity=self.SPARSITY),
            KWinners(n=int(self.hidden_size / self.fac), percent_on=self.PERCENT_ON_LIN, relu=self.relu_on,
                     boost_strength=self.BOOST_STRENGTH))

        self.headX1 = nn.Sequential(
            SparseWeights(nn.Linear(self.hidden_size, int(self.hidden_size / self.fac)), sparsity=self.SPARSITY),
            KWinners(n=int(self.hidden_size / self.fac), percent_on=self.PERCENT_ON_LIN, relu=self.relu_on,
                     boost_strength=self.BOOST_STRENGTH))
        self.headX2 = nn.Sequential(
            SparseWeights(nn.Linear(self.hidden_size, int(self.hidden_size / self.fac)), sparsity=self.SPARSITY),
            KWinners(n=int(self.hidden_size / self.fac), percent_on=self.PERCENT_ON_LIN, relu=self.relu_on,
                     boost_strength=self.BOOST_STRENGTH))
        self.headX3 = nn.Sequential(
            SparseWeights(nn.Linear(self.hidden_size, int(self.hidden_size / self.fac)), sparsity=self.SPARSITY),
            KWinners(n=int(self.hidden_size / self.fac), percent_on=self.PERCENT_ON_LIN, relu=self.relu_on,
                     boost_strength=self.BOOST_STRENGTH))
        self.headX4 = nn.Sequential(
            SparseWeights(nn.Linear(self.hidden_size, int(self.hidden_size / self.fac)), sparsity=self.SPARSITY),
            KWinners(n=int(self.hidden_size / self.fac), percent_on=self.PERCENT_ON_LIN, relu=self.relu_on,
                     boost_strength=self.BOOST_STRENGTH))
        self.headX5 = nn.Sequential(
            SparseWeights(nn.Linear(self.hidden_size, int(self.hidden_size / self.fac)), sparsity=self.SPARSITY),
            KWinners(n=int(self.hidden_size / self.fac), percent_on=self.PERCENT_ON_LIN, relu=self.relu_on,
                     boost_strength=self.BOOST_STRENGTH))
        self.headX6 = nn.Sequential(
            SparseWeights(nn.Linear(self.hidden_size, int(self.hidden_size / self.fac)), sparsity=self.SPARSITY),
            KWinners(n=int(self.hidden_size / self.fac), percent_on=self.PERCENT_ON_LIN, relu=self.relu_on,
                     boost_strength=self.BOOST_STRENGTH))
        self.headX7 = nn.Sequential(
            SparseWeights(nn.Linear(self.hidden_size, int(self.hidden_size / self.fac)), sparsity=self.SPARSITY),
            KWinners(n=int(self.hidden_size / self.fac), percent_on=self.PERCENT_ON_LIN, relu=self.relu_on,
                     boost_strength=self.BOOST_STRENGTH))
        self.headX8 = nn.Sequential(
            SparseWeights(nn.Linear(self.hidden_size, int(self.hidden_size / self.fac)), sparsity=self.SPARSITY),
            KWinners(n=int(self.hidden_size / self.fac), percent_on=self.PERCENT_ON_LIN, relu=self.relu_on,
                     boost_strength=self.BOOST_STRENGTH))
        self.headX9 = nn.Sequential(
            SparseWeights(nn.Linear(self.hidden_size, int(self.hidden_size / self.fac)), sparsity=self.SPARSITY),
            KWinners(n=int(self.hidden_size / self.fac), percent_on=self.PERCENT_ON_LIN, relu=self.relu_on,
                     boost_strength=self.BOOST_STRENGTH))
        self.headX10 = nn.Sequential(
            SparseWeights(nn.Linear(self.hidden_size, int(self.hidden_size / self.fac)), sparsity=self.SPARSITY),
            KWinners(n=int(self.hidden_size / self.fac), percent_on=self.PERCENT_ON_LIN, relu=self.relu_on,
                     boost_strength=self.BOOST_STRENGTH))
        self.headX11 = nn.Sequential(
            SparseWeights(nn.Linear(self.hidden_size, int(self.hidden_size / self.fac)), sparsity=self.SPARSITY),
            KWinners(n=int(self.hidden_size / self.fac), percent_on=self.PERCENT_ON_LIN, relu=self.relu_on,
                     boost_strength=self.BOOST_STRENGTH))
        self.headX12 = nn.Sequential(
            SparseWeights(nn.Linear(self.hidden_size, int(self.hidden_size / self.fac)), sparsity=self.SPARSITY),
            KWinners(n=int(self.hidden_size / self.fac), percent_on=self.PERCENT_ON_LIN, relu=self.relu_on,
                     boost_strength=self.BOOST_STRENGTH))
        self.headX13 = nn.Sequential(
            SparseWeights(nn.Linear(self.hidden_size, int(self.hidden_size / self.fac)), sparsity=self.SPARSITY),
            KWinners(n=int(self.hidden_size / self.fac), percent_on=self.PERCENT_ON_LIN, relu=self.relu_on,
                     boost_strength=self.BOOST_STRENGTH))
        self.headX14 = nn.Sequential(
            SparseWeights(nn.Linear(self.hidden_size, int(self.hidden_size / self.fac)), sparsity=self.SPARSITY),
            KWinners(n=int(self.hidden_size / self.fac), percent_on=self.PERCENT_ON_LIN, relu=self.relu_on,
                     boost_strength=self.BOOST_STRENGTH))
        self.headX15 = nn.Sequential(
            SparseWeights(nn.Linear(self.hidden_size, int(self.hidden_size / self.fac)), sparsity=self.SPARSITY),
            KWinners(n=int(self.hidden_size / self.fac), percent_on=self.PERCENT_ON_LIN, relu=self.relu_on,
                     boost_strength=self.BOOST_STRENGTH))
        self.headX16 = nn.Sequential(
            SparseWeights(nn.Linear(self.hidden_size, int(self.hidden_size / self.fac)), sparsity=self.SPARSITY),
            KWinners(n=int(self.hidden_size / self.fac), percent_on=self.PERCENT_ON_LIN, relu=self.relu_on,
                     boost_strength=self.BOOST_STRENGTH))
        self.headX17 = nn.Sequential(
            SparseWeights(nn.Linear(self.hidden_size, int(self.hidden_size / self.fac)), sparsity=self.SPARSITY),
            KWinners(n=int(self.hidden_size / self.fac), percent_on=self.PERCENT_ON_LIN, relu=self.relu_on,
                     boost_strength=self.BOOST_STRENGTH))
        self.headX18 = nn.Sequential(
            SparseWeights(nn.Linear(self.hidden_size, int(self.hidden_size / self.fac)), sparsity=self.SPARSITY),
            KWinners(n=int(self.hidden_size / self.fac), percent_on=self.PERCENT_ON_LIN, relu=self.relu_on,
                     boost_strength=self.BOOST_STRENGTH))
        self.headX19 = nn.Sequential(
            SparseWeights(nn.Linear(self.hidden_size, int(self.hidden_size / self.fac)), sparsity=self.SPARSITY),
            KWinners(n=int(self.hidden_size / self.fac), percent_on=self.PERCENT_ON_LIN, relu=self.relu_on,
                     boost_strength=self.BOOST_STRENGTH))

        # self.headX0 = nn.Linear(self.hidden_size, int(self.hidden_size / self.fac))
        # self.headX1 = nn.Linear(self.hidden_size, int(self.hidden_size / self.fac))
        # self.headX2 = nn.Linear(self.hidden_size, int(self.hidden_size / self.fac))
        # self.headX3 = nn.Linear(self.hidden_size, int(self.hidden_size / self.fac))
        # self.headX4 = nn.Linear(self.hidden_size, int(self.hidden_size / self.fac))
        # self.headX5 = nn.Linear(self.hidden_size, int(self.hidden_size / self.fac))
        # self.headX6 = nn.Linear(self.hidden_size, int(self.hidden_size / self.fac))
        # self.headX7 = nn.Linear(self.hidden_size, int(self.hidden_size / self.fac))
        # self.headX8 = nn.Linear(self.hidden_size, int(self.hidden_size / self.fac))
        # self.headX9 = nn.Linear(self.hidden_size, int(self.hidden_size / self.fac))
        # self.headX10 = nn.Linear(self.hidden_size, int(self.hidden_size / self.fac))
        # self.headX11 = nn.Linear(self.hidden_size, int(self.hidden_size / self.fac))
        # self.headX12 = nn.Linear(self.hidden_size, int(self.hidden_size / self.fac))
        # self.headX13 = nn.Linear(self.hidden_size, int(self.hidden_size / self.fac))
        # self.headX14 = nn.Linear(self.hidden_size, int(self.hidden_size / self.fac))
        # self.headX15 = nn.Linear(self.hidden_size, int(self.hidden_size / self.fac))
        # self.headX16 = nn.Linear(self.hidden_size, int(self.hidden_size / self.fac))
        # self.headX17 = nn.Linear(self.hidden_size, int(self.hidden_size / self.fac))
        # self.headX18 = nn.Linear(self.hidden_size, int(self.hidden_size / self.fac))
        # self.headX19 = nn.Linear(self.hidden_size, int(self.hidden_size / self.fac))

        # REGRESSOR FOR W Values x 9 W vals
        self.headW0 = nn.Linear(int(self.hidden_size / self.fac), self.no_bits)
        self.headW1 = nn.Linear(int(self.hidden_size / self.fac), self.no_bits)
        self.headW2 = nn.Linear(int(self.hidden_size / self.fac), self.no_bits)
        self.headW3 = nn.Linear(int(self.hidden_size / self.fac), self.no_bits)
        self.headW4 = nn.Linear(int(self.hidden_size / self.fac), self.no_bits)
        self.headW5 = nn.Linear(int(self.hidden_size / self.fac), self.no_bits)
        self.headW6 = nn.Linear(int(self.hidden_size / self.fac), self.no_bits)
        self.headW7 = nn.Linear(int(self.hidden_size / self.fac), self.no_bits)
        self.headW8 = nn.Linear(int(self.hidden_size / self.fac), self.no_bits)
        # REGRESSOR FOR L Values  x 9 W vals
        self.headL0 = nn.Linear(int(self.hidden_size / self.fac), self.no_bits)
        self.headL1 = nn.Linear(int(self.hidden_size / self.fac), self.no_bits)
        self.headL2 = nn.Linear(int(self.hidden_size / self.fac), self.no_bits)
        self.headL3 = nn.Linear(int(self.hidden_size / self.fac), self.no_bits)
        self.headL4 = nn.Linear(int(self.hidden_size / self.fac), self.no_bits)
        self.headL5 = nn.Linear(int(self.hidden_size / self.fac), self.no_bits)
        self.headL6 = nn.Linear(int(self.hidden_size / self.fac), self.no_bits)
        self.headL7 = nn.Linear(int(self.hidden_size / self.fac), self.no_bits)
        self.headL8 = nn.Linear(int(self.hidden_size / self.fac), self.no_bits)
        # REGRESSOR FOR C Value
        self.headC0 = nn.Linear(int(self.hidden_size / self.fac), self.no_bits)
        # REGRESSOR FOR R Value
        self.headR0 = nn.Linear(int(self.hidden_size / self.fac), self.no_bits)
        ############### Multiregressor ################

        self.apply(self.__init__weights)
        self.mem_init()


    def __init__weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=np.sqrt(6) / np.sqrt(self.input_size))
            if module.bias is not None:
                module.bias.data.zero_()

    def set_range(self, input, max, min):
        try:
            range = (min - max) * ((input - torch.min(input)) / (torch.max(input) * torch.min(input))) + max

        except ZeroDivisionError:
            range = torch.tensor(0.05)

        if torch.isnan(range):
            range = torch.tensor(0.05)
        return [range]

    def mem_init(self):
        self.h_0 = Variable(torch.zeros(self.num_layers, 1, int(self.input_size/4)).to(device))
        self.c_0 = Variable(torch.zeros(self.num_layers, 1, int(self.input_size/4)).to(device))

    def forward(self, x):
        # print(x.size())

        self.data_flow_counter += 1.
        x = x.reshape([1,1,x.size(0)])

        # print(h_0.size())
        # print(c_0.size())
        # print(x.size())
        output, (self.h_0, self.c_0) = self.lstmLatent(x, (self.h_0, self.c_0))
        # torch.set_printoptions(profile="full")
        # self.MemStruct0 = torch.add(x, self.MemStruct0) / torch.max(self.MemStruct0)
        x = self.h_0.reshape(14, 10, 3, 3)
        x = self.convLAWinn(self.convLatentA(x))
        # x = F.layer_norm(x,[14, 3, 3])
        x = self.convLBWinn(self.convLatentB(x))
        # print(self.MemStruct1)
        # x = torch.flatten(torch.cat([x, self.MemStruct0]))

        #print(x.size())
        x = torch.flatten(x)

        x = self.linearLatent(x)
        # x0 = torch.flatten(self.conv0b(self.conv0a(x)))
        x0 = self.headX0(x)
        W0 = self.headW0(x0)
        W0c = self.w_max

        # x1 = torch.flatten(self.conv1b(self.conv1a(x)))
        x1 = self.headX1(x)
        W1 = self.headW1(x1)
        W1c = self.w_hb

        # x2 = torch.flatten(self.conv2b(self.conv2a(x)))
        x2 = self.headX2(x)
        W2 = self.headW2(x2)
        W2c = self.w_max

        # x3 = torch.flatten(self.conv3b(self.conv3a(x)))
        x3 = self.headX3(x)
        W3 = self.headW3(x3)
        W3c = self.w_max

        # x4 = torch.flatten(self.conv4b(self.conv4a(x)))
        x4 = self.headX4(x)
        W4 = self.headW4(x4)
        W4c = self.w_max

        # x5 = torch.flatten(self.conv5b(self.conv5a(x)))
        x5 = self.headX5(x)
        W5 = self.headW5(x5)
        W5c = self.w_hb

        # x6 = torch.flatten(self.conv6b(self.conv6a(x)))
        x6 = self.headX6(x)
        W6 = self.headW6(x6)
        W6c = self.w_max

        # x7 = torch.flatten(self.conv7b(self.conv7a(x)))
        x7 = self.headX7(x)
        W7 = self.headW7(x7)
        W7c = self.w_max

        # x8 = torch.flatten(self.conv8b(self.conv8a(x)))
        x8 = self.headX8(x)
        W8 = self.headW8(x8)
        W8c = self.w_pass_max

        # x9 = torch.flatten(self.conv9b(self.conv9a(x)))
        x9 = self.headX9(x)
        L0 = self.headL0(x9)
        L0c = self.l_max

        # x10 = torch.flatten(self.conv10b(self.conv10a(x)))
        x10 = self.headX10(x)
        L1 = self.headL1(x10)
        L1c = self.l_max

        # x11 = torch.flatten(self.conv11b(self.conv11a(x)))
        x11 = self.headX11(x)
        L2 = self.headL2(x11)
        L2c = self.l_max

        # x12 = torch.flatten(self.conv12b(self.conv12a(x)))
        x12 = self.headX12(x)
        L3 = self.headL3(x12)
        L3c = self.l_max

        # x13 = torch.flatten(self.conv13b(self.conv13a(x)))
        x13 = self.headX13(x)
        L4 = self.headL4(x13)
        L4c = self.l_max

        # x14 = torch.flatten(self.conv14b(self.conv14a(x)))
        x14 = self.headX14(x)
        L5 = self.headL5(x14)
        L5c = self.l_max

        # x15 = torch.flatten(self.conv15b(self.conv15a(x)))
        x15 = self.headX15(x)
        L6 = self.headL6(x15)
        L6c = self.l_max

        # x16 = torch.flatten(self.conv16b(self.conv16a(x)))
        x16 = self.headX16(x)
        L7 = self.headL7(x16)
        L7c = self.l_max

        # x17 = torch.flatten(self.conv17b(self.conv17a(x)))
        x17 = self.headX17(x)
        L8 = self.headL8(x17)
        L8c = self.l_max

        # x18 = torch.flatten(self.conv18b(self.conv18a(x)))
        x18 = self.headX18(x)
        C0 = self.headC0(x18)
        C0c = self.C_max

        # x19 = torch.flatten(self.conv19b(self.conv19a(x)))
        x19 = self.headX19(x)
        R0 = self.headR0(x19)
        R0c = self.R_max

        self.const = [W0c, W1c, W2c, W3c, W4c, W5c, W6c, W7c, W8c, L0c, L1c, L2c, L3c, L4c, L5c, L6c, L7c, L8c, C0c,
                      R0c]
        return W0, W1, W2, W3, W4, W5, W6, W7, W8, L0, L1, L2, L3, L4, L5, L6, L7, L8, C0, R0

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        else:
            pass
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class Qtrainer:
    def __init__(self, model, lr, alpha, gamma):
        self.lr = lr
        self.gamma = gamma
        self.alpha = alpha
        self.model = model
        self.loss_list = []
        self.no_heads = 20
        self.no_enhancments = int(
            self.model.no_bits * 0.5)  # smaller => slower training but more robust (just like in k-winners)
        self.treshold_bce = nn.Hardtanh(0., 1.)
        # self.swa_model = AveragedModel(model)
        # self.optim = torch.optim.SGD(model.parameters(), lr=self.lr)
        self.optim = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=2e-4, amsgrad=True)
        # self.optim = torch.optim.RMSprop(model.parameters(),lr=self.lr,weight_decay=2e-3)
        # self.scheduler = CosineAnnealingLR(self.optim, T_max=100)
        self.criterion = nn.MSELoss(reduce='mean')  # reduce='sum')
        # pos_weight = torch.full([self.model.no_bits], 20.).to(device)
        # self.criterion = nn.BCEWithLogitsLoss(reduce='mean')#, pos_weight=pos_weight)

    def train_step(self, state, action, reward, next_state):
        global loss, loss_t

        state = torch.tensor(np.array(state), dtype=torch.float32).to(device)
        action = torch.tensor(np.array(action), dtype=torch.float32).to(device)
        reward = torch.tensor(np.array(reward), dtype=torch.float32).to(device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32).to(device)
        # game_over = torch.tensor(np.array(game_over), dtype=torch.float).to(device)
        # print("state:", state, state.size())
        # print("action:", action, action.size())
        # print("reward:", reward)
        # print("next_state:", next_state.size())

        self.optim.zero_grad()
        torch.set_grad_enabled(True)
        self.model.train()

        target = torch.zeros((self.no_heads, self.model.no_bits)).to(device)

        ################# Standard Q Learning Equation #####################
        # Qnew (state(t),action(t)) <-
        # <- Q(state(t),action(t)) + alpha * (REWARD(t) + gamma *
        #  * maxQ(state(t+1),actions) - Q(state(t),action(t))
        ################# Standard Q Learrning Equation #####################

        for idx in range(0, int(np.array(state.cpu().detach().numpy()).shape[0])):
            # Q_s = F.normalize(Q_s, dim=0)
            prediction = self.model(state[idx])
            next_prediction = self.model(next_state[idx])
            for i in range(0, self.no_heads):
                target[i] = prediction[i].clone()
            for jdx in range(0, 20):
                next_pred_max, next_pred_argmax = torch.topk(next_prediction[jdx], self.no_enhancments)
                Q_new = self.alpha * (reward[idx] + self.gamma * next_pred_max)
                action_max, action_argmax = torch.topk(action[idx][jdx], self.no_enhancments)

                target[jdx][action_argmax] = Q_new

                # print(" |||| state |XX|", state[idx], " |XX| target |XX|", target[idx], " |XX| preds |XX|",
                #       prediction[idx], " |XX| Q_new |XX|", Q_new, " |XX| R |XX|", reward[idx], " |XX| ACTION |XX|",
                #       action[0][idx], " |||| ")

                # target[jdx] = self.treshold_bce(target[jdx])
                if jdx == 0:
                    loss = self.criterion(target[jdx], prediction[jdx])
                    loss = Variable(loss, requires_grad=True)
                else:
                    loss_t = self.criterion(target[jdx], prediction[jdx])
                    loss = loss + Variable(loss_t, requires_grad=True)
            # print(target)
            if reward[idx] == torch.max(reward):
                # print(reward[idx])
                ##for kdx in range(0,reward.size(0)):
                # loss = loss / loss
                self.loss_list.append(loss.item())
                loss.backward()
                self.optim.step()
                self.model.apply(update_boost_strength)
                self.model.apply(rezero_weights)
            else:
                self.loss_list.append(loss.item())
                loss.backward()
                self.optim.step()
                self.model.apply(update_boost_strength)
                self.model.apply(rezero_weights)
