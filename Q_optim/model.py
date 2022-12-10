import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from numba import cuda, vectorize, guvectorize, jit, njit
from torch.autograd import Variable

use_cuda = True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")


class Qnet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = self.input_size * 2
        self.l_min = 0.05
        self.l_max = 10
        self.w_min = 0.05
        self.w_max = 50
        self.w_pass_min = 10000.
        self.w_pass_max = 25000.
        self.C_min = 1.
        self.C_max = 25.
        self.R_min = 5.
        self.R_max = 55.
        self.w_hb = 80.
        self.c_1 = 40
        self.c_2 = 20
        self.k1 = self.k2 = 3
        self.p = 1
        self.s = 1
        # LATENT NON-LINEAR SHARED SPACE MAPPING
        # TODO : More degrees of freedom
        # TODO : LATENT SPACE FOR EACH OF THE HEADS??!

        self.convLatentA = nn.Conv2d(in_channels=20, out_channels=31, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.convLatentB = nn.Conv2d(in_channels=31, out_channels=20, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.linear1 = nn.Linear(self.input_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        # LATENT NON-LINEAR SHARED SPACE MAPPING

        # TODO : RL Multiregressor vs. MultiLabelClassifier vs. MultiheadBinaryClassifier vs. Mixed

        ############### Multiregressor ################

        self.conv0a = nn.Conv2d(in_channels=self.c_1, out_channels=self.c_2, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv0b = nn.Conv2d(in_channels=self.c_2, out_channels=self.c_1, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv1a = nn.Conv2d(in_channels=self.c_1, out_channels=self.c_2, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv1b = nn.Conv2d(in_channels=self.c_2, out_channels=self.c_1, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv2a = nn.Conv2d(in_channels=self.c_1, out_channels=self.c_2, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv2b = nn.Conv2d(in_channels=self.c_2, out_channels=self.c_1, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv3a = nn.Conv2d(in_channels=self.c_1, out_channels=self.c_2, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv3b = nn.Conv2d(in_channels=self.c_2, out_channels=self.c_1, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv4a = nn.Conv2d(in_channels=self.c_1, out_channels=self.c_2, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv4b = nn.Conv2d(in_channels=self.c_2, out_channels=self.c_1, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv5a = nn.Conv2d(in_channels=self.c_1, out_channels=self.c_2, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv5b = nn.Conv2d(in_channels=self.c_2, out_channels=self.c_1, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv6a = nn.Conv2d(in_channels=self.c_1, out_channels=self.c_2, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv6b = nn.Conv2d(in_channels=self.c_2, out_channels=self.c_1, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv7a = nn.Conv2d(in_channels=self.c_1, out_channels=self.c_2, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv7b = nn.Conv2d(in_channels=self.c_2, out_channels=self.c_1, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv8a = nn.Conv2d(in_channels=self.c_1, out_channels=self.c_2, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv8b = nn.Conv2d(in_channels=self.c_2, out_channels=self.c_1, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv9a = nn.Conv2d(in_channels=self.c_1, out_channels=self.c_2, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv9b = nn.Conv2d(in_channels=self.c_2, out_channels=self.c_1, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv10a = nn.Conv2d(in_channels=self.c_1, out_channels=self.c_2, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv10b = nn.Conv2d(in_channels=self.c_2, out_channels=self.c_1, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv11a = nn.Conv2d(in_channels=self.c_1, out_channels=self.c_2, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv11b = nn.Conv2d(in_channels=self.c_2, out_channels=self.c_1, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv12a = nn.Conv2d(in_channels=self.c_1, out_channels=self.c_2, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv12b = nn.Conv2d(in_channels=self.c_2, out_channels=self.c_1, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv13a = nn.Conv2d(in_channels=self.c_1, out_channels=self.c_2, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv13b = nn.Conv2d(in_channels=self.c_2, out_channels=self.c_1, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv14a = nn.Conv2d(in_channels=self.c_1, out_channels=self.c_2, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv14b = nn.Conv2d(in_channels=self.c_2, out_channels=self.c_1, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv15a = nn.Conv2d(in_channels=self.c_1, out_channels=self.c_2, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv15b = nn.Conv2d(in_channels=self.c_2, out_channels=self.c_1, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv16a = nn.Conv2d(in_channels=self.c_1, out_channels=self.c_2, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv16b = nn.Conv2d(in_channels=self.c_2, out_channels=self.c_1, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv17a = nn.Conv2d(in_channels=self.c_1, out_channels=self.c_2, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv17b = nn.Conv2d(in_channels=self.c_2, out_channels=self.c_1, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv18a = nn.Conv2d(in_channels=self.c_1, out_channels=self.c_2, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv18b = nn.Conv2d(in_channels=self.c_2, out_channels=self.c_1, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv19a = nn.Conv2d(in_channels=self.c_1, out_channels=self.c_2, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        self.conv19b = nn.Conv2d(in_channels=self.c_2, out_channels=self.c_1, kernel_size=(self.k1, self.k2), stride=(self.s, self.s),padding=self.p)
        # REGRESSOR FOR W Values x 9 W vals
        self.headW0 = nn.Linear(self.hidden_size, 1)
        self.headW1 = nn.Linear(self.hidden_size, 1)
        self.headW2 = nn.Linear(self.hidden_size, 1)
        self.headW3 = nn.Linear(self.hidden_size, 1)
        self.headW4 = nn.Linear(self.hidden_size, 1)
        self.headW5 = nn.Linear(self.hidden_size, 1)
        self.headW6 = nn.Linear(self.hidden_size, 1)
        self.headW7 = nn.Linear(self.hidden_size, 1)
        self.headW8 = nn.Linear(self.hidden_size, 1)
        # REGRESSOR FOR L Values  x 9 W vals
        self.headL0 = nn.Linear(self.hidden_size, 1)
        self.headL1 = nn.Linear(self.hidden_size, 1)
        self.headL2 = nn.Linear(self.hidden_size, 1)
        self.headL3 = nn.Linear(self.hidden_size, 1)
        self.headL4 = nn.Linear(self.hidden_size, 1)
        self.headL5 = nn.Linear(self.hidden_size, 1)
        self.headL6 = nn.Linear(self.hidden_size, 1)
        self.headL7 = nn.Linear(self.hidden_size, 1)
        self.headL8 = nn.Linear(self.hidden_size, 1)
        # REGRESSOR FOR C Value
        self.headC0 = nn.Linear(self.hidden_size, 1)
        # REGRESSOR FOR R Value
        self.headR0 = nn.Linear(self.hidden_size, 1)
        ############### Multiregressor ################

        self.apply(self.__init__weights)

    def __init__weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1 / np.sqrt(self.input_size))
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

    def forward(self, x):
        x = x.reshape(31, 20, 1, 1)
        x = F.relu(self.convLatentA(x))
        x = F.relu(self.convLatentB(x))
        x = torch.flatten(x)
        x = self.linear1(x)
        x = self.linear2(F.dropout(x, 0.1))
        x = x.reshape(31, 40, 1, 1)
        x0 = torch.flatten(self.conv0b(self.conv0a(x)))
        W0 = F.hardtanh(self.headW0(x0), self.w_min, self.w_max)
        x1 = torch.flatten(self.conv1b(self.conv1a(x)))
        W1 = F.hardtanh(self.headW1(x1), self.w_min, self.w_hb)
        x2 = torch.flatten(self.conv2b(self.conv2a(x)))
        W2 = F.hardtanh(self.headW2(x2), self.w_min, self.w_max)
        x3 = torch.flatten(self.conv3b(self.conv3a(x)))
        W3 = F.hardtanh(self.headW3(x3), self.w_min, self.w_max)
        x4 = torch.flatten(self.conv4b(self.conv4a(x)))
        W4 = F.hardtanh(self.headW4(x4), self.w_min, self.w_max)
        x5 = torch.flatten(self.conv5b(self.conv5a(x)))
        W5 = F.hardtanh(self.headW5(x5), self.w_min, self.w_hb)
        x6 = torch.flatten(self.conv6b(self.conv6a(x)))
        W6 = F.hardtanh(self.headW6(x6), self.w_min, self.w_max)
        x7 = torch.flatten(self.conv7b(self.conv7a(x)))
        W7 = F.hardtanh(self.headW7(x7), self.w_min, self.w_max)
        x8 = torch.flatten(self.conv8b(self.conv8a(x)))
        W8 = F.hardtanh(self.headW8(x8), self.w_pass_min, self.w_pass_max)
        x9 = torch.flatten(self.conv9b(self.conv9a(x)))
        L0 = F.hardtanh(self.headL0(x9), self.l_min, self.l_max)
        x10 = torch.flatten(self.conv10b(self.conv10a(x)))
        L1 = F.hardtanh(self.headL1(x10), self.l_min, self.l_max)
        x11 = torch.flatten(self.conv11b(self.conv11a(x)))
        L2 = F.hardtanh(self.headL2(x11), self.l_min, self.l_max)
        x12 = torch.flatten(self.conv12b(self.conv12a(x)))
        L3 = F.hardtanh(self.headL3(x12), self.l_min, self.l_max)
        x13 = torch.flatten(self.conv13b(self.conv13a(x)))
        L4 = F.hardtanh(self.headL4(x13), self.l_min, self.l_max)
        x14 = torch.flatten(self.conv14b(self.conv14a(x)))
        L5 = F.hardtanh(self.headL5(x14), self.l_min, self.l_max)
        x15 = torch.flatten(self.conv15b(self.conv15a(x)))
        L6 = F.hardtanh(self.headL6(x15), self.l_min, self.l_max)
        x16 = torch.flatten(self.conv16b(self.conv16a(x)))
        L7 = F.hardtanh(self.headL7(x16), self.l_min, self.l_max)
        x17 = torch.flatten(self.conv17b(self.conv17a(x)))
        L8 = F.hardtanh(self.headL8(x17), self.l_min, self.l_max)
        x18 = torch.flatten(self.conv18b(self.conv18a(x)))
        C0 = F.hardtanh(self.headC0(x18), self.C_min, self.C_max)
        x19 = torch.flatten(self.conv19b(self.conv19a(x)))
        R0 = F.hardtanh(self.headR0(x19), self.R_min, self.R_max)

        return [W0, W1, W2, W3, W4, W5, W6, W7, W8, L0, L1, L2, L3, L4, L5, L6, L7, L8, C0, R0]

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        else:
            pass
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class Qtrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optim = optim.Adam(model.parameters(), lr=self.lr, )
        self.criterion = nn.MSELoss(reduce='mean')

    def train_step(self, state, action, reward, next_state, game_over):

        state = torch.tensor(np.array(state), dtype=torch.float).to(device)
        action = torch.tensor(np.array(action), dtype=torch.long).to(device)
        reward = torch.tensor(np.array(reward), dtype=torch.float).to(device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(device)
        # game_over = torch.tensor(np.array(game_over), dtype=torch.float).to(device)
        # print("state:", state, state.size())
        # print("action:", action, action.size())
        # print("reward:", reward)
        # print("next_state:", next_state.size())

        self.optim.zero_grad()
        torch.set_grad_enabled(True)
        # predicted Values
        for idx in range(0, int(np.array(state.cpu().detach().numpy()).shape[0])):
            Q_s = torch.abs(torch.tensor(self.model(state[idx]))).to(device)
            #Q_ns = Q_s.clone()
            # print("prediction\n : ", prediction)
            maximum_reward = torch.argmax(reward)
            Q_ns = torch.abs(torch.tensor(self.model(next_state[maximum_reward]))).to(device)
            # print("predss next state\n : ", preds)
            #if reward[idx] < 0.:
            #    Q_ns = torch.mul(preds, (1 + abs( reward[maximum_reward])))
            #else:
            #    Q_ns = torch.mul(preds, (1 + 1 / reward[maximum_reward]))
            # print("qns\n : ", Q_ns)
            # print('Memorization step:',idx)
            loss = self.criterion(Q_s.to(device),Q_ns.to(device))
            loss = Variable(loss, requires_grad=True)
            loss.backward()
            self.optim.step()
