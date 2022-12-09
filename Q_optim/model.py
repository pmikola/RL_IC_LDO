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
        self.hidden_size = self.input_size * 8
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
        # LATENT NON-LINEAR SHARED SPACE MAPPING
        # TODO : More degrees of freedom
        # TODO : LATENT SPACE FOR EACH OF THE HEADS??!
        self.linear1 = nn.Linear(self.input_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        # LATENT NON-LINEAR SHARED SPACE MAPPING

        # TODO : RL Multiregressor vs. MultiLabelClassifier vs. MultiheadBinaryClassifier vs. Mixed

        ############### Multiregressor ################
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
        x = self.linear1(x)
        x = F.relu(self.linear2(F.dropout(x, 0.1)))
        W0 = F.hardtanh(self.headW0(x), self.w_min, self.w_max)
        W1 = F.hardtanh(self.headW1(x), self.w_min, self.w_hb)
        W2 = F.hardtanh(self.headW2(x), self.w_min, self.w_max)
        W3 = F.hardtanh(self.headW3(x), self.w_min, self.w_max)
        W4 = F.hardtanh(self.headW4(x), self.w_min, self.w_max)
        W5 = F.hardtanh(self.headW5(x), self.w_min, self.w_hb)
        W6 = F.hardtanh(self.headW6(x), self.w_min, self.w_max)
        W7 = F.hardtanh(self.headW7(x), self.w_min, self.w_max)
        W8 = F.hardtanh(self.headW8(x), self.w_pass_min, self.w_pass_max)
        L0 = F.hardtanh(self.headL0(x), self.l_min, self.l_max)
        L1 = F.hardtanh(self.headL1(x), self.l_min, self.l_max)
        L2 = F.hardtanh(self.headL2(x), self.l_min, self.l_max)
        L3 = F.hardtanh(self.headL3(x), self.l_min, self.l_max)
        L4 = F.hardtanh(self.headL4(x), self.l_min, self.l_max)
        L5 = F.hardtanh(self.headL5(x), self.l_min, self.l_max)
        L6 = F.hardtanh(self.headL6(x), self.l_min, self.l_max)
        L7 = F.hardtanh(self.headL7(x), self.l_min, self.l_max)
        L8 = F.hardtanh(self.headL8(x), self.l_min, self.l_max)
        C0 = F.hardtanh(self.headC0(x), self.C_min, self.C_max)
        R0 = F.hardtanh(self.headR0(x), self.R_min, self.R_max)
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
        if len(state.shape) == 1:
            # state has one dim
            # (1,x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over,)  # tuple with only one value

        # predicted Values
        for idx in range(0, int(np.array(state.cpu().detach().numpy()).shape[0])):
            Q_s = torch.abs(torch.tensor(self.model(state[idx]))).to(device)
            Q_ns = Q_s.clone()
            # print("prediction\n : ", prediction)
            preds = torch.abs(torch.tensor(self.model(next_state[idx]))).to(device)
            # print("predss next state\n : ", preds)
            if reward[idx] < 0.:
                Q_ns = torch.mul(preds, (abs(1 + reward[idx])))
            else:
                Q_ns = torch.mul(preds, (1 + 1 / reward[idx]))
            # print("qns\n : ", Q_ns)
            # print('Memorization step:',idx)
            loss = self.criterion(Q_s.to(device), Q_ns.to(device))
            loss = Variable(loss, requires_grad=True)
            loss.backward()
            self.optim.step()
