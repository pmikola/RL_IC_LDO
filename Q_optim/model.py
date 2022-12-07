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
        self.hidden_size = self.input_size * 4

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
        ############### Multiregressor ################

        self.apply(self.__init__weights)

    def __init__weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1 / np.sqrt(self.input_size))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(F.dropout(x, 0.2)))
        W0 = self.headW0(x)
        W1 = self.headW1(x)
        W2 = self.headW2(x)
        W3 = self.headW3(x)
        W4 = self.headW4(x)
        W5 = self.headW5(x)
        W6 = self.headW6(x)
        W7 = self.headW7(x)
        W8 = self.headW8(x)
        L0 = self.headL0(x)
        L1 = self.headL1(x)
        L2 = self.headL2(x)
        L3 = self.headL3(x)
        L4 = self.headL4(x)
        L5 = self.headL5(x)
        L6 = self.headL6(x)
        L7 = self.headL7(x)
        L8 = self.headL8(x)
        C0 = self.headC0(x)
        return torch.tensor([W0, W1, W2, W3, W4, W5, W6, W7, W8, L0, L1, L2, L3, L4, L5, L6, L7, L8, C0])

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
        self.criterion = nn.MSELoss()

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
        Q_s = torch.abs(self.model(state))
        Q_ns = Q_s.clone()
        for idx in range(len(game_over)):
            # print("prediction\n : ", prediction)
            preds = torch.abs(self.model(next_state[idx])).to(device)
            #print("predss next state\n : ", preds)
            Q_ns = torch.mul(preds, (1 + 1 / reward[idx]))
            #print("qns\n : ", Q_ns)
        loss = self.criterion(Q_ns.to(device),Q_s.to(device))
        loss = Variable(loss, requires_grad=True)
        loss.backward()
        self.optim.step()
