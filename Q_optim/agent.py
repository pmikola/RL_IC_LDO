import sys
import threading
import time
from IPython import display
import matplotlib
import torch
import random
import numpy as np
from collections import deque  # datastructure to store memory vals

from Q_optim.model import Qnet, Qtrainer
from env import LDO_SIM

########################### BELLMAN EQUATiON ##############################
# NewQ(state,action) = Q(state,action) + lr[R(state,action) + gamma*maxQ'(state',action') - Q(state,action(]
# NewQ(state,action) - New Q vale for that state and action
# Q(state,action) - current Q value in state and action
# lr - learning rate
# R(state,action) - reward for thaking that action and that state
# gamma- discount rate
# maxQ'(state',action') - Maximum expected future reward for given new state and all possible actions at that new state
########################### BELLMAN EQUATiON ##############################


MAX_MEMORY = 100_000
BATCH_SIZE = 128
LR = 0.001
white = True
black = False
# matplotlib.use('Qt5Agg')
use_cuda = True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")


class Agent:
    def __init__(self, ldo_sim):
        self.ldo_sim = ldo_sim
        self.model_input = ldo_sim.sim_state
        self.model_output_len = ldo_sim.output_shape.shape[0]
        self.I_min = None
        self.I_max = None
        self.V_output_min = None
        self.V_output_max = None
        self.V_output = None
        self.n_games = 0
        self.epsilon = 0.5  # randomness
        self.gamma = 0.95  # discount rate
        self.alpha = 0.2  # learning rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Qnet(len(self.model_input)).float().to(device)
        self.trainer = Qtrainer(self.model, lr=0.001, gamma=self.gamma)  # , alpha=self.alpha)
        self.game = None
        self.agent = None
        self.plot_scores = []
        self.plot_mean_scores = []
        self.plot_loss = []
        self.total_score = 0.
        self.record = -1000.
        self.reward = 0.
        self.agent = []
        self.plot_n_games = []
        self.done = False

    def define_goals(self, Vout, I_max, I_min, error):
        self.V_output = Vout
        self.V_output_max = self.V_output + self.V_output * error
        self.V_output_min = self.V_output - self.V_output * error
        self.I_max = I_max
        self.I_min = I_min

    def rate_this_state(self, agent, reward):
        Voltages = agent.ldo_sim.V_source_list
        for i in range(0, len(Voltages)):
            V_d1 = np.mean(Voltages[-i])
            V_d2 = np.mean(Voltages[-i - 1])
            max_v = np.max(Voltages[-i])
            min_v = np.min(Voltages[-i])
            max_diff = abs(self.V_output_max - max_v)
            min_diff = abs(self.V_output_min - min_v)
            k = abs(V_d1 - V_d2)
            rscaler = 10
            reward += 1/(max_diff + min_diff)
            reward += 1/(k * rscaler)
            if self.V_output_max > max_v > self.V_output_min and self.V_output_min < min_v < self.V_output_max:
                reward += 20
            if self.V_output_min < V_d1 < self.V_output_max:
                reward += 10
            if self.V_output_min < V_d2 < self.V_output_max:
                reward += 10
        return reward

    def get_state(self):
        state = self.ldo_sim.sim_state
        return state

    def remember_state(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        # print(len(state)) # 3787
        self.epsilon = 80 - self.n_games
        final_move = self.ldo_sim.output_shape
        if random.randint(0, 200) < self.epsilon:
            for i in range(0,len(self.ldo_sim.output_shape)):
                final_move[i] = np.random.uniform(self.ldo_sim.dim_range_min,self.ldo_sim.dim_range_max)
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(device)
            predictions = self.model(state0)
            final_move = predictions.cpu().detach().numpy()
        return final_move.astype(float)
        # def get_reward(self):

    def onet2b(self, final_move):
        multiplier_val_range = range(0, self.ldo_sim.nbits_multiplier)
        choosen_dev_range = range(self.ldo_sim.nbits_multiplier, self.ldo_sim.nbits_multiplier + self.ldo_sim.nbits_dev)
        MbVal = final_move[multiplier_val_range]
        DevbVal = final_move[choosen_dev_range]
        stop_bit = final_move[-1]
        return MbVal, DevbVal, stop_bit

    @staticmethod
    def b2val(MbVal, DevbVal):
        MaVal = 0.
        for i in range(0, len(MbVal)):
            MaVal += (2 ** i) * MbVal[i]
        #print(MaVal)
        DevaVal = 0.
        for i in range(0, len(DevbVal)):
            DevaVal += (2 ** i) * DevbVal[i]
        #print(DevaVal)
        return int(MaVal), int(DevaVal)