import random
import time
from collections import deque  # datastructure to store memory vals

import numpy as np
import torch

from Q_optim.model import Qnet, Qtrainer

MAX_MEMORY = 100_000
MAX_SHORT_MEMORY = 32
MAX_BEST_MEMORY = 100_000
BATCH_SIZE = 512
BATCH_SIZE_SHORT = 8
BATCH_SIZE_BEST = 128
top_k_div = 20
# matplotlib.use('Qt5Agg')
use_cuda = True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")


class Agent:
    def __init__(self, ldo_sim):
        self.ldo_sim = ldo_sim
        self.model_input = ldo_sim.sim_state
        self.I_min = None
        self.I_max = None
        self.V_output_min = None
        self.V_output_max = None
        self.V_output = None
        self.n_games = 1
        self.epsilon = 0.5  # randomness
        self.gamma = 0.85  # discount rate
        self.alpha = 0.75  # learning rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.short_memory = deque(maxlen=MAX_SHORT_MEMORY)
        self.best_memory = deque(maxlen=MAX_BEST_MEMORY)
        self.model = Qnet(len(self.model_input), 13).float().to(device)
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("No. of Parametres : ", pytorch_total_params)
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("No. of Trainable parametres : ", pytorch_total_params)
        time.sleep(0.5)
        self.trainer = Qtrainer(self.model, lr=0.0000625, alpha=self.alpha, gamma=self.gamma)  # , alpha=self.alpha)
        self.game = None
        self.agent = None
        self.scores = []
        self.mean_scores = []
        self.plot_loss = []
        self.total_score = 0.
        self.record = -1000.
        self.reward = 0.
        self.agent = []
        self.plot_n_games = []
        self.done = False
        self.rewards = []

    def define_goals(self, Vout, I_max, I_min, error):
        self.V_output = Vout
        self.V_output_max = self.V_output + self.V_output * error/2
        self.V_output_min = self.V_output - self.V_output * error/2
        self.I_max = I_max
        self.I_min = I_min

    def rate_this_state(self, agent, reward):
        Voltages = agent.ldo_sim.V_source_list
        for i in range(0, len(Voltages)):
            V_d1 = np.mean(Voltages[-i])
            # # V_d2 = np.mean(Voltages[-i - 1])
            # max_v = np.max(Voltages[-i])
            # min_v = np.min(Voltages[-i])
            # k = 1 + (1 / abs(V_d1 - self.V_output))  # - V_d2)
            # l = 1 / abs(max_v - min_v)
            # rscaler = 2
            # # reward += (k + l) * rscaler
            # reward += l * rscaler
            # if self.V_output_max > max_v > self.V_output_min and self.V_output_min < min_v < self.V_output_max:
            #     reward += 100
            # if self.V_output_min < V_d1 < self.V_output_max:
            #     reward += 10
            # # if self.V_output_min < V_d2 < self.V_output_max:
            # #    reward += 25
            if V_d1 > 0.:
                reward += 15
            for j in range(0, len(Voltages[-i])):
                if self.V_output_max > Voltages[-i][j] > self.V_output_min:
                    reward += 1
                else:
                    reward -= 0
        return reward

    def get_state(self):
        state = self.ldo_sim.sim_state
        return state

    def remember_state(self, state, action, reward, next_state, dones):
        self.memory.append((state, action, reward, next_state, dones))
        self.short_memory.append((state, action, reward, next_state, dones))
        self.best_memory.append((state, action, reward, next_state, dones))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states)

    def train_short_memory(self):
        if len(self.short_memory) > BATCH_SIZE_SHORT:
            mini_sample = random.sample(self.short_memory, BATCH_SIZE_SHORT)  # list of tuples
        else:
            mini_sample = self.short_memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states)

    def train_best_memories(self):
        if len(self.best_memory) > BATCH_SIZE_BEST:
            mini_sample = random.sample(self.best_memory, BATCH_SIZE_BEST)  # list of tuples
        else:
            mini_sample = self.best_memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        if torch.tensor(rewards).size(0) < top_k_div:
            top_k = top_k_div
        else:
            top_k = torch.tensor(rewards).size(0)
        r_max, r_argmax = torch.topk(torch.tensor(rewards), int(top_k / top_k_div))
        self.trainer.train_step(torch.tensor(states)[r_argmax], torch.tensor(actions)[r_argmax],
                                torch.tensor(rewards)[r_argmax], torch.tensor(next_states)[r_argmax])

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        # print(len(state)) # 3787

        self.epsilon = 80 - self.n_games
        preds = []
        predictions_r = []
        self.model.train()
        state0 = torch.tensor(state, dtype=torch.float).to(device)
        predictions = self.model(state0)
        if random.randint(0, 200) < self.epsilon:
            for i in range(len(self.model.const)):
                n = random.randint(0, self.model.no_bits)
                m = self.model.no_bits - n
                p = np.ones(n + m)
                p[:m] = 0
                np.random.shuffle(p)
                predictions_r.append(p)
                if np.all(p == 0.):
                    preds.append(self.model.const[i])
                else:
                    aVal = self.b2val(p)
                    aVal = 2 * self.model.no_bits / aVal
                    physDim = aVal * self.model.const[i]
                    if physDim < self.ldo_sim.ch_value:
                        physDim = self.ldo_sim.ch_value
                    else:
                        pass
                    preds.append(physDim)

            final_move = np.array(preds)
            print("Random Choice")
            return final_move.astype(float), np.array(predictions_r).astype(float)
        else:
            for i in range(len(self.model.const)):
                p = predictions[i].cpu().detach().numpy()
                # self.twopass(p, 0.5, 0., 1, 0) # 4bce2mse

                if np.all(p == 0.):
                    preds.append(self.model.const[i])
                else:
                    aVal = self.b2val(p)
                    aVal = 2 * self.model.no_bits / aVal
                    physDim = aVal * self.model.const[i]
                    if physDim < self.ldo_sim.ch_value:
                        physDim = self.ldo_sim.ch_value
                    else:
                        pass
                    preds.append(physDim)
                predictions_r.append(p)
            final_move = np.array(preds)
            print("Model Choice")

            return final_move.astype(float), np.array(predictions_r).astype(float)
        # def get_reward(self):

    @staticmethod
    def b2val(MbVal):
        MaVal = 0.
        for i in range(0, MbVal.shape[0]):
            MaVal += (2 ** i) * MbVal[i]
        # print(MaVal)
        return float(MaVal)

    @staticmethod
    def twopass(data, upper_threshold, lower_threshold, default_value1, default_value2):
        data[data > upper_threshold] = default_value1
        data[data < lower_threshold] = default_value2
