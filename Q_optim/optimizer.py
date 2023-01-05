import time
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from env import LDO_SIM
from agent import Agent

use_cuda = True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
torch.manual_seed(2022)
np.random.seed(2022)


def train():
    # INIT
    total_score = 0.
    record = -1000.
    reward = 0.
    done = 0
    ldo_sim = LDO_SIM()
    ldo_sim.generate_output_dim()
    ldo_sim.play_step(0, [0.], [0.], [0.])
    agent = Agent(ldo_sim)
    agent.define_goals(0.5, 350, 0.001, 0.1)
    # INIT
    i_counter = 0
    while True:
        # get old state
        state_old = agent.get_state()
        # get new move
        final_move, action = agent.get_action(state_old)
        # take a move
        # MbVal, DevbVal, done = agent.onet2b(final_move)
        # MaVal, DevaVal = agent.b2val(MbVal, DevbVal)
        # var_transmutation = agent.ldo_sim.ch_value * MaVal
        # print("TRANSMUTATION VAL",var_transmutation)
        reward = agent.ldo_sim.set_current_var(final_move, reward)
        agent.ldo_sim.play_step(1, agent.scores, agent.mean_scores, agent.trainer.loss_list)
        # # make a judgment about those moves
        reward = agent.rate_this_state(agent, reward)
        # take new state
        state_new = agent.get_state()
        # remember
        agent.remember_state(state_old, action, reward, state_new, done)
        # train short memory
        # agent.train_short_memory() #
        agent.train_best_memories()  # TODO : put in main loop or after finished cycle?
        i_counter += 1
        if reward > record or i_counter > 20:
            done = 1
            agent.model.done = 1
            i_counter = 0

        print("REWARD : ", reward)
        # reward -= 1
        total_score += reward
        agent.scores.append(reward)
        try:
            mean_score = sum(agent.scores) / len(agent.scores)
        except ZeroDivisionError:
            mean_score = 0.
        agent.mean_scores.append(mean_score)
        reward = 0
        if done:
            # train long memory, plot result
            agent.n_games += 1
            agent.train_long_memory()

            if agent.scores[-1] > record:
                record = agent.scores[-1]
                agent.model.save()
            print('Sim Cycle : ', agent.n_games, 'Score : ', agent.scores[-1], 'Record : ', record)
            ldo_sim.play_step(1, agent.scores, agent.mean_scores, agent.trainer.loss_list)
            done = 1
            # print(agent.ldo_sim.state_var_str)
            ldo_sim = LDO_SIM()
            ldo_sim.generate_output_dim()
            # ldo_sim.play_step(1)


if __name__ == '__main__':
    train()

    # print('Woow - Thats awesome !!')
