import time
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from env import LDO_SIM
from agent import Agent


def train():
    # INIT
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    reward = 0.
    done = 0
    ldo_sim = LDO_SIM()
    ldo_sim.generate_output_dim()
    ldo_sim.play_step(0)
    agent = Agent(ldo_sim)
    agent.define_goals(0.5, 350, 0.001, 0.1)
    # INIT
    while True:
        # get old state
        state_old = agent.get_state()

        # get new move
        final_move = agent.get_action(state_old)
        # take a move

        # MbVal, DevbVal, done = agent.onet2b(final_move)
        # MaVal, DevaVal = agent.b2val(MbVal, DevbVal)

        # var_transmutation = agent.ldo_sim.ch_value * MaVal
        # print("TRANSMUTATION VAL",var_transmutation)
        reward = agent.ldo_sim.set_current_var(final_move,reward)
        agent.ldo_sim.play_step(0)
        # # make a judgment about those moves
        reward = agent.rate_this_state(agent, reward)
        # take new state
        state_new = agent.get_state()
        # # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        # # # remember your actions and states
        agent.remember_state(state_old, final_move, reward, state_new, done)
        reward -= 1

        if reward == 100:
            done = 1
        if done:
            # train long memory, plot result
            agent.n_games += 1
            agent.train_long_memory()

            if reward > record:
                record = reward
                agent.model.save()

            print('Game', agent.n_games, 'Score', reward, 'Record:', record)

            plot_scores.append(reward)
            total_score += reward
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            reward = 0.
            done = 0
            print(agent.ldo_sim.state_var_str)
            ldo_sim = LDO_SIM()
            ldo_sim.generate_output_dim()
            ldo_sim.play_step(0)


if __name__ == '__main__':
    train()

    # print('Woow - Thats awesome !!')
