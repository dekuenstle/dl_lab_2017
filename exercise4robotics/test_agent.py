#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# custom modules
from utils     import Options, rgb2gray
from simulator import Simulator
from agent_def import agent

def append_to_hist(state, obs):
    """
    Add observation to the state.
    """
    for i in range(state.shape[0]-1):
        state[i, :] = state[i+1, :]
    state[-1, :] = obs

# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
# setup a large transitiontable that is filled during training
maxlen = 100000

if opt.disp_on:
    win_all = None
    win_pob = None

state_with_history_dim = opt.hist_len * opt.state_siz
epi_step = 0
nepisodes = 1
target_found_count = 0

state = sim.newGame(opt.tgt_y, opt.tgt_x)
state_with_history = np.zeros((opt.hist_len, opt.state_siz))
append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
next_state_with_history = np.copy(state_with_history)
while nepisodes <= opt.eval_nepisodes:
    if state.terminal:
        print("Agent reached the target!")
        target_found_count += 1
    elif epi_step >= opt.early_stop:
        print("Early stop, too many steps!")
    if state.terminal or epi_step >= opt.early_stop:
        epi_step = 0
        nepisodes += 1
        # reset the game
        state = sim.newGame(opt.tgt_y, opt.tgt_x)
        # and reset the history
        state_with_history[:] = 0
        append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
        next_state_with_history = np.copy(state_with_history)
        print("New episode: {}".format(nepisodes))
    epi_step += 1
    print("  Step {}/{} (episode {}/{})".format(epi_step, opt.early_stop, nepisodes, opt.eval_nepisodes))
    action = agent.action(state_with_history.reshape(-1))
    next_state = sim.step(action)
    # append to history
    append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))
    # add to the transition table
    state_with_history = np.copy(next_state_with_history)
    state = next_state

    if opt.disp_on:
        if win_all is None:
            plt.subplot(121)
            win_all = plt.imshow(state.screen)
            plt.subplot(122)
            win_pob = plt.imshow(state.pob)
        else:
            win_all.set_data(state.screen)
            win_pob.set_data(state.pob)
        plt.pause(opt.disp_interval)
        plt.draw()

print("----------------------")
print("Agent found target in {} of {} episodes.".format(target_found_count, opt.eval_nepisodes))
