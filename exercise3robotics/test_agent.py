import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from random import randrange
from keras.models import load_model

# custom modules
from utils     import Options, rgb2gray
from simulator import Simulator

# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# this script assumes you did generate your model with the train_agent.py script
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
agent = load_model(opt.model_fil)

# 1. control loop
if opt.disp_on:
    win_all = None
    win_pob = None
epi_step = 0    # #steps in current episode
nepisodes = 0   # total #episodes executed
nepisodes_solved = 0
action = 0     # action to take given by the network

# start a new game
state = sim.newGame(opt.tgt_y, opt.tgt_x)
for step in range(opt.eval_steps):
    # episode just started
    if epi_step == 0:
        full_state = np.zeros([opt.hist_len, opt.state_siz])
        full_state[:,:] = rgb2gray(state.pob).reshape((-1,))

    # check if episode ended
    if state.terminal or epi_step >= opt.early_stop:
        epi_step = 0
        nepisodes += 1
        if state.terminal:
            nepisodes_solved += 1
        # start a new game
        state = sim.newGame(opt.tgt_y, opt.tgt_x)
    else:
        full_state = np.append(full_state, rgb2gray(state.pob).reshape((1, -1)), 0)
        full_state = np.delete(full_state, 0, 0)
        x = full_state.reshape(1, opt.hist_len * opt.state_siz)
        y = agent.predict(x)
        action = y.argmax()
        state = sim.step(action)

        epi_step += 1

    if state.terminal or epi_step >= opt.early_stop:
        epi_step = 0
        nepisodes += 1
        if state.terminal:
            nepisodes_solved += 1
        # start a new game
        state = sim.newGame(opt.tgt_y, opt.tgt_x)

    if step % opt.prog_freq == 0:
        print(step)

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

# 2. calculate statistics
print(float(nepisodes_solved) / float(nepisodes))

# 3. TODO perhaps  do some additional analysis
