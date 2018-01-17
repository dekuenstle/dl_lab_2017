#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# custom modules
from utils     import Options, rgb2gray
from simulator import Simulator
from transitionTable import TransitionTable
from model import DQLAgent, mlp_factory, cnn_factory

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
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                        opt.minibatch_siz, maxlen)

if opt.disp_on:
    win_all = None
    win_pob = None


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# You should prepare your network training here. I suggest to put this into a
# class by itself but in general what you want to do is roughly the following
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

steps = 1 * 10**6
state_with_history_dim = opt.hist_len * opt.state_siz
epi_step = 0
nepisodes = 0

q_fn = cnn_factory(state_with_history_dim, filters=[8, 16, 16, 32],
                   kernels_size=[64, 32, 16, 4], hidden_units=[],
		   output_units=opt.act_num)
q_fn().summary()
model = DQLAgent(q_fn, opt.act_num, model_dir=opt.checkpoint_dir,
                 learning_rate=opt.learning_rat,
                 discount=opt.q_loss_discount,
                 epsilon=opt.policy_eps, epsilon_min=opt.policy_eps_min,
                 epsilon_decay_interval=steps//opt.train_interval//2)
start_step = model.train_step * opt.train_interval
state = sim.newGame(opt.tgt_y, opt.tgt_x)
state_with_history = np.zeros((opt.hist_len, opt.state_siz))
append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
next_state_with_history = np.copy(state_with_history)
for step in range(start_step, steps):
    if state.terminal or epi_step >= opt.early_stop:
        epi_step = 0
        nepisodes += 1
        # reset the game
        state = sim.newGame(opt.tgt_y, opt.tgt_x)
        # and reset the history
        state_with_history[:] = 0
        append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
        next_state_with_history = np.copy(state_with_history)
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # here you let your agent take its action
    # and remember it
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # this just gets a random action
    action = model.action(state_with_history.reshape(-1))
    action_onehot = trans.one_hot_action(action)
    next_state = sim.step(action)
    # append to history
    append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))
    # add to the transition table
    trans.add(state_with_history.reshape(-1), action_onehot, next_state_with_history.reshape(-1), next_state.reward, next_state.terminal)
    # mark next state as current state
    state_with_history = np.copy(next_state_with_history)
    state = next_state
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # here you train your agent
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if opt.train and step % opt.train_interval == 0:
        model.train(*trans.sample_minibatch())

    # every once in a while you test your agent here so that you can track its performance
    if step % opt.eval_interval == 0:
        eval_results = model.evaluate(*trans.sample_minibatch())
        print("step {}, update {}, loss {:.4f}, eps {:.4f}"
              .format(step, eval_results['global_step'], eval_results['loss'], model.current_epsilon))

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


# 2. perform a final test of your model and save it
eval_results = model.evaluate(*trans.sample_minibatch())
print("loss {:.4f}".format(eval_results['loss']))
# Checkpoint of model ist stored automatically while training.
print("Latest model checkpoint stored in {}.".format(model.latest_checkpoint()))
