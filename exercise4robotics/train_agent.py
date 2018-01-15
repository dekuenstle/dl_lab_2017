#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import tensorflow as tf
import keras.layers as klayers
import matplotlib.pyplot as plt
from random import randrange, random
import tensorflow as tf

# custom modules
from utils     import Options, rgb2gray
from simulator import Simulator
from transitionTable import TransitionTable


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# this is a little helper function that calculates the Q error for you
# so that you can easily use it in tensorflow as the loss
# you can copy this into your agent class or use it from here
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def Q_loss(Q_s, action_onehot, Q_s_next, best_action_next, reward, terminal, discount=0.99):
    """
    All inputs should be tensorflow variables!
    We use the following notation:
       N : minibatch size
       A : number of actions
    Required inputs:
       Q_s: a NxA matrix containing the Q values for each action in the sampled states.
            This should be the output of your neural network.
            We assume that the network implments a function from the state and outputs the 
            Q value for each action, each output thus is Q(s,a) for one action 
            (this is easier to implement than adding the action as an additional input to your network)
       action_onehot: a NxA matrix with the one_hot encoded action that was selected in the state
                      (e.g. each row contains only one 1)
       Q_s_next: a NxA matrix containing the Q values for the next states.
       best_action_next: a NxA matrix with the best current action for the next state
       reward: a Nx1 matrix containing the reward for the transition
       terminal: a Nx1 matrix indicating whether the next state was a terminal state
       discount: the discount factor
    """
    # calculate: reward + discount * Q(s', a*),
    # where a* = arg max_a Q(s', a) is the best action for s' (the next state)
    with tf.name_scope('QLoss'):
        with tf.name_scope('TargetValue'):
            target_q = (1. - terminal) * discount * tf.reduce_sum(best_action_next * Q_s_next, 1, keep_dims=True) + reward
            # NOTE: we insert a stop_gradient() operation since we don't want to change Q_s_next, we only
            #       use it as the target for Q_s
            target_q = tf.stop_gradient(target_q)
        # calculate: Q(s, a) where a is simply the action taken to get from s to s'
        with tf.name_scope('SelectedValue'):
            selected_q = tf.reduce_sum(action_onehot * Q_s, 1, keep_dims=True)
        with tf.name_scope('SumSquareDiff'):
            loss = tf.reduce_sum(tf.square(selected_q - target_q))
    return loss

# TODO: Try to avoid separate prediction step at each iteration by merging prediction inside graph
#       using layer variable sharing.
# def dense_q_model(state, state_next, units=[100], num_actions=opt.act_num):
#     """ Build model representing Q-function as a multilayer perceptron.

#     Args:
#       state: Tensorflow placeholder for current state.
#       state_next: Tensorflow placeholder for next state.
#       units: List of number of hidden units.
#       num_actions: Number of actions

#     Returns:
#       (Q_s, Q_s_next): Q values of all actions for current and next state.
#     """
#     model = Sequential()
#     for u in units:
#         model.add(tf.keras.layers.Dense(u, activation='relu', input_dim=state.shape[1]))
#     model.add(tf.keras.layers.Dense(num_actions, activation='linear'))
#     return model(state), model(state_next)


def dql_model_fn(features, mode, params, config):
    """ Build the DQL model up to where it may used for inference.

    Args:
      state: State placeholder.
      action: Action placeholder.
      hidden1_units: Size of the first hidden layer.

    Returns:
      ???
    """
    state = features['state']

    def q_fn(state):
        with tf.variable_scope('DenseQ', reuse=tf.AUTO_REUSE):
            # TODO: use layer objects instead of reuse flags.
            hidden1 = tf.layers.dense(inputs=state,
                                      units=10,
                                      activation=tf.nn.relu,
            name='hidden1')
            output = tf.layers.dense(inputs=hidden1, units=opt.act_num, name='output')
            return output
    with tf.name_scope('ThisQValue'):
        Q_s = q_fn(state)

    with tf.name_scope('Prediction'):
        predictions = {
            "action": tf.argmax(Q_s, axis=1),
            "action_onehot": tf.one_hot(tf.argmax(Q_s, axis=1), int(opt.act_num)),
            "Q_s": Q_s
        }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    reward, terminal = (features['reward'], features['terminal'])
    action, state_next = (features['action_onehot'], features['state_next'])
    with tf.name_scope('NextQValue'):
        Q_s_next = q_fn(state_next)
    best_action_next = tf.one_hot(tf.argmax(Q_s_next, axis=1), int(opt.act_num))

    loss = Q_loss(Q_s, action,
                  Q_s_next, best_action_next,
                  reward, terminal, opt.q_loss_discount)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=opt.learning_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    raise NotImplementedError(mode)

def eps_greedy_policy(action):
    # TODO: Add eps decay
    if random() < opt.policy_eps or action is None:
        print('random')
        return randrange(opt.act_num)
    else:
        print('chosen')
        return action



def append_to_hist(state, obs):
    """
    Add observation to the state.
    """
    for i in range(state.shape[0]-1):
        state[i, :] = state[i+1, :]
    state[-1, :] = obs

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# In contrast to your last exercise you DO NOT generate data before training
# instead the TransitionTable is build up while you are training to make sure
# that you get some data that corresponds roughly to the current policy
# of your agent
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
# setup a large transitiontable that is filled during training
maxlen = 100000
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                        opt.minibatch_size, maxlen)

if opt.disp_on:
    win_all = None
    win_pob = None


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# You should prepare your network training here. I suggest to put this into a
# class by itself but in general what you want to do is roughly the following
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""
# setup placeholders for states (x) actions (u) and rewards and terminal values
x = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.hist_len*opt.state_siz))
u = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.act_num))
ustar = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.act_num))
xn = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.hist_len*opt.state_siz))
r = tf.placeholder(tf.float32, shape=(opt.minibatch_size, 1))
term = tf.placeholder(tf.float32, shape=(opt.minibatch_size, 1))

# get the output from your network
Q = my_network_forward_pass(x)
Qn =  my_network_forward_pass(xn)

# calculate the loss
loss = Q_loss(Q, u, Qn, ustar, r, term)

# setup an optimizer in tensorflow to minimize the loss
"""
model = tf.estimator.Estimator(model_fn=dql_model_fn, model_dir=opt.checkpoint_dir, params={})
# lets assume we will train for a total of 1 million steps
# this is just an example and you might want to change it
steps = 1 * 10**6
epi_step = 0
nepisodes = 0

state = sim.newGame(opt.tgt_y, opt.tgt_x)
state_with_history = np.zeros((opt.hist_len, opt.state_siz))
append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
next_state_with_history = np.copy(state_with_history)
for step in range(steps):
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
    if step == 0 and nepisodes == 0:
        action = eps_greedy_policy(None)
    else:
        state_feat = np.expand_dims(state_with_history.reshape(-1), axis=0).astype(np.float32)

        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'state': state_feat},
            num_epochs=1,
            shuffle=False)
        pred_action = list(model.predict(input_fn=predict_input_fn,
                                         predict_keys=['action']))[0]['action']
        action = eps_greedy_policy(pred_action)

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
    if step % opt.train_interval == 0:
        state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = trans.sample_minibatch()
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'state': state_batch, 'state_next': next_state_batch,
               'action_onehot': action_batch,
               'reward': reward_batch, 'terminal': terminal_batch},
            num_epochs=1,
            batch_size=opt.minibatch_size,
        shuffle=True)
        model.train(train_input_fn)

    # every once in a while you test your agent here so that you can track its performance
    if step % opt.eval_interval == 0:
        state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = trans.sample_minibatch()
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'state': state_batch, 'state_next': next_state_batch,
               'action_onehot': action_batch,
               'reward': reward_batch, 'terminal': terminal_batch},
            num_epochs=1,
        shuffle=False)
        eval_results = model.evaluate(input_fn=eval_input_fn)
        print(eval_results)


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
# Checkpoint of model ist stored automatically while training.
