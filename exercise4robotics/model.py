from random import randrange, random

import numpy as np
import tensorflow as tf
klayers = tf.keras.layers


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

def mlp_factory(input_dim, hidden_units, output_units,
                hidden_activation='relu', output_activation='linear'):
    """ Build model representing Q(s) function as a multilayer perceptron.

    Args:
      input_dim: Number of input units.
      hidden_units: List of hidden layer unit numbers.
      output_units: Number of units in output layer.
      hidden_activation: Activation function of hidden layers. Default: 'relu'.
      output_activation: Activation function of output layer. Default: 'linear'.

    Returns:
      factory_fn: Function building model with given params.
    """
    def _build_model():
        model = tf.keras.models.Sequential()
        input_layer = klayers.Dense(hidden_units[0], activation='relu', input_dim=input_dim)
        model.add(input_layer)
        for u in hidden_units[1:]:
            hidden_layer = klayers.Dense(u, activation='relu')
            model.add(hidden_layer)
        output_layer = klayers.Dense(output_units, activation='linear')
        model.add(output_layer)
        return model
    return _build_model

def cnn_factory(input_dim, filters, kernels_size, output_units,
                output_activation='linear'):
    """ Build model representing Q(s) function with multiple convolutional layers
        and a dense layer at the end.

    Args:
      input_dim: Number of input units.
      filters: List of filter number in convolution layers.
      kernel_size: List of kernel size in convolution layers.
      output_units: Number of units in output layer.
      output_activation: Activation function of output layer. Default: 'linear'.

    Returns:
      factory_fn: Function building model with given params.
    """
    def _build_model():
        model = tf.keras.models.Sequential()
        model.add(klayers.Reshape((input_dim, 1),
                    input_shape=(input_dim,)))
        for f, k in zip(filters, kernels_size):
            conv_layer = klayers.Conv1D(f, kernel_size=k, strides=1, activation='relu')
            model.add(conv_layer)
            max_layer = klayers.MaxPooling1D(pool_size=2)
            model.add(max_layer)
        model.add(klayers.Flatten())
        model.add(klayers.Dense(output_units, activation=output_activation))
        return model
    return _build_model

class DQLAgent:
    """ Agent learning using Q learning algorithms with neural network for Q function """

    def __init__(self, q_model_fn, num_actions, model_dir=None,
                 minibatch_size=32, learning_rate=0.001, discount=0.99,
                 epsilon=1, epsilon_decay_interval=100000, epsilon_min=0.1):
        self.q_model_fn = q_model_fn
        self.estimator = tf.estimator.Estimator(model_fn=self._model_fn,
                                                model_dir=model_dir, params={})
        self.minibatch_size = minibatch_size
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay_interval = epsilon_decay_interval

    def train(self, state_batch, action_batch, next_state_batch, reward_batch, terminal_batch):
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'state': state_batch, 'state_next': next_state_batch,
               'action_onehot': action_batch,
               'reward': reward_batch, 'terminal': terminal_batch},
            batch_size=self.minibatch_size,
            num_epochs=1, shuffle=True)
        return self.estimator.train(input_fn=train_input_fn)

    def predict(self, state_batch, predict_keys=None):
        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'state': state_batch},
            num_epochs=1, shuffle=False)
        return self.estimator.predict(input_fn=pred_input_fn,
                                      predict_keys=predict_keys)

    def predict_action(self, state):
        state_batch = np.expand_dims(state, axis=0).astype(np.float32)
        Q_s_batch = self.predict(state_batch)
        Q_s = list(Q_s_batch)[0]
        action = np.argmax(Q_s)
        return action

    def action(self, state):
        if self._get_global_step() < 1:
            # Workaround: before first training, prediction fails
            return self._epsilon_greedy(None)
        else:
            pred_action = self.predict_action(state)
            return self._epsilon_greedy(pred_action)

    def evaluate(self, state_batch, action_batch, next_state_batch, reward_batch, terminal_batch):
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'state': state_batch, 'state_next': next_state_batch,
               'action_onehot': action_batch,
               'reward': reward_batch, 'terminal': terminal_batch},
            batch_size=self.minibatch_size,
            num_epochs=1, shuffle=False)
        return self.estimator.evaluate(input_fn=eval_input_fn)

    @property
    def current_epsilon(self):
        return max(self.epsilon_min,
                   self.epsilon - (self.epsilon - self.epsilon_min)
                   * self._get_global_step() / self.epsilon_decay_interval)

    def _get_global_step(self):
        try:
            return self.estimator.get_variable_value('global_step:0')
        except ValueError:
            # Workaround: before first training, prediction fails
            return 0

    def _epsilon_greedy(self, action):
        if random() < self.current_epsilon:
            return randrange(self.num_actions)
        else:
            return action

    def _model_fn(self, features, mode, params, config):
        """ First class function representing tensorflow model
        for use with tf.estimator.Estimator
        """
        state = features['state']

        q_model = self.q_model_fn()
        with tf.name_scope('ThisQValue'):
            Q_s = q_model(state)
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=Q_s)

        reward, terminal = (features['reward'], features['terminal'])
        action, state_next = (features['action_onehot'], features['state_next'])
        with tf.name_scope('NextQValue'):
            Q_s_next = q_model(state_next)
            best_action_next = tf.one_hot(tf.argmax(Q_s_next, axis=1), int(self.num_actions))

        loss = Q_loss(Q_s, action,
                      Q_s_next, best_action_next,
                      reward, terminal, self.discount)
        if (mode == tf.estimator.ModeKeys.TRAIN
            or mode == tf.estimator.ModeKeys.EVAL):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            eval_metric_ops = {}
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                              eval_metric_ops=eval_metric_ops,
                                              train_op=train_op)
