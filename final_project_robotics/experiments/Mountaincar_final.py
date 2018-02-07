import gym
import tensorflow as tf
from baselines import deepq
from baselines.bench import Monitor

def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = len(lcl['episode_rewards']) > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= -110
    #print(lcl['t'], lcl['episode_rewards'])
    return is_solved


def main():
    env = "MountainCar-v0"
    run_config(env, "dqn")
    run_config(env, "dqn-noisy-net", noisy_net = True)
    run_config(env, "dqn-param-noise", param_noise = True)

def run_config(env_name, run_name, **kwargs):
    with tf.Graph().as_default():
        tf.set_random_seed(42)
        env = gym.make(env_name)    
        monitor = Monitor(env, env_name + '-' + run_name)
        env = monitor
        model = deepq.models.mlp([64], layer_norm = True, sigma0=0.4)
        act = deepq.learn(
            env,
            q_func=model,
            lr=1e-3,
            max_timesteps=200000,
            buffer_size=50000,
            exploration_fraction=0.1,
            exploration_final_eps=0.02,
            print_freq=10,
            callback=callback,
            **kwargs
        )


if __name__ == '__main__':
    main()
