#!/usr/bin/env python3

import gym


env = gym.make('Breakout-v0')
observation = env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    if done:
        print("Episode finished.")
        break

    lives_left = info['ale.lives']
    # nothing: 0, nothing: 1, right: 2, left: 3
    print("action {} (total {}), observation size {}, reward {}, lives {}"
          .format(action, env.action_space.n, observation.shape, reward, lives_left))
