from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt

def plot_mean_rwd(file_name):
	data = genfromtxt(file_name,delimiter=',',names=True,skip_header = 1)
	rewards = data['r']
	episodes = np.arange(len(rewards)) + 1
	steps_per_eps = data['l']
	total_steps = np.cumsum(steps_per_eps)
	mean_rwd_100_eps = np.convolve(rewards,np.ones(100)/100,mode='valid')
	plt.plot(total_steps[99:], mean_rwd_100_eps)
        
plt.rcParams['figure.figsize'] = 14,8	
 
plot_mean_rwd('MountainCar-v0-dqn.monitor.csv')
plot_mean_rwd('MountainCar-v0-dqn-noisy-net.monitor.csv')
plot_mean_rwd('MountainCar-v0-dqn-param-noise.monitor.csv')

plt.axhline(y=-110,color='r', linestyle='-')
plt.text(25000,-109,'-110: Environment Solved')
plt.xlabel('Number of Steps')
plt.ylabel('Mean Reward of 100 Episodes')
plt.legend(['Vanilla DDQN','DDQN with Noisy Networks','DDQN with Parameter Noise'],loc=1)
plt.savefig('test_img.png')
plt.show()	
