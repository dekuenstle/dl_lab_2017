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
	plt.plot(mean_rwd_100_eps)
	


#data_dqn = genfromtxt('MountainCar-v0-dqn.monitor.csv',delimiter=',',names=True,skip_header = 1)
#data_noisy = genfromtxt('MountainCar-v0-dqn-noisy-net.monitor.csv',delimiter=',',names=True,skip_header = 1)

 
plot_mean_rwd('MountainCar-v0-dqn.monitor.csv')
plot_mean_rwd('MountainCar-v0-dqn-noisy-net.monitor.csv')
plot_mean_rwd('MountainCar-v0-dqn-param-noise.monitor.csv')

plt.axhline(y=-110,color='r', linestyle='-')
plt.show()	#return mean_rwd_100_eps
