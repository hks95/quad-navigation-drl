'''
The main function to execute the DDPG algorithm on the ROS hector gazebo quadcopter
'''

import environment
import rospy
import time
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import argparse
from keras.models import model_from_json, Model,load_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
import os
import json
import pdb
import argparse
import keras.backend as K

'''
Loading the helper functions here - actor, critic and experience replay
'''
from Replay_Buffer import Replay_Buffer
from Actor_Network import Actor_Network
from Critic_Network import Critic_Network


def ou_func(x, mu, theta, sigma=0.3):
	'''
	Ornstein-Uhlenbeck process for noise
	'''
	return theta * (mu - x) + sigma * np.random.randn(1)


def train_quad(debug=True):
	'''
	function to interact with quad in gazebo and train it
	'''

	# Rohit's custom environment
	env = environment.Environment(debug)  

	obs_dim = env.num_states
	act_dim = env.num_actions

	# hyper parameters
	buffer_size = 5000
	batch_size = 32
	gamma = 0.98
	tau = 0.001
	explore = 100000
	eps_count = 20000
	epsilon = 1

	# might be useful to remove this and train
	np.random.seed(1337)

	# flags to turn on/off plotting
	plot_state = False
	plot_reward = True

	episode_rewards = []
	episode = []

	#Tensorflow GPU optimization
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	from keras import backend as K
	K.set_session(sess)

	# actor, critic and buffer
	n_states = env.num_states  # 1 is for battery
	goal_pos = env.goalPos
	battery = env.battery
	actor = Actor_Network(env, sess, n_states)
	critic = Critic_Network(env, sess, n_states)
	replay_buffer = Replay_Buffer()

	# directory to save results
	save_dir = os.path.join(os.getcwd(), 'results_wo_batterytermination_wo_goal')
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	os.chdir(save_dir)

	# inititating the plotting
	plt.ion()
	plt.title('Training Curve')
	plt.xlabel('Episodes')
	plt.ylabel('Total Reward')
	plt.grid()

	for epi in range (eps_count):

		s_t = env._reset()
		s_t = np.asarray(s_t)

		total_reward = 0
		done = False
		step = 0

		while(done == False):
			if step > 200:
				break
			
			step += 1

			loss = 0
			epsilon -= 1.0/explore

			a_t = np.zeros([1, act_dim])
			noise_t = np.zeros([1, act_dim])

			# select action according to current policy and exploration noise
			a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
			
			noise_t[0][0] = max(epsilon,0) * ou_func(a_t_original[0][0],  0.0 , 0.60, 0.30)
			noise_t[0][1] = max(epsilon,0) * ou_func(a_t_original[0][1],  0.0 , 0.60, 0.30)
			noise_t[0][2] = max(epsilon,0) * ou_func(a_t_original[0][2],  0.0 , 0.60, 0.30)

			a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
			a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
			a_t[0][2] = a_t_original[0][2] + noise_t[0][2]


			s_t1, r_t, done, _ = env._step(a_t[0])
			s_t1 = np.asarray(s_t1)

			# add to replay buffer
			replay_buffer.add(s_t, a_t[0], r_t, s_t1, done)

			# sample from replay buffer
			batch = replay_buffer.sample_batch()
			states = np.asarray([e[0] for e in batch])
			actions = np.asarray([e[1] for e in batch])
			rewards = np.asarray([e[2] for e in batch])
			new_states = np.asarray([e[3] for e in batch])
			dones = np.asarray([e[4] for e in batch])
			y_t = np.asarray([e[1] for e in batch])

			target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

			# get the reward for the full batch
			for k in range (len(batch)):
				if dones[k]:
					y_t[k] = rewards[k]
				else:
					y_t[k] = rewards[k] + gamma*target_q_values[k]

			# calculate critic loss function 
			loss += critic.model.train_on_batch([states, actions], y_t)
			
			# compute gradients
			a_for_grad = actor.model.predict(states)
			grads = critic.gradients(states, a_for_grad)
			
			# train and update weights
			actor.train(states, grads)
			actor.target_train()
			critic.target_train()

			total_reward += r_t
			s_t = s_t1

		if ((epi+1)%50 == 0):
			a_model_name = '%d_actor_model.h5' %(epi+1)
			c_model_name = '%d_critic_model.h5' % (epi+1)
			filepath = os.path.join(save_dir, a_model_name)
			actor.model.save(a_model_name)
			critic.model.save(c_model_name)

		print('epi: {} step: {} reward: {:.2f} battery: {}'.format(epi+1,step,total_reward,int(env.battery)))
		print('-----------------------------------------------')

	# Plotting rewards 
		if plot_reward:
			episode_rewards.append(total_reward)
			episode.append(epi+1)
			plt.plot(episode,episode_rewards,'b')
			plt.pause(0.001)
		
	plt.savefig("Training Curve.png")


def test_quad(debug = True):
	''' 
	Testing function for the quad
	'''

	# Rohit's custom environment 
	env = environment.Environment(debug)  

	obs_dim = env.num_states
	act_dim = env.num_actions

	# hyper parameters
	gamma = 0.98
	tau = 0.001
	eps_count = 10
	max_steps = 100000
	reward = 0

	# flags to turn on/off plotting
	plot_state = False
	plot_reward = True

	#Tensorflow GPU optimization
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	from keras import backend as K
	K.set_session(sess)

	# actor, critic and buffer
	dir_name = 'results_wo_batterytermination_wo_goal' 
	load_dir = os.path.join(os.getcwd(), dir_name)

	# initiating the plotting
	plt.ion()
	plt.title('Testing Curve')
	plt.xlabel('Episodes')
	plt.ylabel('Total Reward')
	plt.grid()

	model_num = []
	mean_reward = []

	# look at the training plot, load the best range of models. make the uppler limit 50 more than the final model
	for i in range(500,2100,50):

		actor_model_name = '%d_actor_model.h5' %(i)

		filepath1 = os.path.join(load_dir, actor_model_name)

		# load only the actor here because the agent is the actor
		actor = load_model(filepath1)
		cumulative_reward = []
		model_num.append(i)

		for epi in range (eps_count):

			# receive initial observation state
			s_t = env._reset() 
			s_t = np.asarray(s_t)
			total_reward = 0
			done = False
			step = 0

			while(done == False):
				if step > 200:
					break
				
				step += 1
				loss = 0
				a_t = np.zeros([1, act_dim])
				# select action according to current policy and exploration noise
				a_t_original = actor.predict(s_t.reshape(1, s_t.shape[0]))
				a_t[0][0] = a_t_original[0][0]
				a_t[0][1] = a_t_original[0][1]
				a_t[0][2] = a_t_original[0][2]

				s_t1, r_t, done, _ = env._step(a_t[0])
				s_t1 = np.asarray(s_t1)
				total_reward += r_t
				s_t = s_t1

			print('episode: {} step: {} total reward: {} battery level: {}'.format(epi+1,step,total_reward,env.battery))
			cumulative_reward.append(total_reward)
			# print(epi)

		print('model: {} mean reward: {}'.format(i,np.mean(cumulative_reward)))
		print('----------------------------------------')
		mean_reward.append(np.mean(cumulative_reward))
		plt.plot(model_num,mean_reward,'b')
		plt.pause(0.001)

	save_dir = os.path.join(os.getcwd(), 'results_wo_batterytermination_wo_goal')
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	os.chdir(save_dir)          
	plt.savefig("Learning Curve.png")

import signal, sys
def signal_handler(signal, frame):
	reason = 'Because'
	rospy.signal_shutdown(reason)
	sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def parse_arguments():
	parser = argparse.ArgumentParser(description='DDPG Network Argument Parser')
	parser.add_argument('--train',dest='train',type=int,default=1)
	parser.add_argument('--debug',dest='debug',type=bool,default=False)
	return parser.parse_args()    

if __name__ == "__main__":
	rospy.init_node('quad', anonymous=True)
	args = parse_arguments()

	# Training = 1, Test = 0
	train_indicator = args.train  

	# If you want debugging print statements
	debug = args.debug  

	if train_indicator==1:
		train_quad(debug)
	else:
		test_quad(debug)