import gym
import numpy as np 
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
import pdb

from Replay_Buffer import Replay_Buffer


class Actor_Network(object):
    def __init__(self, env, sess, batch_size=32, tau=0.125, learning_rate=0.0001):
        self.env = env
        self.sess = sess

        self.obs_dim = self.env.num_states
        self.act_dim = self.env.num_actions

        # hyperparameters
        self.lr = learning_rate
        self.bs = batch_size 
        self.eps = 1.0
        self.eps_decay = 0.995
        self.gamma = 0.95
        self.tau = tau
        self.buffer_size = 5000
        self.hidden_dim = 300

        # replay buuffer
        self.replay_buffer = Replay_Buffer(self.buffer_size)

        # create model
        self.model, self.weights, self.state = self.create_actor()
        self.target_model, self.target_weights, self.target_state = self.create_actor()

        # gradients
        self.action_gradient = tf.placeholder(tf.float32, [None, self.act_dim])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)  # negative for grad ascend
        grads = zip(self.params_grad, self.weights)

        # optimizer & run
        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

    def create_actor(self):
        obs_in = Input(shape = [self.obs_dim])  # 3 states
        # pdb.set_trace()

        h1 = Dense(self.hidden_dim, activation = 'relu')(obs_in)
        h2 = Dense(self.hidden_dim, activation = 'relu')(h1)
        h3 = Dense(self.hidden_dim, activation = 'relu')(h2)

        out = Dense(self.act_dim, activation='tanh')(h3)

        model = Model(input = obs_in, output = out)

        # no loss function for actor apparently
        return model, model.trainable_weights, obs_in


    def train(self, states, action_grads):
        self.sess.run (self.optimize, 
                    feed_dict={self.state: states, self.action_gradient: action_grads})


    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()

        # update target network
        for i in range(len(actor_weights)):  # used to be xrange
            actor_target_weights[i] = self.tau*actor_weights[i] + (1 - self.tau)*actor_target_weights[i]

        self.target_model.set_weights(actor_target_weights)

