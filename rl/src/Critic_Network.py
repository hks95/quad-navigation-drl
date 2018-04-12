import gym
import numpy as np
import math
# from keras.initializations import normal, identity
from keras.models import model_from_json, load_model
# from keras.engine.training import collect_trainable_weights
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf


class Critic_Network(object):
    def __init__(self, env, sess, batch_size=32, tau=0.125, learning_rate=0.001):
        self.env = env
        self.sess = sess
        self.bs = batch_size

        self.obs_dim = self.env.num_states
        self.act_dim = self.env.num_actions

        # hyperparameters
        self.lr = learning_rate
        self.bs = batch_size 
        self.tau = tau
        self.buffer_size = 5000
        self.hidden_dim = 300

        K.set_session(sess)

        self.model, self.action, self.state = self.create_critic_network()
        self.target_model, self.target_action, self.target_state = self.create_critic_network()
        self.action_grads = tf.gradients (self.model.output, self.action)
        self.sess.run(tf.initialize_all_variables())


    def create_critic_network(self):

        # parallel 1
        state_input = Input(shape = [self.obs_dim])
        w1 = Dense(self.hidden_dim, activation = 'relu')(state_input)
        h1 = Dense(self.hidden_dim, activation = 'linear')(w1)

        # parallel 2
        action_input = Input(shape = [self.act_dim], name = 'action2')
        a1 = Dense(self.hidden_dim, activation = 'linear')(action_input)

        # merge
        h2 = merge([h1, a1], mode = 'sum')
        h3 = Dense(self.hidden_dim, activation = 'relu')(h2)
        value_out = Dense(self.act_dim, activation = 'linear')(h3)

        model = Model(input = [state_input, action_input], output = [value_out])
        adam = Adam(self.lr)
        model.compile(loss = 'mse', optimizer = adam)

        return model, action_input, state_input


    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict = {
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()

        for i in range (len(critic_weights)):  # used to be xrange
            critic_target_weights[i] = self.tau*critic_weights[i] + (1 - self.tau)*critic_target_weights[i]

        self.target_model.set_weights(critic_target_weights)