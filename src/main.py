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
from Replay_Buffer import Replay_Buffer
from Actor_Network import Actor_Network
from Critic_Network import Critic_Network
import keras.backend as K


def ou_func(x, mu, theta, sigma=0.3):
    return theta * (mu - x) + sigma * np.random.randn(1)


def train_quad(debug=True):
    
    env = environment.Environment(debug)  # Rohit's custom environment

    obs_dim = env.num_states
    act_dim = env.num_actions

    buffer_size = 5000
    batch_size = 32
    gamma = 0.98
    tau = 0.001

    np.random.seed(1337)

    vision = False

    explore = 100000
    eps_count = 20000

    max_steps = 100000
    reward = 0
    done = False
    epsilon = 1
    indicator = 0

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

    ###########
    #  ROHIT  #
    ###########
    n_states = env.num_states + len(env.goalPos) + 1  # 1 is for battery
    goal_pos = env.goalPos
    battery = env.battery
    actor = Actor_Network(env, sess, n_states)
    critic = Critic_Network(env, sess, n_states)
    replay_buffer = Replay_Buffer()

    # try:
    #   actor.model.load_weights("actormodel.h5")
    #   critic.model.load_weights("criticmodel.h5")
    #   actor.target_model.load_weights("actormodel.h5")
    #   critic.target_model.load_weights("criticmodel.h5")
    #   print("Weight load successfully")
    # except:
    #   print("WOW WOW WOW, Cannot find the weight")

    save_dir = os.path.join(os.getcwd(), 'saved_models_hari_6')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    os.chdir(save_dir)

    plt.ion()
    plt.title('Training Curve')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.grid()

    for epi in range (eps_count):

        s_t = env._reset()  # state
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

            for k in range (len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + gamma*target_q_values[k]

        
            loss += critic.model.train_on_batch([states, actions], y_t)
            a_for_grad = actor.model.predict(states)
            grads = critic.gradients(states, a_for_grad)
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

                # print ('saving model')
                # actor.model.save_weights("actormodel.h5", overwrite=True)
                # with open("actormodel.json", "w") as outfile:
                #     json.dump(actor.model.to_json(), outfile)

                # critic.model.save_weights("criticmodel.h5", overwrite=True)
                # with open("criticmodel.json", "w") as outfile:
                #     json.dump(critic.model.to_json(), outfile)
        print('episode: {} step: {} total reward: {} battery level: {}'.format(epi+1,step,total_reward,env.battery))
        ############# Plotting states ############
        # if plot_state:
        #     states = env.plotState
        #     xs = states[:,0]
        #     ys = states[:,1]
        #     zs = states[:,2]

        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection='3d')

        #     ax.plot(xs, ys, zs)
        #     ax.set_xlabel('X')
        #     ax.set_ylabel('Y')
        #     ax.set_zlabel('Z')
        #     # plt.show()
        #     save_path = './plots/'+str(e)+'.png'
        #     plt.savefig(save_path)
        #########################################

    ################ Plotting rewards ############## 
        if plot_reward:
            episode_rewards.append(total_reward)
            episode.append(epi+1)
            plt.plot(episode,episode_rewards,'b')
            plt.pause(0.001)
        
    plt.savefig("Training Curve.png")

    ################################################

def test_quad(debug = True):
    env = environment.Environment(debug)  # Rohit's custom environment

    obs_dim = env.num_states
    act_dim = env.num_actions

    gamma = 0.98
    tau = 0.001

    vision = False

    eps_count = 3
    max_steps = 100000
    reward = 0
    done = False
    indicator = 0

    plot_state = False
    plot_reward = True

    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    # actor, critic and buffer
    dir_name = 'saved_models_hari_6' 
    load_dir = os.path.join(os.getcwd(), dir_name)


    plt.ion()
    plt.title('Testing Curve')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.grid()

    model_num = []
    mean_reward = []
    #not the numbers, they are based on how i saved
    for i in range(1250,2050,50): #(50,1050,50)
         #change this manually according to ur saved models
    
        actor_model_name = '%d_actor_model.h5' %(i)
        critic_model_name = '%d_critic_model.h5' %(i)       
        filepath1 = os.path.join(load_dir, actor_model_name)
        actor = load_model(filepath1)
        #filepath2 = os.path.join(load_dir, critic_model_name)
        #critic.model = load_model(filepath2) 
        cumulative_reward = []
        model_num.append(i)

        for epi in range (eps_count):

            # receive initial observation state
            s_t = env._reset()  # cos theta, sin theta, theta dot
            s_t = np.asarray(s_t)
            total_reward = 0
            done = False
            step = 0

            while(done == False):
                if step > 200:
                    break
                
                step += 1
                # print('--------------------------------')
                # print('step: {}'.format(step))

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
        mean_reward.append(np.mean(cumulative_reward))
        plt.plot(model_num,mean_reward,'b')
        plt.pause(0.001)

    save_dir = os.path.join(os.getcwd(), 'saved_models_hari_6')
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
    train_indicator = args.train  # Training = 1, Test = 0
    debug = args.debug  # If you want debugging print statements
    if train_indicator==1:
        print("------------- starting training----------------")
        train_quad(debug)
    # else:
        print("------------- starting testing----------------")
        test_quad(debug)
    else:
        test_quad(debug)
