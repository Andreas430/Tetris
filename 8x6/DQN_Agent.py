#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import tensorflow 
from tensorflow import keras
from tensorflow.keras import layers
import gym
import tensorflow as tf
import time
from matplotlib import animation
import matplotlib.pyplot as plt
import os
import pickle
from matplotlib import animation
from collections import deque
import cv2
from PIL import Image
from time import sleep
import random

class DQN_Agent():
    def __init__(self,epsilon =0.9, gamma = 0.99,update_after_actions = 1,
                 epsilon_greedy_frames = 250000, epsilon_random_frames =12500,
                 epsilon_min = 0.1, max_memory_length = 125000,
                 update_network = 2500, batch_size =32,num_actions=24):
        
        #initialization of the parameters of the agent
        self.num_actions = num_actions
        self.actions = [i for i in range(num_actions)]
        self.max_memory_length = max_memory_length
        self.loss_function = tensorflow.keras.losses.Huber()
        self.optimizer = tensorflow.keras.optimizers.Adam(learning_rate = 0.00025, clipnorm = 1)
        self.epsilon_random_frames = epsilon_random_frames
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_min = epsilon_min
        self.epsilon_greedy_frames = epsilon_greedy_frames
        self.epsilon = epsilon
        self.epsilon_interval = 1- self.epsilon_min
        self.update_after_actions = update_after_actions
        self.buffer_actions = deque(maxlen = max_memory_length)
        self.buffer_state = deque(maxlen = max_memory_length)
        self.buffer_nextstate = deque(maxlen = max_memory_length)
        self.buffer_done = deque(maxlen = max_memory_length)
        self.buffer_rewards = deque(maxlen = max_memory_length)

        self.frames = 0
        self.update_network = update_network
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        
    def update_replay_buffer(self,action,state,state_next,done,reward):
        #filling the replay memory with the recent state, next state ,action,reward , and done information
        self.buffer_actions.append(action)
        self.buffer_state.append(state)
        self.buffer_nextstate.append(state_next)
        self.buffer_done.append(done)
        self.buffer_rewards.append(reward)
        
    def update_weights(self):
        #update the weights of the target model 
        self.target_model.set_weights(self.model.get_weights())
        
    def choose_action(self,state):
      #choosing an action based on an epsilon greedy policy
        self.frames += 1
        if np.random.uniform(0,1) < self.epsilon or self.frames <self.epsilon_random_frames:
            action = np.random.choice(self.actions)
        else:
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            # outputs the Q value for each action 
            Q_val = self.model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(Q_val[0]).numpy()
        
        #epsilon decay
        self.epsilon -= self.epsilon_interval /self.epsilon_greedy_frames
        # epsilon capped at minimum epsilon (0.1) 
        self.epsilon = max(self.epsilon,self.epsilon_min)
        
        return action
    
    def learn(self):
      #based on the hyperparameters , the agent samples states from the replay memory to update its weights
        if self.frames % self.update_after_actions==0 and len(self.buffer_actions) > self.batch_size:
            indices = np.random.choice(range(len(self.buffer_actions)), size = self.batch_size)
            state_sample = np.array([self.buffer_state[i] for i in indices])
            state_next_sample = np.array([self.buffer_nextstate[i] for i in indices])
            rewards_sample = [self.buffer_rewards[i] for i in indices]
            action_sample = [self.buffer_actions[i] for i in indices]
            done_sample = tf.convert_to_tensor([float(self.buffer_done[i]) for i in indices])
            
            
            future_rewards = self.target_model.predict(state_next_sample)
            updated_q_values = rewards_sample + self.gamma *tf.reduce_max(future_rewards, axis =1)
            
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample
            
            masks = tf.one_hot(action_sample, self.num_actions)
            
            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = self.model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = self.loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        #update the weights of the target network each 10000 frames 
        if self.frames % self.update_network == 0:
            self.update_weights()
            
    def create_model(self):
        # Network defined by the Deepmind paper
        inputs = layers.Input(shape=(8, 11,))

        # Convolutions on the frames on the screen
        layer1 = layers.Conv1D(32, 3, strides=3, activation="relu",kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2))(inputs)
        layer2 = layers.Conv1D(64, 2, strides=2, activation="relu",kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2))(layer1)

        layer4 = layers.Flatten()(layer2)

        layer5 = layers.Dense(512, activation="relu",kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2))(layer4)
        action = layers.Dense(self.num_actions, activation="linear",kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2))(layer5)

        return keras.Model(inputs=inputs, outputs=action)
    
    def save_model(self,episode_num):
        self.model.save('models/model_frames_'+ str(self.frames)+ '.h5')
        print(f'Model saved succesfully at episode {episode_num}', flush = True)
        
    def save_buffer(self,i):
      #save the current buffer in case of programm crashes
        with open('buffer/history.txt','w') as b:
            b.write(f'Latest Frames Saved : {self.frames} Episode : {i}\n')
        pickle.dump( self.buffer_rewards, open('buffer/buffer_rewards.p', "wb" ))
        pickle.dump( self.buffer_state, open('buffer/buffer_state.p', "wb" ))
        pickle.dump( self.buffer_nextstate, open('buffer/buffer_nextstate.p', "wb" ))
        pickle.dump( self.buffer_actions, open('buffer/buffer_actions.p', "wb" ))
        pickle.dump( self.buffer_done, open('buffer/buffer_done.p', "wb" ))

    def load_model_buffer(self, frames, epsilon):
        self.model.load_weights('models/model_frames_' + str(frames) + '.h5')
        self.update_weights()
        self.epsilon = epsilon
        self.frames = frames
        with open('buffer/buffer_rewards.p', 'rb') as f:
            self.buffer_rewards = pickle.load(f)
        with open('buffer/buffer_state.p', 'rb') as f:
            self.buffer_state = pickle.load(f)
        with open('buffer/buffer_nextstate.p', 'rb') as f:
            self.buffer_nextstate = pickle.load(f)
        with open('buffer/buffer_actions.p', 'rb') as f:
            self.buffer_actions = pickle.load(f)
        with open('buffer/buffer_done.p', 'rb') as f:
            self.buffer_done = pickle.load(f)
        print('----------Loaded model and buffer Succesfully----------')