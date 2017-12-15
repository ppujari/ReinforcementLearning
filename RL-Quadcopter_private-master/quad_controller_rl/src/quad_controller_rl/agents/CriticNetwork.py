import numpy as np
import math
from keras.layers import Dense, Flatten, Input, Add, Lambda, Activation
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

class CriticNetwork(object):
    def __init__(self, state_size, action_size, batch_size, tau, lrc, hidden_units):
        self.batch_size = batch_size
        self.tau = tau
        self.lrc = lrc

        # set tensorflow session
        self.sess = tf.Session()
        K.set_session(self.sess)
        K.set_learning_phase(1)
        
        # now create the model
        self.model, self.action_input, self.state_input = self.create_critic_network(state_size, action_size, hidden_units)  
        print(self.model.summary())
        self.target_model, self.target_action_input, self.target_state_input = self.create_critic_network(state_size, action_size, hidden_units) 

        # for training the actor
        self.action_grads = tf.gradients(self.model.output, self.action_input) 

        # Initialize for later gradient calculations
        self.sess.run(tf.global_variables_initializer()) 

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.tau * critic_weights[i] + (1 - self.tau)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size, action_size, hidden_units):
        print("Now we build the model")
        state_input = Input(shape=(None, state_size))  
        action_input = Input(shape=(None, action_size))   
        w1 = Dense(hidden_units, activation='relu')(state_input)
        h1 = Dense(hidden_units, activation='linear')(w1)
        a1 = Dense(hidden_units, activation='linear')(action_input) 
        h2 = Add()([h1,a1])    
        h3 = Dense(hidden_units, activation='relu')(h2)
        output = Dense(action_size, activation='linear')(h3)   
        model = Model(inputs=[state_input, action_input], outputs=output)
        adam = Adam(lr=self.lrc)
        model.compile(loss='mse', optimizer=adam)
        return model, action_input, state_input 

    def get_grads(self, states, a_for_grad):
        grads = self.sess.run(self.action_grads, feed_dict={
            self.state_input: states,
            self.action_input: a_for_grad
            })[0]
        return grads

    def train(self, states, actions, y_t):
        self.model.train_on_batch([states, actions], y_t)

    def save_weights(self, path):
        self.model.save_weights(path)