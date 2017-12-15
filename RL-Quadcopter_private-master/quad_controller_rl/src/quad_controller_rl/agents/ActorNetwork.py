import numpy as np
import math
from keras.models import Model
from keras.layers import Dense, Input, merge
from keras.optimizers import Adam
from keras.initializers import Constant
import tensorflow as tf
import keras.backend as K

class ActorNetwork(object):
    def __init__(self, state_size, action_size, batch_size, tau, lra, hidden_units, bias_initializer):
        self.batch_size = batch_size
        self.tau = tau
        self.lra = lra
        self.bias_initializer = bias_initializer

        # set tensorflow session
        self.sess = tf.Session()
        K.set_session(self.sess)
        K.set_learning_phase(1)

        # now create the model
        self.model, self.weights, self.state_input = self.create_actor_network(state_size, action_size, hidden_units)   
        print(self.model.summary())
        self.target_model, self.target_weights, self.target_state_input = self.create_actor_network(state_size, action_size, hidden_units) 

        # for training the actor
        self.action_gradient = tf.placeholder(tf.float32, (None, 1, action_size)) 
        self.params_grad = tf.gradients(self.model.output, 
            self.model.trainable_weights, -self.action_gradient)
        self.grads = zip(self.params_grad, self.model.trainable_weights)
        self.optimize = tf.train.AdamOptimizer(self.lra).apply_gradients(self.grads)

        # initialize for later gradient calculations
        self.sess.run(tf.global_variables_initializer()) 

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.tau * actor_weights[i] + (1 - self.tau)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size, action_size, hidden_units):
        print("Now we build the model")
        state_input = Input(shape=(None, state_size))   
        h0 = Dense(hidden_units, activation='relu')(state_input)
        h1 = Dense(hidden_units, activation='relu')(h0)
        action_t = Dense(action_size, activation='tanh', bias_initializer = self.bias_initializer)(h1)  
        model = Model(inputs=state_input, outputs=action_t)
        return model, model.trainable_weights, state_input

    def train(self, states, grads):
        self.sess.run(self.optimize, feed_dict={
            self.state_input: states,
            self.action_gradient: grads
            })

    def save_weights(self, path):
        self.model.save_weights(path)
