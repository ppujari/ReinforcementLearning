"""Combined agent."""

import numpy as np
from quad_controller_rl.agents.base_agent import BaseAgent
from .ReplayBuffer import ReplayBuffer
from .ActorNetwork import ActorNetwork
from .CriticNetwork import CriticNetwork
from keras.initializers import Constant

class DDPGCombined(BaseAgent):
    """Sample agent that searches for optimal policy randomly."""

    def __init__(self, task):
        # task (environment) information
        self.task = task  # should contain observation_space and action_space
        self.state_size = 1 #np.prod(self.task.observation_space.shape)
        self.state_range = self.task.observation_space.high - self.task.observation_space.low
        self.action_size = 1 #np.prod(self.task.action_space.shape)
        self.action_range = self.task.action_space.high - self.task.action_space.low

        # score tracker parameter
        self.best_score = -np.inf

        # learning parameters
        self.batch_size = 40
        self.buffer_size = 100000
        self.tau = 0.01
        self.lra = 0.0001
        self.hidden_units = 1
        self.hidden_units_hover = 3
        self.bias_initializer_actor = Constant(.5)

        # getting path for pre-trained networks
        self.actor_path_takeoff = '/home/robond/catkin_ws/src/RL-Quadcopter/takeoff_actor.h5'
        self.actor_path_hover = '/home/robond/catkin_ws/src/RL-Quadcopter/hover_actor.h5'
        self.actor_path_landing = '/home/robond/catkin_ws/src/RL-Quadcopter/landing_actor.h5'
        
        # create actors for each case
        self.actor_takeoff = ActorNetwork(self.state_size, self.action_size, 
            self.batch_size, self.tau, self.lra, self.hidden_units, self.bias_initializer_actor)
        self.actor_hover = ActorNetwork(self.state_size, self.action_size, 
            self.batch_size, self.tau, self.lra, self.hidden_units_hover, self.bias_initializer_actor)
        self.actor_landing = ActorNetwork(self.state_size, self.action_size, 
            self.batch_size, self.tau, self.lra, self.hidden_units, self.bias_initializer_actor)

        # load pre-trained weights
        self.actor_takeoff.model.load_weights(self.actor_path_takeoff)
        self.actor_hover.model.load_weights(self.actor_path_hover)
        self.actor_landing.model.load_weights(self.actor_path_landing)

        # episode variables
        # self.reset_episode_vars()

    def step(self, state, reward, done, task_id):
        # center state vector
        state = np.array([state[0, 2] - self.task.target_z])
        state = np.expand_dims(state, axis=0)

        # choose an action
        action = self.act(state, task_id) 
        
        ros_action = np.array([0.0, 0.0, action, 0.0, 0.0, 0.0])

        return ros_action

    def act(self, state, task_id):
        # choose action based on given state and policy
        if task_id == 'takeoff':
            action_unscaled = self.actor_takeoff.model.predict(np.expand_dims(state, axis=0))
        elif task_id == 'hover':
            action_unscaled = self.actor_hover.model.predict(np.expand_dims(state, axis=0))
        elif task_id == 'landing':
            action_unscaled = self.actor_landing.model.predict(np.expand_dims(state, axis=0))
        action = self.task.action_space.low[2] + (((action_unscaled + 1.0) / 2.0) * self.action_range[2])
        
        return action.reshape((1, 1))
