"""Policy search agent."""

import numpy as np
from quad_controller_rl.agents.base_agent import BaseAgent
from .ReplayBuffer import ReplayBuffer
from .ActorNetwork import ActorNetwork
from .CriticNetwork import CriticNetwork
from keras.initializers import Constant

class DDPGLanding(BaseAgent):
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
        self.lrc = 0.001
        self.hidden_units = 1
        self.gamma = 1.0
        self.bias_initializer_actor = Constant(.9)

        # noise parameters
        self.noise_process = np.zeros(self.action_size)
        self.noise_scale = 0.1 * (self.action_range[2]) * np.ones(self.action_size)
        self.noise_decay = 0.99
        self.exploration_mu = 0.0
        self.exploration_theta = 0.1
        self.exploration_sigma = 0.2

        # saving output and training result
        self.actor_path = '/home/robond/catkin_ws/src/RL-Quadcopter/landing_actor.h5'
        self.critic_path = '/home/robond/catkin_ws/src/RL-Quadcopter/landing_critic.h5'
        self.output_path = '/home/robond/catkin_ws/src/RL-Quadcopter/landing_output.txt'

        # create actor, critic, and replay buffer
        self.actor = ActorNetwork(self.state_size, self.action_size, 
            self.batch_size, self.tau, self.lra, self.hidden_units, self.bias_initializer_actor)
        self.critic = CriticNetwork(self.state_size, self.action_size, 
            self.batch_size, self.tau, self.lrc, self.hidden_units)
        self.buff = ReplayBuffer(self.buffer_size, self.batch_size)

        # episode variables
        self.reset_episode_vars()

    def reset_episode_vars(self):
        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0
        self.count = 0 # for calculating score

    def step(self, state, reward, done):
        # scale state vector to [0.0, 1.0] with z position only
        state = np.array([state[0, 2] - self.task.target_z])
        state = np.expand_dims(state, axis=0)
        # choose an action
        action = self.act(state) 
        
        # save experience / reward
        if self.last_state is not None and self.last_action is not None:
            self.count += 1 # for calculating score
            self.total_reward += reward # for calculating score
            self.buff.add(self.last_state, self.last_action, reward, state, done)
            self.train()

        # get final score and start new episode
        if done:
            self.get_score()
            self.reset_episode_vars()
            self.decay_noise()
            print(self.noise_scale)
            print(self.noise_process)

        self.last_state = state
        self.last_action = action

        ros_action = np.array([0.0, 0.0, action, 0.0, 0.0, 0.0])

        return ros_action

    def act(self, state):
        # choose action based on given state and policy
        action_unscaled = self.actor.model.predict(np.expand_dims(state, axis=0))
        action = self.task.action_space.low[2] + (((action_unscaled + 1.0) / 2.0) * self.action_range[2])
        self.noise_process += (self.exploration_theta * (self.exploration_mu - self.noise_process)) + (self.exploration_sigma * np.random.randn(self.action_size))
        action += self.noise_scale * self.noise_process
        return action.reshape((1, 1))

    def train(self):
        # do the batch update
        batch = self.buff.getBatch()
        states = np.asarray([e[0] for e in batch])
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        new_states = np.asarray([e[3] for e in batch])
        dones = np.asarray([e[4] for e in batch])
        y_t = np.asarray([e[1] for e in batch])

        target_q_values = self.critic.target_model.predict(
            [new_states, self.actor.target_model.predict(new_states)])

        for k in range(len(batch)):
            if dones[k]:
                y_t[k] = rewards[k]
            else:
                y_t[k] = rewards[k] + self.gamma*target_q_values[k] 

        # prep for training actor 
        a_for_grad = self.actor.model.predict_on_batch(states)
        grads = self.critic.get_grads(states, a_for_grad)
        # train actor
        self.actor.train(states, grads)
        # train target actor
        self.actor.target_train()
        # save actor weights
        self.actor.save_weights(self.actor_path)
        # train critic
        self.critic.train(states, actions, y_t)
        # train target critic
        self.critic.target_train()
        # save critic weights
        self.critic.save_weights(self.critic_path)


    def get_score(self):
        score = self.total_reward / float(self.count) if self.count else 0.0
        if score > self.best_score:
            self.best_score = score

        print("DDPG.learn(): t = {:4d}, score = {:7.3f} (best = {:7.3f})".format(
                self.count, score, self.best_score))  # [debug]

        with open(self.output_path, "a") as text_file:
            text_file.write(str(score) + "\n")

    def decay_noise(self):
        self.noise_scale *= self.noise_decay