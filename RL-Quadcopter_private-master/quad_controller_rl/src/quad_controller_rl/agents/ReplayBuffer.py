from collections import deque
import random

class ReplayBuffer(object):

    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer_count = 0
        self.buffer = deque(maxlen=buffer_size)

    def getBatch(self):
        # Randomly sample examples
        if self.buffer_count < self.batch_size:
            return random.sample(self.buffer, self.buffer_count)
        else:
            return random.sample(self.buffer, self.batch_size)

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        self.buffer.append(experience)
        if self.buffer_count < self.buffer_size:
            self.buffer_count += 1
