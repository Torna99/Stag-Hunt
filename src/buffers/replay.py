### Note: This code is based on the original paper and the implementation of DQN in the official pytorch tutorial
###
### Here we implement the replay buffer, which is used to store the experiences of the agent during training.
### It allows the agent to learn from past expereience, to stabilze the training and to not be biased by the most recent experiences.

import torch
from collections import namedtuple, deque
import random


Sample = namedtuple(
    'Sample',
    ('state', 'action', 'reward', 'next_state', 'done')
)

class ReplayMemory(object):

    def __init__(self, dim):
        self.mem = deque([], maxlen=dim)

    def push(self, *sample):
        self.mem.append(Sample(*sample))

    def samples_batch(self, batch_size):
        '''
            Return a batch of samples of bats_size dimension from the replay memory.
        '''
        return random.sample(self.mem, batch_size)

    def __len__(self):
        return len(self.mem)

