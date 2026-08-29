### Enrico Tornabene (0001191764)
###
### Note: This code is based on the implementation of DQN in the official pytorch tutorial 
###


import torch
from collections import namedtuple, deque
import random

# Sample = namedtuple(
#     "Sample", 
#         [
#             ("state_current", torch.Tensor),                                       
#             ("action", torch.Tensor),
#             ("reward", torch.Tensor),
#             ("future_state", torch.Tensor),
#             ("done", torch.Tensor)
#         ]
# )  
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
        return random.sample(self.mem, batch_size)

    def __len__(self):
        return len(self.mem)

