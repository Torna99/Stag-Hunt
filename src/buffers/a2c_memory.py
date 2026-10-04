from collections import namedtuple, deque
import random

Transition = namedtuple(
    "Transition",
    (
        "states",
        "actions",
        "rewards",
        "log_probs",
        "values",
        "dones"
    )
)

class Memory(object):
    '''
    A simple memory buffer to store transitions for the MAPPO agent.
    '''
    def __init__(self):
        self.buffer = deque()

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def pop(self):
        if self.buffer:
            return self.buffer.pop()
        else:
            raise IndexError("pop from an empty buffer")

    def get_all(self):
        return list(self.buffer)


    def wipe(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

