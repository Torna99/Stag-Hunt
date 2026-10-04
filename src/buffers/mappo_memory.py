from collections import namedtuple, deque
import random

Transition = namedtuple(
    'Transition',
    (
        'global_state', 
        'obs1', 
        'obs2', 
        'action1', 
        'action2', 
        'reward', 
        'log_prob1', 
        'log_prob2', 
        'done',
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


# if __name__ == "__main__":
#     # Test the Memory class
#     memory = Memory()
#     for i in range(3):
#         global_state=f"global_state_{i}"
#         obs1=f"obs1_{i}"
#         obs2=f"obs2_{i}"
#         action1=f"action1_{i}"
#         action2=f"action2_{i}"
#         reward=i
#         log_prob1=f"log_prob1_{i}"
#         log_prob2=f"log_prob2_{i}"
#         done=(i % 2 == 0)
#         memory.push(global_state, obs1, obs2, action1, action2, reward, log_prob1, log_prob2, done)

#     all_transitions = memory.get_all()
#     print("All transitions:")
#     for transition in all_transitions:
#         print(transition.global_state)
