### Note: This code is based on the paper "Human-level control through deep reinforcement learning" by Mnih et al.
###
###
### Using only one network (the policy one) lead us to unstable training and divergence of the Q-values. This happens
### because the target values are changing at each step of the training, which makes the learning process unstable, trying to reach 
### a moving target (moving target problem). The estimate of the Q-vlaues [Q(s,a, theta)] and the actual target [r + gamma * max_a' Q(s',a', theta)] 
### use the same parameters (theta) and this leads to the moving target problem.
### 
### To solve this problem we use also a target network, which is a copy of the policy net and is updated only every C steps. 
### The new weights of the target net will be the same as the policy net. In this way the target values are  more stable and 
### the learning process is more stable.

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer

from buffers.replay import ReplayMemory, Sample
from models.dqn import DQN

import random
import numpy as np

class DQNAgent:

    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1, buffer_size = 1e4, batch_size=32, device="cpu"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        ## initialize C: number of steps after which the target network is updated with the policy network weights
        self.C = 50 

        ## initialize hyperparameters: gamma is the discount factor
        self.lr = lr
        self.gamma = gamma

        ## epsilon is the exploration for epsilon-greedy policy
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        ## instatiate the replay memory
        self.rep_memory = ReplayMemory(int(buffer_size))

        ## instatiate the policy and target networks
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)

        ## copy the weights of the policy net to the target net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # set the target net in evaluation mode since we don't want to update its weights during training via backpropagation (so using optimizer.step())
        
        ## instatiate the optimizer (use Adam inseatd of SGD for better performance) and the loss function (MSE)
        self.optimizer = optimizer.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

   
    def select_action(self, state):
        '''
        Select an action based on the current state using epsilon-greedy policy.
        '''
        sample = random.random()

        ## Exploration
        if sample < self.epsilon:
            return torch.tensor([[random.randrange(self.action_dim)]], dtype=torch.long)

        ## Exploitation
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return self.policy_net(state).max(1).indices.view(1, 1)

    # def epsilon_decay_step(self):
    #     '''
    #     Decay the epsilon value.
    #     '''
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon -= self.epsilon_decay
    #         #self.epsilon *= self.epsilon_decay
    #         if self.epsilon < self.epsilon_min:
    #             self.epsilon = self.epsilon_min

    def epsilon_decay_step(self):
        '''
        Decay the epsilon value.
        '''
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_epsilon(self):
        '''
        Used for logging the current epsilon value.
        '''
        return self.epsilon

    def store_sample(self, state, action, reward, next_state, done):
        self.rep_memory.push(state, action, reward, next_state, done)

    
    def update_target_network(self):
        '''
        Update the target network with the weights of the policy network.
        '''
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def optimize_model(self):
        '''
        This function is the single train step of the DQN agent. 
        It samples a batch of experiences from the replay memory, computes the loss, and updates the networks.
        '''

        # if rep memory is not filled, return
        if len(self.rep_memory) < self.batch_size:
            return 

        # sampling 
        samples = self.rep_memory.samples_batch(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        # Convert the samples to tensors
        states = torch.cat(states, dim=0).to(self.device).float()
        next_states = torch.cat(next_states, dim=0).to(self.device).float()
        actions = torch.tensor(
            [a.item() if isinstance(a, torch.Tensor) else int(a) for a in actions],
            dtype=torch.long,
            device=self.device
        ).view(-1, 1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).view(-1, 1)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).view(-1, 1)
        
        # compute Q(s_t, a) the current estimate of the Q-values for the current state-action pairs
        state_action_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():

            # evaluate the target Q-values using the target network for the next states
            next_state_values = self.target_net(next_states).max(1)[0].unsqueeze(1)

            # temporal difference target: r + gamma * max_a' Q(s',a', theta_target)
            expected_state_action_values = rewards + (self.gamma * next_state_values * (1 - dones))


        # compute the loss
        loss = self.loss(state_action_values, expected_state_action_values)

        # optimize the model 
        self.optimizer.zero_grad() 
        loss.backward()
        self.optimizer.step()

