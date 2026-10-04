### Note: This code is based on the original paper and the implementation of DQN in the official pytorch tutorial
###
### In this code we implement a standard DQN agent:
###     We'll use a Policy Network (Q-Network) to approximate the Q-function, and a Replay Memory to 
###     store experiences  to avoid correlation between consecutive updates.The agent will use an 
###     epsilon-greedy policy for action selection (this is an off-policy algorithm). 

import torch 
import torch.nn as nn
import torch.optim as optimizer

from buffers.replay import ReplayMemory
from models.dqn import DQN

import random
import numpy as np

class DQNAgent:

    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.1, buffer_size = 1e4, batch_size=32, device="cpu", **kwargs):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        ## initialize hyperparameters: gamma is the discount factor
        self.lr = lr
        self.gamma = gamma

        ## epsilon is the exploration for epsilon-greedy policy
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.batch_size = batch_size
        self.buffer_size = buffer_size

        ## instatiate the replay memory (to avoid correlation between consecutive updates)
        self.rep_memory = ReplayMemory(int(buffer_size))

        ## instatiate the policy
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        
        ## instatiate the optimizer (use Adam inseatd of SGD for better performance) and the loss function (MSE)
        self.optimizer = optimizer.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

    def select_action(self, state):
        '''
        Select an action based on the current state using epsilon-greedy policy.
        '''
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        sample = random.random()

        ## Exploration
        if sample < self.epsilon:
            return random.randrange(self.action_dim)

        ## Exploitation
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return self.policy_net(state).max(1).indices.view(1, 1).item()

    def epsilon_decay_step(self):
        '''
        Decay the epsilon value in a multiplicative manner until it reaches the minimum value.
        '''
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_epsilon(self):
        '''
        Used for logging the current epsilon value.
        '''
        return self.epsilon

    def store_sample(self, state, action, reward, next_state, done):
        self.rep_memory.push(state, action, reward, next_state, done)

    def optimize_model(self):
        '''
        This function is the single train step of the DQN agent. 
            1) Sample a batch of experiences from the replay memory to ensure that the updates are uncorrelated and more stable
            2) Compute the Q-values for the current state-action pairs using the policy network
            3) Compute the target: y = r + gamma * max_a' Q(s', a', theta) for non-terminal states, and y = r for terminal states
            4) Compute the loss between the current Q-values and the target Q-values
            5) Perform a gradient descent step to update the policy network parameters
        '''

        # if rep memory is not filled, return
        if len(self.rep_memory) < self.batch_size:
            return 

        ## 1) Sampling 
        samples = self.rep_memory.samples_batch(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        # Convert the samples to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(
            [a.item() if isinstance(a, torch.Tensor) else int(a) for a in actions],
            dtype=torch.long,
            device=self.device
        ).view(-1, 1)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device).view(-1, 1)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(self.device).view(-1, 1)
        
        ## 2) compute Q(s_t, a)
        state_action_values = self.policy_net(states).gather(1, actions)

        ## 3) compute target 
        # y = r + gamma * max_a' Q(s', a', theta) for non-terminal states
        with torch.no_grad():
            # get the q values for the next states: max_a' Q(s', a', theta)
            next_state_values = self.policy_net(next_states).max(1)[0].unsqueeze(1)

            # compute the expected Q values (target q for the loss function)
            # use (1 - dones) to handle the terminal states
            y_j = rewards + (self.gamma * next_state_values * (1 - dones))

        ## 4) compute the loss between the current Q-values and the target Q-values
        loss = self.loss(state_action_values, y_j)

        ## 5) optimize the model 
        self.optimizer.zero_grad() 
        loss.backward()
        self.optimizer.step()


        # return the mean of the estimated q-values for the current state-action pairs to monitor the overestimation bias during training
        return state_action_values.mean().item()

