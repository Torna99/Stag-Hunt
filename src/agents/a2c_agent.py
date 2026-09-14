import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer

from models.a2c import A2CNet

import random
import numpy as np

class A2CAgent:

    def __init__(self, state_dim, action_dim, lr = 1e-3, gamma=0.99, entropy_coeff=0.001, critic_coeff=0.5, device="cpu"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        ## initialize hyperparameters: gamma is the discount factor, entropy is the weight for the entropy regularization term
        self.lr = lr
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.critic_coeff = critic_coeff

        ## Instatiate the network and the optimizer 
        self.acNet = A2CNet(state_dim, action_dim).to(self.device)
        self.optimizer = optimizer.Adam(self.acNet.parameters(), lr=self.lr)

        ## TODO: cosider using a buffer 
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []


    def select_action(self, state):
        '''
        Select an action based on the current state using the actor network.
        
        Args:
            state (torch.Tensor): The current state of the environment.
        Returns:
            action (torch.Tensor): The selected action.
            log_prob (torch.Tensor): The log probability of the selected action.
        '''

        logits, v_state = self.acNet(state)
        action_distribution = torch.distributions.Categorical(logits=logits)

        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)

        return action, log_prob, v_state

    def store_transition(self, state, action, reward, log_prob, value, done):
        '''
        Store the transition in the agent's memory.
        
        Args:
            state (torch.Tensor): The current state of the environment.
            action (torch.Tensor): The action taken by the agent.
            reward (float): The reward received after taking the action.
            log_prob (torch.Tensor): The log probability of the selected action.
            value (torch.Tensor): The value of the current state as estimated by the critic.
            done (bool): Whether the episode has ended.
        '''
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def wipe_memory(self):
        '''
        Clear the agent's memory.
        '''
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def discounted_rewards(self, rewards, dones, values, next_value):
        '''
        Compute the discounted rewards for the stored transitions.
        
        Args:
            next_value (float): The value of the next state.
            done (bool): Whether the episode has ended.
        Returns:
            discounted_rewards (torch.Tensor): The discounted rewards.
        '''
        disc_returns = torch.zeros(len(rewards), dtype=torch.float32).to(self.device)
        for i in reversed(range(len(rewards))):
            disc_returns[i] = rewards[i] + (self.gamma * next_value * (1 - dones[i]))
            next_value = disc_returns[i]
        return disc_returns

    def optimize_model(self, next_state, done):
        '''
        Optimize the actor and critic networks based on the stored transitions.
        
        Args:
            next_state (torch.Tensor): The next state of the environment after the last action.
            done (bool): Whether the episode has ended.
        '''

        if len(self.rewards) == 0:
            #return 0.0
            return None

        with torch.no_grad():
            _, next_value = self.acNet(next_state)
            next_value = next_value.item() if not done else 0.0

        discounted_returns = self.discounted_rewards(self.rewards, self.dones, self.values, next_value).unsqueeze(1).to(self.device)  # shape: (T, 1)

        # convert to Tensors 
        values = torch.cat(self.values)
        log_probs = torch.cat(self.log_probs).unsqueeze(1)

        # Advantage: R - V(s) 
        advantages = discounted_returns - values
        # Normalize the advantages to have mean 0 and std 1
        # if advantages.numel() > 1:
        #     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


        # compute losses
        critic_loss = F.mse_loss(values, discounted_returns)
        actor_loss = -(log_probs * advantages.detach()).mean()

        # Entropy regularization
        states_tensor = torch.cat(self.states)
        logits, _ = self.acNet(states_tensor)
        entropy = torch.distributions.Categorical(logits=logits).entropy().mean()

        # Total loss
        total_loss = actor_loss + self.critic_coeff * critic_loss - self.entropy_coeff * entropy    

        # Optimize the networks
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Clear the memory after optimization
        self.wipe_memory()  
        return total_loss.item()





        