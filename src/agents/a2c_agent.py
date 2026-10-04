import torch 
import torch.nn as nn
import torch.optim as optimizer

from models.a2c import A2CNet
from buffers.a2c_memory import Memory

import random
import numpy as np

class A2CAgent:

    def __init__(self, state_dim, action_dim, lr = 1e-3, gamma=0.99, entropy_coeff=0.001, critic_coeff=0.5, device="cpu", **kwargs):
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

        self.memory = Memory()


    def select_action(self, state):
        '''
        Select an action based on the current state using the actor network.
        1) extract the logits and value from the actor-critic network
        2) create a categorical distribution from the logits
        3) sample an action from the distribution
        4) return the action, log probability of the action, and the value of the state
        '''
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device) 

        ## 1) extract logits and value network
        logits, v_state = self.acNet(state)

        ## 2) create categorical distribution
        action_distribution = torch.distributions.Categorical(logits=logits)

        ## 3) sample action from distribution
        action = action_distribution.sample()

        log_prob = action_distribution.log_prob(action)

        ## 4) return 
        return action.item(), log_prob, v_state

    def store_transition(self, state, action, reward, log_prob, value, done):
        '''
        Store the transition in the agent's memory.
        '''
        self.memory.push(state, action, reward, log_prob, value, done)

    def wipe_memory(self):
        '''
        Clear the agent's memory.
        '''
        self.memory.wipe()

    def discounted_rewards(self, rewards, dones, next_value):
        '''
        Compute the discounted rewards for the stored transitions.
        Following the formula: G_t = R_t + gamma * G_{t+1} * (1 - done)
        '''
        disc_returns = torch.zeros(len(rewards), dtype=torch.float32).to(self.device)
        for i in reversed(range(len(rewards))):
            disc_returns[i] = rewards[i] + (self.gamma * next_value * (1 - dones[i]))
            next_value = disc_returns[i]
        return disc_returns

    def optimize_model(self, next_state, done):
        '''
        Optimize the actor and critic networks based on the stored transitions.
        1) Compute the discounted returns for the stored transitions.
        2) Compute the advantages: A(s, a) = R - V(s)
        3) Compute the actor loss: L_actor = -log_prob * A(s, a)
        4) Compute the critic loss: L_critic = MSE(V(s), R)
        5) Compute the entropy regularization term
        6) Compute the total loss: L_total = L_actor + critic_coeff *   L_critic - entropy_coeff * entropy_regularization
        7) Backpropagate the total loss and update the network parameters.
        '''

        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)  # shape: (1, state_dim)

        transitions = self.memory.get_all()
        if len(transitions) == 0:
            return None
        self.states, self.actions, self.rewards, self.log_probs, self.values, self.dones = zip(*transitions)

        ## 1) Compute the discounted returns
        with torch.no_grad():
            _, next_value = self.acNet(next_state)
            next_value = next_value.item() if not done else 0.0

        discounted_returns = self.discounted_rewards(self.rewards, self.dones, next_value).unsqueeze(1).to(self.device)  # shape: (T, 1)

        # convert to Tensors 
        values = torch.cat(self.values)                      
        log_probs = torch.cat(self.log_probs).unsqueeze(1)    
         
        ## 2) Compute the advantages [A(s, a) = R - V(s)]
        advantages = discounted_returns - values
        # Normalize the advantages to have mean 0 and std 1
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


        ## 3, 4) Compute losses
        critic_loss = torch.nn.functional.mse_loss(values, discounted_returns)
        actor_loss = -(log_probs * advantages.detach()).mean()

        ## 5) Compute the entropy regularization term
        states_tensor = torch.tensor(self.states, dtype=torch.float32).to(self.device)
        logits, _ = self.acNet(states_tensor)
        entropy = torch.distributions.Categorical(logits=logits).entropy().mean()

        ## 6) Compute the total loss
        total_loss = actor_loss + self.critic_coeff * critic_loss - self.entropy_coeff * entropy    

        ## 7) Backpropagate the total loss and update 
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Clear the memory after optimization
        self.wipe_memory()  
        return total_loss.item()





        