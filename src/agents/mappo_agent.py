#
#   global_state: the global state of the environment (the state of the environment from a centralized perspective)
#   observation: the observation of the agent (the state of the environment from the agent's perspective)
#


import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer

from models.mappo import Actor 
from models.mappo import Critic
from buffers import mappo_memory

import random
import numpy as np

class MAPPOAgent():

    def __init__(self, global_state_dim, state_dim, action_dim, lr=1e-3, gamma=0.99, gae_lambda=0.95, epsilon_clip=0.2, entropy_coeff=0.01, ppo_epochs=5, batch_size=32, device="cpu", **kwargs):

        self.gloabl_state_dim = global_state_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # TODO: separare lr per actor e critic, e aggiungere coefficiente per critic loss 
        self.a_lr = lr
        self.c_lr = lr 
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon_clip = epsilon_clip
        self.value_clip = epsilon_clip  # Using the same epsilon for value clipping as for policy clipping but the meaning is different: the value clipping is used to avoid large updates to the critic network (value function), while the policy clipping is used to avoid large updates to the actor networks (probabilities).
        self.entropy_coeff = entropy_coeff
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.critic_coeff = 0.5  # Hardcoded for now, but can be made a parameter in the future

        self.dec_actor1 = Actor(state_dim, action_dim).to(self.device)
        self.dec_actor2 = Actor(state_dim, action_dim).to(self.device)
        self.cen_critic = Critic(global_state_dim).to(self.device)

        self.actor_opt = optimizer.Adam(list(self.dec_actor1.parameters()) + list(self.dec_actor2.parameters()), lr=self.a_lr)
        self.criti_opt = optimizer.Adam(self.cen_critic.parameters(), lr=self.c_lr)

        self.memory = mappo_memory.Memory()

    def select_action(self, observation, agent_id):
        '''
        Select an action based on the current observation using the decentralized actor networks.
        
        1) it takes the logits from the actors nets: the real output of the networks which are not probabilities (their sum != 1 and they can be negative)
        2) take the logits and compute the action distribution (applying softmax to the logits: the exponential make every value > 0 and the normalization make the sum = 1)
        3) then sample an action from the distribution 
        4) lastly we compute the log prob of the selected action (log_prob = log(pi(a|s)) where pi(a|s)): since the prob is in [0,1] the log will be <= 0 and this is used to compute the loss function
        
        Args:
            observation: The current observation of the agent.
            agent_id: The ID of the agent (1 or 2).
        Returns:
            action (int): The selected action.
            log_prob (float): The log probability of the selected action.
        '''
        actor = self.dec_actor1 if agent_id == 1 else self.dec_actor2
        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = actor(observation)

            action_distribution = torch.distributions.Categorical(logits=logits)

            action = action_distribution.sample()
            
            log_prob = action_distribution.log_prob(action)

        return action.item(), log_prob.item(), 

    def store_sample(self, global_state, observation1, observation2, action1, action2, reward, log_prob1, log_prob2, done):
        '''
        Store the transition in the agent's memory.
        The storing operation is done with the native types of the arguments. 
        If necessary the conversion to torch.Tensor will be done in others parts of the code.
        '''
        self.memory.push(global_state, observation1, observation2, action1, action2, reward, log_prob1, log_prob2, done)

    def generalized_advantage_estimation(self, rewards, values, next_value, dones):
        '''
        Compute the advantages and returns using Generalized Advantage Estimation (GAE):
        The general formula for GAE is:
            delta_gae = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_t+l
        where:
            delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        is the TD error at time step t.
        It's value is indicates how much better or worse the action taken at time step t was compared to the expected value of the state s_t.
        
        The GAE can be estimated as: 
            A(s_t, a_t) = G_t - V(s_t)
        If G_{t} is > V(s_t) then the action was better than expected, if G_{t} is < V(s_t) then the action was worse than expected.

        If lambda = 0, the GAE reduces to the standard TD error, and if lambda = 1, it becomes the Monte Carlo estimate of the advantage.
        '''  
        discounted_returns = []        # G_{t}

        lambda_gae = 0
        for r_t, v_t, d_t in zip(reversed(rewards), reversed(values), reversed(dones)):
            # Compute the TD error (delta) for GAE: r + gamma * V(s_{t+1}) * (1 - d) - V(s_t)
            delta_t = r_t + self.gamma * next_value * (1 - d_t) - v_t

            # Update the GAE recursively (to match the formula above)
            lambda_gae = delta_t + self.gamma * self.gae_lambda * (1 - d_t) * lambda_gae

            # Compute the return G_t for the current time step
            G_t = lambda_gae + v_t
            discounted_returns.append(G_t)

            next_value = v_t  # Update next_value for the next iteration

        # Reverse to maintain the original order and convert to tensor
        discounted_returns.reverse()  
        discounted_returns = torch.tensor(discounted_returns, dtype=torch.float32).to(self.device)

        # A(s_t, a_t) = Q(s_t, a_t) - V(s_t) ~= G_t - V(s_t)
        advantages = discounted_returns - values

        # Normalize the advantages to have mean 0 and std 1 (to stabilize training of the actor networks)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, discounted_returns

    def entropy_decay(self, initial_entropy_coeff, episode, max_episodes):
        '''
        Decay the entropy coefficient over episodes to encourage exploration in the early stages of training and exploitation in the later stages.
        '''
        self.entropy_coeff = max(0.001, initial_entropy_coeff * (1 - episode / max_episodes))

    def ppo_step(self, global_states, observations1, observations2, actions1, actions2, log_probs1, log_probs2, advantages, discounted_returns, old_values):
        '''
        Perform the PPO optimization step for the actor and critic networks.
        '''
        tot_loss = 0.0
        for _ in range(self.ppo_epochs):
            random_indices = torch.randperm(len(self.memory)).to(self.device)


            for i in range(0, len(self.memory), self.batch_size):
                # Get the random batch indices for the current mini-batch
                batch_indices = random_indices[i:i+self.batch_size]

                logits1 = self.dec_actor1(observations1[batch_indices])
                logits2 = self.dec_actor2(observations2[batch_indices])

                distribution1 = torch.distributions.Categorical(logits=logits1)
                distribution2 = torch.distributions.Categorical(logits=logits2)

                new_log_probs1 = distribution1.log_prob(actions1[batch_indices])
                new_log_probs2 = distribution2.log_prob(actions2[batch_indices])

                r_t_theta1 = torch.exp(new_log_probs1 - log_probs1[batch_indices])
                r_t_theta2 = torch.exp(new_log_probs2 - log_probs2[batch_indices])

                surr1_1 = r_t_theta1 * advantages[batch_indices]
                surr2_1 = r_t_theta2 * advantages[batch_indices]

                surr1_2 = torch.clamp(r_t_theta1, 1.0 - self.epsilon_clip, 1.0 + self.epsilon_clip) * advantages[batch_indices]
                surr2_2 = torch.clamp(r_t_theta2, 1.0 - self.epsilon_clip, 1.0 + self.epsilon_clip) * advantages[batch_indices]

                actor_loss1 = -torch.min(surr1_1, surr1_2).mean()
                actor_loss2 = -torch.min(surr2_1, surr2_2).mean()

                # # critic loss
                # values = self.cen_critic(global_states[batch_indices]).squeeze(-1)
                # critic_loss = F.mse_loss(values, discounted_returns[batch_indices])

                # Critic loss with clipping
                values = self.cen_critic(global_states[batch_indices]).squeeze(-1)
                values_clipped = old_values[batch_indices] + torch.clamp(
                    values - old_values[batch_indices], -self.value_clip, self.value_clip
                )
                crit_loss_unclipped = (values - discounted_returns[batch_indices]) ** 2
                crit_loss_clipped = (values_clipped - discounted_returns[batch_indices]) ** 2
                critic_loss = torch.mean(torch.max(crit_loss_unclipped, crit_loss_clipped))

                # Entropy loss to encourage exploration
                entropy = (distribution1.entropy().mean() + distribution2.entropy().mean()) / 2.0

                ## TODO: Mettere variabile per critic_coefficient e non farlo hardocded
                loss = (actor_loss1 + actor_loss2) + 0.5 * critic_loss - self.entropy_coeff * entropy

                # optimize the networks
                # self.optimizers.zero_grad()
                # loss.backward()

                actor_loss = (actor_loss1 + actor_loss2) - self.entropy_coeff * entropy
                critic_loss = self.critic_coeff * critic_loss
                self.actor_opt.zero_grad()
                self.criti_opt.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(list(self.dec_actor1.parameters()), 1.0) 
                nn.utils.clip_grad_norm_(list(self.dec_actor2.parameters()), 1.0)
                nn.utils.clip_grad_norm_(self.cen_critic.parameters(), 1.0)
                self.criti_opt.step()
                self.actor_opt.step()
                
                # nn.utils.clip_grad_norm_(self.cen_critic.parameters(), 1.0)
                # nn.utils.clip_grad_norm_(self.dec_actor1.parameters(), 0.5)
                # nn.utils.clip_grad_norm_(self.dec_actor2.parameters(), 0.5)
                # self.optimizers.step() 

                # tot_loss += loss.item()

        self.memory.wipe()  

        # return tot_loss
        return None

    def optimize_model(self, next_global_state, done):
        '''
        Update the actor and critic networks using the stored transitions in the buffer.
        
        Args:
            next_global_state (torch.Tensor): The next global state after the last transition.
            done (bool): Whether the LAST episode has ended.
        '''

        ## Check if there are enough samples in the memory to perform an optimization step
        if self.memory.__len__() < self.batch_size:
            raise ValueError(f"Not enough samples in memory to perform optimization. Required: {self.batch_size}, Available: {self.memory.__len__()}")
            return None  

        ## Convert the values to tensors
        ## ATTENZIONE: maybe i'll have to use sqeueze(1) to add a dimension to the tensors, but for now let's see if it works without it
        transitions = self.memory.get_all()
        #next_global_state = torch.tensor(next_global_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        next_global_state = torch.tensor(next_global_state, dtype=torch.float32).to(self.device)
        global_states = torch.tensor([t.global_state for t in transitions], dtype=torch.float32).to(self.device)
        obs1_batch = torch.tensor([t.obs1 for t in transitions], dtype=torch.float32).to(self.device)
        obs2_batch = torch.tensor([t.obs2 for t in transitions], dtype=torch.float32).to(self.device)
        actions1 = torch.tensor([t.action1 for t in transitions], dtype=torch.int64).to(self.device)
        actions2 = torch.tensor([t.action2 for t in transitions], dtype=torch.int64).to(self.device)
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32).to(self.device)
        log_probs1 = torch.tensor([t.log_prob1 for t in transitions], dtype=torch.float32).to(self.device)
        log_probs2 = torch.tensor([t.log_prob2 for t in transitions], dtype=torch.float32).to(self.device)  
        dones = torch.tensor([t.done for t in transitions], dtype=torch.float32).to(self.device)

        ## Compute the value V(next_global_state) of the next global state using the centralized critic
        with torch.no_grad():

            # If the episode is ended (done), the value of the next state is 0
            if done:
                V_s_tplus1 = torch.tensor(0.0).to(self.device) 
            else:
                # V(s_{t+1})
                V_s_tplus1 = self.cen_critic(next_global_state).item()  

        ## Compute the advantages and returns using GAE (Generalized Advantage Estimation)
        with torch.no_grad():
            values = self.cen_critic(global_states).squeeze(-1)  # it's the value of the current state V(s_t) for each transition in the buffer

            gae_advantages, discounted_returns = self.generalized_advantage_estimation(rewards, values, V_s_tplus1, dones)
        old_values = values.clone().detach()  # Store the old values for the critic loss clipping


        ## Optimize the actor and critic networks using the PPO algorithm
        loss = self.ppo_step(global_states, obs1_batch, obs2_batch, actions1, actions2, log_probs1, log_probs2, gae_advantages, discounted_returns, old_values)

        return loss