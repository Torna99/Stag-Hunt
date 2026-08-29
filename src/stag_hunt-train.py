### Enrico Tornabene (0001191764)
###


import time
import gymnasium as gym
import gymnasium_stag_hunt
import torch

from agents.dqn_agent import DQNAgent
from utils import plot_training_results

import numpy as np
import random
import matplotlib.pyplot as plt

### ========
### Setting variables
### ========

GAME = "Hunt"  # or "Harvest" or "Escalation"

MAX_STEPS_PER_EPISODE = 200
MAX_EPISODES = 2000

# env configuration variables 
GRID_SIZE = 5
OBS_TYPE = "coords"  # or "image"
RENDER_MODE = None # None of "human"
FORAGE_QTA = 2
FORAGE_REWARD = 1
STAG_REWARD = 5
MAULING_PENALTY = -3

# Training hyperparam
BASTCH_SIZE = 64
GAMMA = 0.9
EPS_START = 1
EPS_END = 0.1
EPS_PLAT = 0.8
EPS_DECAY = (EPS_START - EPS_END) / (EPS_PLAT * MAX_EPISODES * MAX_STEPS_PER_EPISODE)
LR = 5e-3
REPLAY_BUFFER_SIZE = 10000

# Setting the accelerator if available
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    #"mps" if torch.backends.mps.is_available() else
    "cpu"
)
seed = 42
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)



### =======
### Main
### =======
if __name__ == "__main__":

    # Create the environment
    env = gym.make(
        "StagHunt-Hunt-v0",
        grid_size=(GRID_SIZE, GRID_SIZE),
        obs_type=OBS_TYPE,           
        forage_quantity=FORAGE_QTA,
        forage_reward=FORAGE_REWARD,
        stag_reward=STAG_REWARD,
        mauling_punishment=MAULING_PENALTY,
        max_episode_steps=MAX_STEPS_PER_EPISODE,
    )
    # ensure reproducibility in the env
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    # Create the agents 
    action_dim = env.action_space.n
    state, info = env.reset()
    state_dim = len(state[0])

    agent1 = DQNAgent(state_dim, action_dim, lr=LR, gamma=GAMMA, epsilon=EPS_START, epsilon_decay=EPS_DECAY, epsilon_min=EPS_END, buffer_size = REPLAY_BUFFER_SIZE, batch_size=BASTCH_SIZE, device=device)
    agent2 = DQNAgent(state_dim, action_dim, lr=LR, gamma=GAMMA, epsilon=EPS_START, epsilon_decay=EPS_DECAY, epsilon_min=EPS_END, buffer_size = REPLAY_BUFFER_SIZE, batch_size=BASTCH_SIZE, device=device)

    # Training loop
    total_rewrds = []
    total_stag_hunted = []
    total_maulings = []
    total_forage = []
    for episode in range(MAX_EPISODES):
        # initialize the environment and get the first state of the episode
        state, info = env.reset()

        state_agent1 = torch.tensor(state[0], dtype=torch.float32).unsqueeze(0).to(device)
        state_agent2 = torch.tensor(state[1], dtype=torch.float32).unsqueeze(0).to(device)

        total_reward_agent1 = 0
        total_reward_agent2 = 0
        tot_stag = 0
        tot_maul = 0
        tot_forage = 0
        done = False
        while not done:
            # Select actions 
            a1 = agent1.select_action(state_agent1)
            a2 = agent2.select_action(state_agent2)

            # Execute the actions and observe the next state and reward
            next_state, reward, terminated, truncated, info = env.step([a1.item(), a2.item()])
            done = terminated or truncated

            # screen render
            #env.render()

            # Analize the rewards for each agent 
            reward_agent1 = reward[0]
            reward_agent2 = reward[1]

            if reward_agent1 == STAG_REWARD and reward_agent2 == STAG_REWARD:
                tot_stag += 1
            tot_maul += (reward_agent1 == MAULING_PENALTY) + (reward_agent2 == MAULING_PENALTY)
            tot_forage += (reward_agent1 == FORAGE_REWARD) + (reward_agent2 == FORAGE_REWARD)

            # Convert the next state to tensors and store the transition in memory
            next_state_agent1 = torch.tensor(next_state[0], dtype=torch.float32).unsqueeze(0).to(device)
            next_state_agent2 = torch.tensor(next_state[1], dtype=torch.float32).unsqueeze(0).to(device)

            agent1.store_sample(state_agent1, a1, reward_agent1, next_state_agent1, done)
            agent2.store_sample(state_agent2, a2, reward_agent2, next_state_agent2, done)

            # Train step and update of the epsilon value
            agent1.optimize_model()
            agent2.optimize_model()

            agent1.epsilon_decay_step()
            agent2.epsilon_decay_step()

            # update the state for the next step and accumulate the rewards
            state_agent1 = next_state_agent1
            state_agent2 = next_state_agent2
            total_reward_agent1 += reward_agent1
            total_reward_agent2 += reward_agent2

        total_rewrds.append(total_reward_agent1 + total_reward_agent2)
        total_stag_hunted.append(tot_stag)
        total_maulings.append(tot_maul)
        total_forage.append(tot_forage)

        if episode % 50 == 0 or episode == 0:
            print(f"[EP {episode+1}/{MAX_EPISODES}] -> Total reward: {total_reward_agent1 + total_reward_agent2:.1f}\t| Stags: {tot_stag}\t| Maulings: {tot_maul}\t| Forage: {tot_forage}")



### =======
### Plotting the training results
### =======
plot_training_results(
    rewards=total_rewrds,
    stags=total_stag_hunted,
    maulings=total_maulings,
    forage=total_forage,
    window=50
)



