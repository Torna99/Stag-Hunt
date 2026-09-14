### Here we implement the training of 2 agent, based on th advanced DQN agent with 2 different nets (target and policy) 
###

import time
import gymnasium as gym
import gymnasium_stag_hunt
import torch

from agents.a2c_agent import A2CAgent
import utils

import numpy as np
import random
import matplotlib.pyplot as plt

from pathlib import Path

### ================================================================================================================
### Setting variables
### ================================================================================================================
GAME = "Hunt"  # or "Harvest" or "Escalation"

MAX_STEPS_PER_EPISODE = 200
MAX_EPISODES = 2000

# env configuration variables 
GRID_SIZE = 10
OBS_TYPE = "coords"  # or "image"
RENDER_MODE = None # None of "human"
FORAGE_QTA = 2
FORAGE_REWARD = 1
STAG_REWARD = 5
MAULING_PENALTY = -1

# Training hyperparam
BASTCH_SIZE = 64
GAMMA = 0.9
LR = 5e-4
CRITIC_COEFF = 0.5
ENTROPY_COEFF = 0.1

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

# setting variable to store the training results
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_NAME = "a2c_2agents"
MODEL_DIR = BASE_DIR / "saved_models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

### ================================================================================================================
### Main
### ================================================================================================================
if __name__ == "__main__":

    # Create the environment
    env = gym.make(
        "StagHunt-Hunt-v0",
        grid_size=(GRID_SIZE, GRID_SIZE),
        obs_type=OBS_TYPE,           
        flip_obs=True,
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

    agent1 = A2CAgent(state_dim=state_dim, action_dim=action_dim, lr=LR, gamma=GAMMA, device=device, critic_coeff=CRITIC_COEFF, entropy_coeff=ENTROPY_COEFF)
    agent2 = A2CAgent(state_dim=state_dim, action_dim=action_dim, lr=LR, gamma=GAMMA, device=device, critic_coeff=CRITIC_COEFF, entropy_coeff=ENTROPY_COEFF)

    # Training loop
    total_rewrds = []
    total_stag_hunted = []
    total_maulings = []
    total_forage = []
    epsilon_values = [] 
    total_q_values = []

    tot_step = 0
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
            a1, log_prob1, v1 = agent1.select_action(state_agent1)
            a2, log_prob2, v2 = agent2.select_action(state_agent2)

            # Execute the actions and observe the next state and reward
            next_state, reward, terminated, truncated, info = env.step([a1.item(), a2.item()])
            done = terminated or truncated
            #rewards.append(reward)
            # states.append((state_agent1, state_agent2))
            # actions.append((a1, a2))

            agent1.store_transition(state_agent1, a1, reward[0], log_prob1, v1, done)
            agent2.store_transition(state_agent2, a2, reward[1], log_prob2, v2, done)

            # Analize the rewards for each agent 
            reward_agent1 = reward[0]
            reward_agent2 = reward[1]

            if reward_agent1 == STAG_REWARD and reward_agent2 == STAG_REWARD:
                tot_stag += 1
            tot_maul += (reward_agent1 == MAULING_PENALTY) + (reward_agent2 == MAULING_PENALTY)
            tot_forage += (reward_agent1 == FORAGE_REWARD) + (reward_agent2 == FORAGE_REWARD)

            total_reward_agent1 += reward_agent1
            total_reward_agent2 += reward_agent2

            state = next_state
            state_agent1 = torch.tensor(state[0], dtype=torch.float32).unsqueeze(0).to(device)
            state_agent2 = torch.tensor(state[1], dtype=torch.float32).unsqueeze(0).to(device)


        ## optimize the agents after the episode ends cause A2C is an on-policy algorithm
        state_agent1 = torch.tensor(state[0], dtype=torch.float32).unsqueeze(0).to(device)
        state_agent2 = torch.tensor(state[1], dtype=torch.float32).unsqueeze(0).to(device)
        agent1.optimize_model(state_agent1, done)
        agent2.optimize_model(state_agent2, done)

        total_rewrds.append(total_reward_agent1 + total_reward_agent2)
        total_stag_hunted.append(tot_stag)
        total_maulings.append(tot_maul)
        total_forage.append(tot_forage)

        # if episode % 50 == 0 or episode == 0:
        #     print(f"[EP {episode+1}/{MAX_EPISODES}] -> Total reward: {total_reward_agent1 + total_reward_agent2:.1f}\t| Stags: {tot_stag}\t| Maulings: {tot_maul}\t| Forage: {tot_forage}")
        print(f"[EP {episode+1}/{MAX_EPISODES}] -> Total reward: {total_reward_agent1 + total_reward_agent2:.1f}\t| Stags: {tot_stag}\t| Maulings: {tot_maul}\t| Forage: {tot_forage}")


            
### ================================================================================================================
### Plotting the training results and saving 
### ================================================================================================================

# torch.save(agent1.policy_net.state_dict(), f"{MODEL_DIR}/{MODEL_NAME}_agent1.pth")
# torch.save(agent2.policy_net.state_dict(), f"{MODEL_DIR}/{MODEL_NAME}_agent2.pth")
# utils.save_train_metrics(rewards=total_rewrds, stags=total_stag_hunted, maulings=total_maulings, q_values=total_q_values, forage=total_forage, filepath=f"{RESULTS_DIR}/{MODEL_NAME}_train_metrics.csv")

utils.plot_training_results(
    rewards=total_rewrds,
    stags=total_stag_hunted,
    maulings=total_maulings,
    forage=total_forage,
    window=50
)
#utils.plot_epsilon_trend(epsilon_values=epsilon_values, epsilon_min=EPS_END, epsilon_decay=EPS_DECAY)


