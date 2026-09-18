### Here we implement the training of 2 agents, based on the duelingDQN agent with different architecture wrt the standard DQN one
### and the Double DQN algorithm.

import time
import gymnasium as gym
import gymnasium_stag_hunt
import torch

from agents.duelingdqn_agent import DQNAgent
import utils

import numpy as np
import random
import matplotlib.pyplot as plt

from pathlib import Path

### ================================================================================================================
### Setting variables
### ================================================================================================================
GAME = "Harvest"

MAX_STEPS_PER_EPISODE = 200
MAX_EPISODES = 2000

# env configuration variables 
GRID_SIZE = 5
OBS_TYPE = "coords"  # or "image"
RENDER_MODE = None # None of "human"

MAX_PLANTS = 5
CHANCE_MATURE = 0.1
CHANCE_DIE = 0.1
YOUNG_REWARD = 1
MATURE_REWARD = 3


# Training hyperparam
BASTCH_SIZE = 64
GAMMA = 0.9
EPS_START = 1
EPS_END = 0.1
EPS_PLAT = 0.8
#EPS_DECAY = (EPS_START - EPS_END) / (EPS_PLAT * MAX_EPISODES * MAX_STEPS_PER_EPISODE)
EPS_DECAY = 0.995
LR = 5e-4
REPLAY_BUFFER_SIZE = 10000
C = 500 # number of steps after which the target network is updated with the policy network weights

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

# # setting variable to store the training results
# BASE_DIR = Path(__file__).resolve().parent.parent
# MODEL_NAME = "duelingDQN_2agents"
# MODEL_DIR = BASE_DIR / "saved_models"
# RESULTS_DIR = BASE_DIR / "results"

# MODEL_DIR.mkdir(parents=True, exist_ok=True)
# RESULTS_DIR.mkdir(parents=True, exist_ok=True)

### ================================================================================================================
### Main
### ================================================================================================================
if __name__ == "__main__":

    # Create the environment
    env = gym.make(
        "StagHunt-Harvest-v0",
        grid_size=(GRID_SIZE, GRID_SIZE),
        obs_type=OBS_TYPE,           
        flip_obs=True,
        max_episode_steps=MAX_STEPS_PER_EPISODE,
        max_plants=MAX_PLANTS,
        chance_to_mature=CHANCE_MATURE,
        chance_to_die=CHANCE_DIE,
        young_reward=YOUNG_REWARD,
        mature_reward=MATURE_REWARD
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
    total_rewards = []
    young_plants_harvested = []
    mature_plants_harvested = []
    
    tot_step = 0
    for episode in range(MAX_EPISODES):
        # initialize the environment and get the first state of the episode
        # state, info = env.reset(seed=seed + episode)  # Ensure different seed for each episode
        state, info = env.reset()  # Ensure different seed for each episode

        state_agent1 = torch.tensor(state[0], dtype=torch.float32).unsqueeze(0).to(device)
        state_agent2 = torch.tensor(state[1], dtype=torch.float32).unsqueeze(0).to(device)

        total_reward_agent1 = 0
        total_reward_agent2 = 0
        current_young_plants_harvested = 0
        current_mature_plants_harvested = 0

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

            if reward_agent1 == YOUNG_REWARD and reward_agent2 == YOUNG_REWARD:
                current_young_plants_harvested += 1
            elif reward_agent1 == MATURE_REWARD and reward_agent2 == MATURE_REWARD:
                current_mature_plants_harvested += 1

            # Convert the next state to tensors and store the transition in memory
            next_state_agent1 = torch.tensor(next_state[0], dtype=torch.float32).unsqueeze(0).to(device)
            next_state_agent2 = torch.tensor(next_state[1], dtype=torch.float32).unsqueeze(0).to(device)

            agent1.store_sample(state_agent1, a1, reward_agent1, next_state_agent1, done)
            agent2.store_sample(state_agent2, a2, reward_agent2, next_state_agent2, done)

            # Train step and update of the epsilon value (log the mean Q-value for each agent to track the overestimation problem)
            q_val1 = agent1.optimize_model()
            q_val2 = agent2.optimize_model()

            # update the target networks every C steps
            tot_step += 1
            if tot_step % C == 0 and tot_step != 0:
                agent1.update_target_network()
                agent2.update_target_network()

            # update the state for the next step and accumulate the rewards
            state_agent1 = next_state_agent1
            state_agent2 = next_state_agent2

            total_reward_agent1 += reward_agent1
            total_reward_agent2 += reward_agent2

        # Decay the epsilon value for both agents at the end of the episode with a multiplicative factor eps <- max(eps_min, eps * eps_decay)
        agent1.epsilon_decay_step()
        agent2.epsilon_decay_step()

        total_rewards.append(total_reward_agent1 + total_reward_agent2)
        young_plants_harvested.append(current_young_plants_harvested)
        mature_plants_harvested.append(current_mature_plants_harvested)

        if episode % 50 == 0 or episode == 0:
            print(f"[EP {episode+1}/{MAX_EPISODES}] -> Total reward: {total_reward_agent1 + total_reward_agent2:.1f}\t| Young Plants Harvested: {current_young_plants_harvested}\t| Mature Plants Harvested: {current_mature_plants_harvested}")


            
### ================================================================================================================
### Plotting the training results and saving 
### ================================================================================================================

# torch.save(agent1.policy_net.state_dict(), f"{MODEL_DIR}/{MODEL_NAME}_agent1.pth")
# torch.save(agent2.policy_net.state_dict(), f"{MODEL_DIR}/{MODEL_NAME}_agent2.pth")
# utils.save_train_metrics(rewards=total_rewards, stags=total_stag_hunted, maulings=total_maulings, forage=total_forage, q_values=total_q_values, filepath=f"{RESULTS_DIR}/{MODEL_NAME}_train_metrics.csv")

# utils.plot_training_results(
#     rewards=total_rewards,
#     stags=total_stag_hunted,
#     maulings=total_maulings,
#     forage=total_forage,
#     window=50
# )
# utils.plot_epsilon_trend(epsilon_values=epsilon_values, epsilon_min=EPS_END, epsilon_decay=EPS_DECAY)

import matplotlib.pyplot as plt

fig, axs = plt.subplots(3, 1, figsize=(10, 15))
# Plot total rewards
axs[0].plot(total_rewards, label='Total Rewards', color='blue')
axs[0].set_title('Total Rewards per Episode')
axs[0].set_xlabel('Episode')
axs[0].set_ylabel('Total Reward')
# Plot young plants harvested
axs[1].plot(young_plants_harvested, label='Young Plants Harvested', color='green')
axs[1].set_title('Young Plants Harvested per Episode')
axs[1].set_xlabel('Episode')
axs[1].set_ylabel('Young Plants Harvested')
# Plot mature plants harvested
axs[2].plot(mature_plants_harvested, label='Mature Plants Harvested', color='orange')
axs[2].set_title('Mature Plants Harvested per Episode')
axs[2].set_xlabel('Episode')
axs[2].set_ylabel('Mature Plants Harvested')

plt.tight_layout()
plt.show()




