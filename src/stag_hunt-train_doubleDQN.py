### Here we implement the training of 2 agent, based on th doubleDQN agent with 2 different nets (target and policy) 
###

import time
import gymnasium as gym
import gymnasium_stag_hunt
import torch

from agents.doubledqn_agent import DQNAgent
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
GRID_SIZE = 7
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

# setting variable to store the training results
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_NAME = "doubleDQN_2agents"
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

    agent1 = DQNAgent(state_dim, action_dim, lr=LR, gamma=GAMMA, epsilon=EPS_START, epsilon_decay=EPS_DECAY, epsilon_min=EPS_END, buffer_size = REPLAY_BUFFER_SIZE, batch_size=BASTCH_SIZE, device=device)
    agent2 = DQNAgent(state_dim, action_dim, lr=LR, gamma=GAMMA, epsilon=EPS_START, epsilon_decay=EPS_DECAY, epsilon_min=EPS_END, buffer_size = REPLAY_BUFFER_SIZE, batch_size=BASTCH_SIZE, device=device)

    # Training loop
    total_rewrds = []
    total_stag_hunted = []
    total_maulings = []
    total_forage = []
    total_q_values = []
    epsilon_values = [] 

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
        episode_q_values = []

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

            # Train step and update of the epsilon value (log the mean Q-value for each agent to track the overestimation problem)
            q_val1 = agent1.optimize_model()
            q_val2 = agent2.optimize_model()

            if q_val1 is not None and q_val2 is not None:
                episode_q_values.append((q_val1 + q_val2) / 2.0)

            # agent1.epsilon_decay_step()
            # agent2.epsilon_decay_step()

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

        epsilon_values.append(agent1.get_epsilon())  # Assuming both agents have the same epsilon decay
        total_rewrds.append(total_reward_agent1 + total_reward_agent2)
        total_stag_hunted.append(tot_stag)
        total_maulings.append(tot_maul)
        total_forage.append(tot_forage)
        total_q_values.append(np.mean(episode_q_values))

        if episode % 50 == 0 or episode == 0:
            print(f"[EP {episode+1}/{MAX_EPISODES}] -> Total reward: {total_reward_agent1 + total_reward_agent2:.1f}\t| Stags: {tot_stag}\t| Maulings: {tot_maul}\t| Forage: {tot_forage}")


            
### ================================================================================================================
### Plotting the training results and saving 
### ================================================================================================================

torch.save(agent1.policy_net.state_dict(), f"{MODEL_DIR}/{MODEL_NAME}_agent1.pth")
torch.save(agent2.policy_net.state_dict(), f"{MODEL_DIR}/{MODEL_NAME}_agent2.pth")
utils.save_train_metrics(rewards=total_rewrds, stags=total_stag_hunted, maulings=total_maulings, forage=total_forage, q_values=total_q_values, filepath=f"{RESULTS_DIR}/{MODEL_NAME}_train_metrics.csv")

utils.plot_training_results(
    rewards=total_rewrds,
    stags=total_stag_hunted,
    maulings=total_maulings,
    forage=total_forage,
    window=50
)
utils.plot_epsilon_trend(epsilon_values=epsilon_values, epsilon_min=EPS_END, epsilon_decay=EPS_DECAY)




