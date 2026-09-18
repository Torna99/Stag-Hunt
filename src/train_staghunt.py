### TODO:
### - aggiungere salvataggio dei modelli e dei risultati
### - aggiungere mappo e a2c

import time
import gymnasium as gym
import gymnasium_stag_hunt
import torch

from agents import *
import utils 

import numpy as np
import random
import matplotlib.pyplot as plt

from pathlib import Path
import argparse

### ================================================================================================================
### Setting device and seed
### ================================================================================================================
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

seed = 42
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


### ================================================================================================================
### Reafing the arguments from the command line and setting the variables
### ================================================================================================================
argument_parser = argparse.ArgumentParser(description="Training of 2 agents using for stag hunt game using different algorithms")
argument_parser.add_argument("--algorithm", type=str, default="duelingDQN", choices=["vanillaDQN", "standardDQN", "doubleDQN", "duelingDQN", "a2c", "mappo"], help="Algorithm used for training the agents")
argument_parser.add_argument("--replay_buffer_size", type=int, default=10000, help="Size of the replay buffer")

argument_parser.add_argument("--max_episodes", type=int, default=2000, help="Maximum number of episodes for training")
argument_parser.add_argument("--max_steps_per_episode", type=int, default=200, help="Maximum number of steps per episode")

argument_parser.add_argument("--grid_size", type=int, default=5, help="Size of the grid for the environment")
argument_parser.add_argument("--forage_qta", type=int, default=2, help="Quantity of forage available in the environment")
argument_parser.add_argument("--forage_reward", type=int, default=1, help="Reward for foraging")
argument_parser.add_argument("--stag_reward", type=int, default=5, help="Reward for hunting stag")
argument_parser.add_argument("--mauling_penalty", type=int, default=-3, help="Penalty for mauling")

argument_parser.add_argument("--discount_factor", type=float, default=0.99, help="Discount factor for future rewards")
argument_parser.add_argument("--learning_rate", type=float, default=5e-3, help="Learning rate for the optimizer")
argument_parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
argument_parser.add_argument("--epsilon_start", type=float, default=1.0, help="Starting value of epsilon for epsilon-greedy policy")
argument_parser.add_argument("--epsilon_decay", type=float, default=0.995, help="Decay rate of epsilon for epsilon-greedy policy")
argument_parser.add_argument("--epsilon_min", type=float, default=0.1, help="Minimum value of epsilon for epsilon-greedy policy")
argument_parser.add_argument("--target_update_freq", type=int, default=500, help="Frequency of target network updates for DQN-based algorithms")
argument_parser.add_argument("--gae_lambda", type=float, default=0.95, help="Lambda parameter for GAE")

args = argument_parser.parse_args()

MAX_EPISODES = args.max_episodes
MAX_STEPS_PER_EPISODE = args.max_steps_per_episode

OBS_TYPE = "coords"  # or "image"
GRID_SIZE = args.grid_size
FORAGE_QTA = args.forage_qta
FORAGE_REWARD = args.forage_reward
STAG_REWARD = args.stag_reward
MAULING_PENALTY = args.mauling_penalty

ALGORITHM = args.algorithm
REPLAY_BUFFER_SIZE = args.replay_buffer_size
GAMMA = args.discount_factor
LR = args.learning_rate
BATCH_SIZE = args.batch_size
EPS_START = args.epsilon_start
EPS_DECAY = args.epsilon_decay
EPS_END = args.epsilon_min
C = args.target_update_freq

GAE_LAMBDA = args.gae_lambda

NUM_AGENTS = 2

if ALGORITHM == "vanillaDQN":
    from agents.vanilladqn_agent import DQNAgent as Agent
elif ALGORITHM == "standardDQN":
    from agents.standarddqn_agent import DQNAgent as Agent
elif ALGORITHM == "doubleDQN":
    from agents.doubledqn_agent import DQNAgent as Agent
elif ALGORITHM == "duelingDQN":
    from agents.duelingdqn_agent import DQNAgent as Agent
elif ALGORITHM == "a2c":
    from agents.a2c_agent import A2CAgent as Agent
elif ALGORITHM == "mappo":
    from agents.mappo_agent import MAPPOAgent as Agent

## TODO: parte di salvataggio risultati 

###================================================================================================================
### Main training loop
###================================================================================================================

if __name__ == "__main__":

    print(f"Training 2 agents using {ALGORITHM} algorithm for {MAX_EPISODES} episodes with a maximum of {MAX_STEPS_PER_EPISODE} steps per episode.")

    ### Creation of the environment
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

    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    state, info = env.reset() 
    action_dim = env.action_space.n
    state_dim = len(state[0])
    global_state_dim = state_dim * NUM_AGENTS

    ### Creation of the agents
    agent1 = Agent(
        global_state_dim = global_state_dim,
        state_dim = state_dim,
        action_dim = action_dim,
        lr = LR,
        gamma = GAMMA,
        epsilon = EPS_START,
        epsilon_decay = EPS_DECAY,
        epsilon_min = EPS_END,
        buffer_size = REPLAY_BUFFER_SIZE,
        batch_size = BATCH_SIZE,
        device = device
    )
    agent2 = Agent(
        global_state_dim = global_state_dim,
        state_dim = state_dim,
        action_dim = action_dim,
        lr = LR,
        gamma = GAMMA,
        epsilon = EPS_START,
        epsilon_decay = EPS_DECAY,
        epsilon_min = EPS_END,
        buffer_size = REPLAY_BUFFER_SIZE,
        batch_size = BATCH_SIZE,
        device = device
    )

    ### Training loop
    total_rewards = []
    total_stag_hunted = []
    total_maulings = []
    total_forage = []
    total_q_values = []
    epsilon_values = []

    tot_step = 0
    for episode in range(MAX_EPISODES):
        state, info = env.reset()

        # state_agent1 = torch.tensor(state[0], dtype=torch.float32).unsqueeze(0).to(device)
        # state_agent2 = torch.tensor(state[1], dtype=torch.float32).unsqueeze(0).to(device)
        state_agent1 = state[0]
        state_agent2 = state[1]

        total_reward_agent1 = 0
        total_reward_agent2 = 0
        tot_stag = 0
        tot_maul = 0
        tot_forage = 0
        episode_q_values = []

        done = False
        while not done:
            # 1) Select actions 
            a1 = agent1.select_action(state_agent1)
            a2 = agent2.select_action(state_agent2)

            # 2) Execute the actions and observe the next state and reward
            next_state, reward, terminated, truncated, info = env.step([a1, a2])
            done = terminated or truncated

            # 3) Analize the rewards for each agent
            reward_agent1 = reward[0]
            reward_agent2 = reward[1]
            if reward_agent1 == STAG_REWARD and reward_agent2 == STAG_REWARD:
                tot_stag += 1
            tot_maul += (reward_agent1 == MAULING_PENALTY) + (reward_agent2 == MAULING_PENALTY)
            tot_forage += (reward_agent1 == FORAGE_REWARD) + (reward_agent2 == FORAGE_REWARD)  

            # 4) Convert the next state to tensors and store the transition in memory
            # next_state_agent1 = torch.tensor(next_state[0], dtype=torch.float32).unsqueeze(0).to(device)
            # next_state_agent2 = torch.tensor(next_state[1], dtype=torch.float32).unsqueeze(0).to(device)
            next_state_agent1 = next_state[0]
            next_state_agent2 = next_state[1]

            agent1.store_sample(state_agent1, a1, reward_agent1, next_state_agent1, done)
            agent2.store_sample(state_agent2, a2, reward_agent2, next_state_agent2, done)

            # 5) Train step and update of the epsilon value (log the mean Q-value for each agent to track the overestimation problem)
            q_val1 = agent1.optimize_model()
            q_val2 = agent2.optimize_model()
            
            if q_val1 is not None and q_val2 is not None:
                episode_q_values.append((q_val1 + q_val2) / 2.0)

            tot_step += 1
            if ALGORITHM in ["standardDQN","doubleDQN", "duelingDQN"] and tot_step % C == 0 and tot_step > 0:
                agent1.update_target_network()
                agent2.update_target_network()

            # 6) update the state for the next step and accumulate the rewards
            state_agent1 = next_state_agent1
            state_agent2 = next_state_agent2

            total_reward_agent1 += reward_agent1
            total_reward_agent2 += reward_agent2

        # 7) Decay the epsilon value for both agents at the end of the episode with a multiplicative factor eps <- max(eps_min, eps * eps_decay)
        agent1.epsilon_decay_step()
        agent2.epsilon_decay_step()
 
        epsilon_values.append(agent1.get_epsilon())  # Assuming both agents have the same epsilon decay
        total_rewards.append(total_reward_agent1 + total_reward_agent2)
        total_stag_hunted.append(tot_stag)
        total_maulings.append(tot_maul)
        total_forage.append(tot_forage)
        total_q_values.append(np.mean(episode_q_values))
 
        if episode % 50 == 0 or episode == 0:
             print(f"[EP {episode+1}/{MAX_EPISODES}] -> Total reward: {total_reward_agent1 + total_reward_agent2:.1f}\t| Stags: {tot_stag}\t| Maulings: {tot_maul}\t| Forage: {tot_forage} | Mean Q-value: {np.mean(episode_q_values):.4f}\t| Epsilon: {agent1.get_epsilon():.4f}")
            