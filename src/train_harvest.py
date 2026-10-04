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
argument_parser.add_argument("--max_plants", type=int, default=5, help="Maximum number of plants in the environment")
argument_parser.add_argument("--chance_mature", type=float, default=0.1, help="Chance of a plant maturing")
argument_parser.add_argument("--chance_die", type=float, default=0.1, help="Chance of a plant dying")
argument_parser.add_argument("--young_reward", type=int, default=1, help="Reward for harvesting a young plant")
argument_parser.add_argument("--mature_reward", type=int, default=3, help="Reward for harvesting a mature plant")

argument_parser.add_argument("--discount_factor", type=float, default=0.99, help="Discount factor for future rewards")
argument_parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate for the optimizer")
argument_parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
argument_parser.add_argument("--epsilon_start", type=float, default=1.0, help="Starting value of epsilon for epsilon-greedy policy")
argument_parser.add_argument("--epsilon_decay", type=float, default=0.995, help="Decay rate of epsilon for epsilon-greedy policy")
argument_parser.add_argument("--epsilon_min", type=float, default=0.1, help="Minimum value of epsilon for epsilon-greedy policy")
argument_parser.add_argument("--target_update_freq", type=int, default=500, help="Frequency of target network updates for DQN-based algorithms")
argument_parser.add_argument("--gae_lambda", type=float, default=0.95, help="Lambda parameter for GAE")
argument_parser.add_argument("--ppo_epochs", type=int, default=5, help="Number of epochs for PPO updates")
argument_parser.add_argument("--critic_coeff", type=float, default=0.5, help="Coefficient for the critic loss in PPO")
argument_parser.add_argument("--entropy_coeff", type=float, default=0.01, help="Coefficient for the entropy loss in PPO")
argument_parser.add_argument("--epsilon_clip", type=float, default=0.2, help="Clipping parameter for PPOs")

argument_parser.add_argument("--save_dir", type=str, default="general", help="Subdirectory to save the results and models having root dir in results/")

args = argument_parser.parse_args()

MAX_EPISODES = args.max_episodes
MAX_STEPS_PER_EPISODE = args.max_steps_per_episode

OBS_TYPE = "coords"  # or "image"
GRID_SIZE = args.grid_size
MAX_PLANTS = args.max_plants
CHANCE_MATURE = args.chance_mature
CHANCE_DIE = args.chance_die
YOUNG_REWARD = args.young_reward
MATURE_REWARD = args.mature_reward

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
PPO_EPOCHS = args.ppo_epochs
CRITIC_COEFF = args.critic_coeff
EPSILON_CLIP = args.epsilon_clip
ENTROPY_COEFF = args.entropy_coeff

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


BASE_DIR = Path(__file__).resolve().parent.parent
GAME = "Harvest"
RESULTS_DIR = BASE_DIR / "results" / GAME / args.save_dir
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME = f"{ALGORITHM}_{NUM_AGENTS}agents"
#MODEL_DIR = BASE_DIR / "saved_models" / args.save_dir

###================================================================================================================
### Main training loop
###================================================================================================================

if __name__ == "__main__":

    print(f"Training 2 agents using {ALGORITHM} algorithm for {MAX_EPISODES} episodes with a maximum of {MAX_STEPS_PER_EPISODE} steps per episode.")

    ### Creation of the environment
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

    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    state, info = env.reset() 
    action_dim = env.action_space.n
    state_dim = len(state[0])
    global_state_dim = state_dim * NUM_AGENTS

    ### Creation of the agents
    if ALGORITHM in ["vanillaDQN", "standardDQN", "doubleDQN", "duelingDQN"]:
        agent1 = Agent(
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
    elif ALGORITHM == "mappo":
        mappo_agent = Agent(
                global_state_dim=global_state_dim,
                state_dim=state_dim,
                action_dim=action_dim,
                lr=LR,
                gamma=GAMMA,
                gae_lambda=GAE_LAMBDA,
                epsilon_clip=EPSILON_CLIP,
                entropy_coeff=ENTROPY_COEFF,
                ppo_epochs=PPO_EPOCHS,
                batch_size=BATCH_SIZE,
                critic_coeff=CRITIC_COEFF,
                device=device
        )
    elif ALGORITHM == "a2c":
        agent1 = Agent(
                state_dim=state_dim,
                action_dim=action_dim,
                lr=LR,
                gamma=GAMMA,
                entropy_coeff=ENTROPY_COEFF,
                critic_coeff=CRITIC_COEFF,
                device=device
        )
        agent2 = Agent(
                state_dim=state_dim,
                action_dim=action_dim,
                lr=LR,
                gamma=GAMMA,
                entropy_coeff=ENTROPY_COEFF,
                critic_coeff=CRITIC_COEFF,
                device=device
        )
    

    ### Training loop
    total_rewards = []
    total_young_plants_harvested = []
    total_mature_plants_harvested = []
    epsilon_values = []
    entropy_values = []
    total_q_values = []

    tot_step = 0
    for episode in range(MAX_EPISODES):
        state, info = env.reset()

        state_agent1 = state[0]
        state_agent2 = state[1]

        print("State agent 1:", state_agent1)
        print("State agent 2:", state_agent2)

        total_reward_agent1 = 0
        total_reward_agent2 = 0
        young_plants_harvested = 0
        mature_plants_harvested = 0
        episode_q_values = []

        done = False
        while not done:
            # 1) Select actions 
            if ALGORITHM in ["vanillaDQN", "standardDQN", "doubleDQN", "duelingDQN"]:
                a1 = agent1.select_action(state_agent1)
                a2 = agent2.select_action(state_agent2)
            elif ALGORITHM == "mappo":
                a1, log_prob1 = mappo_agent.select_action(state_agent1, agent_id=1)
                a2, log_prob2 = mappo_agent.select_action(state_agent2, agent_id=2)
            elif ALGORITHM == "a2c":
                a1, log_prob1, v1 = agent1.select_action(state_agent1)
                a2, log_prob2, v2 = agent2.select_action(state_agent2)

            # 2) Execute the actions and observe the next state and reward
            next_state, reward, terminated, truncated, info = env.step([a1, a2])
            done = terminated or truncated

            # 3) Analize the rewards for each agent
            reward_agent1 = reward[0]
            reward_agent2 = reward[1]
            young_plants_harvested += (reward_agent1 == YOUNG_REWARD) + (reward_agent2 == YOUNG_REWARD)
            mature_plants_harvested += (reward_agent1 == MATURE_REWARD) + (reward_agent2 == MATURE_REWARD)
            
            # 4) Store the transition in memory
            next_state_agent1 = next_state[0]
            next_state_agent2 = next_state[1]

            if ALGORITHM in ["vanillaDQN", "standardDQN", "doubleDQN", "duelingDQN"]:
                agent1.store_sample(state_agent1, a1, reward_agent1, next_state_agent1, done)
                agent2.store_sample(state_agent2, a2, reward_agent2, next_state_agent2, done)
            elif ALGORITHM == "mappo":
                global_state = np.concatenate([state_agent1, state_agent2], axis=-1)
                step_reward = float(reward_agent1 + reward_agent2)
                mappo_agent.store_sample(
                    global_state=global_state,
                    observation1=state_agent1,
                    observation2=state_agent2,
                    action1=a1,
                    action2=a2,
                    reward=step_reward,
                    log_prob1=log_prob1,
                    log_prob2=log_prob2,
                    done=done
                )
            elif ALGORITHM == "a2c":
                agent1.store_transition(state_agent1, a1, reward_agent1, log_prob1, v1, done)
                agent2.store_transition(state_agent2, a2, reward_agent2, log_prob2, v2, done)


            # 5.1) Train step and update of the epsilon value (log the mean Q-value for each agent to track the overestimation problem)
            if ALGORITHM in ["vanillaDQN", "standardDQN", "doubleDQN", "duelingDQN"]:
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

        # 5.2) Optimize the model at the end of the episode for A2C and MAPPO
        if ALGORITHM == "mappo":
            if episode < 50: 
                rollout = 1
            elif episode < 500:
                rollout = 5
            else:
                rollout = 10
            if (episode + 1) % rollout == 0:
                next_global_state = np.concatenate([state_agent1, state_agent2], axis=-1)
                mappo_agent.optimize_model(next_global_state, done)
                mappo_agent.entropy_decay(ENTROPY_COEFF, episode, MAX_EPISODES)
        elif ALGORITHM == "a2c":
            agent1.optimize_model(state_agent1, done)
            agent2.optimize_model(state_agent2, done)

        # 7) Decay the epsilon value for both agents at the end of the episode with a multiplicative factor eps <- max(eps_min, eps * eps_decay)
        if ALGORITHM in ["vanillaDQN", "standardDQN", "doubleDQN", "duelingDQN"]:
            agent1.epsilon_decay_step()
            agent2.epsilon_decay_step()

        # 8) Log the metrics
        if ALGORITHM in ["vanillaDQN", "standardDQN", "doubleDQN", "duelingDQN"]:
            epsilon_values.append(agent1.epsilon)
            total_q_values.append(np.mean(episode_q_values))
        elif ALGORITHM == "mappo":
            entropy_values.append(mappo_agent.entropy_coeff)
            total_q_values.append(0.0)  # Placeholder for MAPPO, as it doesn't use Q-values
        elif ALGORITHM == "a2c":
            entropy_values.append(agent1.entropy_coeff)
            total_q_values.append(0.0)  # Placeholder for A2C, as it doesn't use Q-values
        total_rewards.append(total_reward_agent1 + total_reward_agent2)
        total_young_plants_harvested.append(young_plants_harvested)
        total_mature_plants_harvested.append(mature_plants_harvested)

        if episode % 50 == 0 or episode == 0:
            if ALGORITHM in ["vanillaDQN", "standardDQN", "doubleDQN", "duelingDQN"]:
                print(f"[EP {episode+1}/{MAX_EPISODES}] -> Total reward: {total_reward_agent1 + total_reward_agent2:.1f}\t| Young Plants Harvested: {young_plants_harvested}\t| Mature Plants Harvested: {mature_plants_harvested}\t| Mean Q-value: {np.mean(episode_q_values):.4f}\t| Epsilon: {agent1.get_epsilon():.4f}")
            elif ALGORITHM == "mappo":
                print(f"[EP {episode+1}/{MAX_EPISODES}] -> Total reward: {total_reward_agent1 + total_reward_agent2:.1f}\t| Young Plants Harvested: {young_plants_harvested}\t| Mature Plants Harvested: {mature_plants_harvested}\t| Entropy Coeff: {mappo_agent.entropy_coeff:.3f}\t| Actor LR: {mappo_agent.a_lr:.6f}\t| Critic LR: {mappo_agent.c_lr:.6f}")
            elif ALGORITHM == "a2c":
                print(f"[EP {episode+1}/{MAX_EPISODES}] -> Total reward: {total_reward_agent1 + total_reward_agent2:.1f}\t| Young Plants Harvested: {young_plants_harvested}\t| Mature Plants Harvested: {mature_plants_harvested} | Entropy Coeff: {agent1.entropy_coeff:.3f}")
    ### Save the results and models
    utils.save_harvest_train_metrics(
        rewards=total_rewards,
        young_plants_harvested=total_young_plants_harvested,
        mature_plants_harvested=total_mature_plants_harvested,
        filepath=f"{RESULTS_DIR}/{MODEL_NAME}_harvest_train_metrics.csv"
    )

    utils.plot_harvest_training_results(
        rewards=total_rewards,
        young_plants_harvested=total_young_plants_harvested,
        mature_plants_harvested=total_mature_plants_harvested,
        window=50
    )

    if ALGORITHM in ["vanillaDQN", "standardDQN", "doubleDQN", "duelingDQN"]:
        utils.plot_epsilon_trend(epsilon_values=epsilon_values, epsilon_min=EPS_END, epsilon_decay=EPS_DECAY)
    elif ALGORITHM in ["mappo", "a2c"]:
        utils.plot_entropy_trend(entropy_values=entropy_values, entropy_min=0.0, entropy_decay=ENTROPY_COEFF)