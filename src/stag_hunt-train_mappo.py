### ================================================================================================================
### Training of 2 agents using MAPPO (CTDE: Decentralized Actors + Centralized Critic)
### ================================================================================================================

import time
import gymnasium as gym
import gymnasium_stag_hunt
import torch

from agents.mappo_agent import MAPPOAgent
import utils

import numpy as np
import random
import matplotlib.pyplot as plt

from pathlib import Path

### ================================================================================================================
### Setting variables
### ================================================================================================================
GAME = "Hunt"

MAX_STEPS_PER_EPISODE = 200
MAX_EPISODES = 2000

# Env configuration variables 
GRID_SIZE = 10
OBS_TYPE = "coords"
RENDER_MODE = None
FORAGE_QTA = 2
FORAGE_REWARD = 1
STAG_REWARD = 10
MAULING_PENALTY = -1
# STAG_REWARD = 5
# MAULING_PENALTY = -3

# Training hyperparameters
NUM_AGENTS = 2
BATCH_SIZE = 32 #64
GAMMA = 0.99
LR = 3e-4
CLIP_EPS = 0.2
PPO_EPOCHS = 4
CRITIC_COEFF = 0.5
ENTROPY_COEFF = 0.01
GAE_LAMBDA = 0.95

# Setting accelerator
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_NAME = "mappo_2agents"
MODEL_DIR = BASE_DIR / "saved_models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

### ================================================================================================================
### Helper: Global State Construction
### ================================================================================================================
def get_global_state(obs_n):
    """
    Concatena le osservazioni locali di tutti gli agenti per il Critic Centralizzato.
    """
    return np.concatenate(obs_n, axis=-1)

### ================================================================================================================
### Main Loop
### ================================================================================================================
if __name__ == "__main__":

    # Create environment
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

    action_dim = env.action_space.n
    state, info = env.reset()
    
    obs_dim = len(state[0])
    global_state_dim = obs_dim * NUM_AGENTS

    # Inizializza il trainer MAPPO (gestisce gli N attori e il critic centralizzato)
    mappo = MAPPOAgent(
        global_state_dim=global_state_dim,
        state_dim=obs_dim,
        action_dim=action_dim,
        lr=LR,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        epsilon_clip=CLIP_EPS,
        entropy_coeff=ENTROPY_COEFF,
        ppo_epochs=PPO_EPOCHS,
        batch_size=BATCH_SIZE,
        #critic_coeff=CRITIC_COEFF,
        device=device
    )

    # Training metrics
    total_rewrds = []
    total_stag_hunted = []
    total_maulings = []
    total_forage = []

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

        done = False
        while not done:
            # Stato globale concatenato
            #global_state = torch.cat([state_agent1, state_agent2], dim=-1)
            global_state = get_global_state([state_agent1, state_agent2])

            # Selezione azioni dei due attori
            # a1, a2, log_prob1, log_prob2 = mappo.select_action(state_agent1, state_agent2)
            a1, log_prob1 = mappo.select_action(state_agent1, agent_id=1)
            a2, log_prob2 = mappo.select_action(state_agent2, agent_id=2)

            next_state, reward, terminated, truncated, info = env.step([a1, a2])
            done = terminated or truncated

            reward_agent1 = reward[0]
            reward_agent2 = reward[1]
            step_reward = float(reward_agent1 + reward_agent2)

            mappo.store_sample(
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

            if reward_agent1 == STAG_REWARD and reward_agent2 == STAG_REWARD:
                tot_stag += 1
            tot_maul += (reward_agent1 == MAULING_PENALTY) + (reward_agent2 == MAULING_PENALTY)
            tot_forage += (reward_agent1 == FORAGE_REWARD) + (reward_agent2 == FORAGE_REWARD)

            total_reward_agent1 += reward_agent1
            total_reward_agent2 += reward_agent2

            state = next_state
            state_agent1 = state[0]
            state_agent2 = state[1]

        # Ottimizzazione a fine episodio 
        if episode < 50: 
            rollout = 1
        elif episode < 500:
            rollout = 5
        else:
            rollout = 10
        if (episode + 1) % rollout == 0:
            next_global_state = get_global_state([state_agent1, state_agent2])
            mappo.optimize_model(next_global_state, done)
            mappo.entropy_decay(ENTROPY_COEFF, episode, MAX_EPISODES)
            ## PROVA
            # mappo.a_lr = mappo.a_lr * 0.999
            # mappo.c_lr = mappo.c_lr * 0.999

        ep_total_reward = total_reward_agent1 + total_reward_agent2
        total_rewrds.append(ep_total_reward)
        total_stag_hunted.append(tot_stag)
        total_maulings.append(tot_maul)
        total_forage.append(tot_forage)

        if (episode + 1) % 50 == 0:
            print(f"[EP {episode+1}/{MAX_EPISODES}] -> Total reward: {ep_total_reward:.1f}\t| Stags: {tot_stag}\t | Maulings: {tot_maul}\t| Forage: {tot_forage}\t| Entropy Coeff: {mappo.entropy_coeff:.3f}\t| Actor LR: {mappo.a_lr:.6f}\t| Critic LR: {mappo.c_lr:.6f}")

### ================================================================================================================
### Plotting the training results and saving 
### ================================================================================================================
# torch.save(mappo.dec_actor1.state_dict(), MODEL_DIR / f"{MODEL_NAME}_actor1.pth")
# torch.save(mappo.dec_actor2.state_dict(), MODEL_DIR / f"{MODEL_NAME}_actor2.pth")
# torch.save(mappo.cen_critic.state_dict(), MODEL_DIR / f"{MODEL_NAME}_critic.pth")

utils.save_train_metrics(
    rewards=total_rewrds,
    stags=total_stag_hunted,
    maulings=total_maulings,
    q_values=[0.0] * len(total_rewrds),
    forage=total_forage,
    filepath=RESULTS_DIR / f"{MODEL_NAME}_train_metrics.csv"
)

utils.plot_training_results(
    rewards=total_rewrds,
    stags=total_stag_hunted,
    maulings=total_maulings,
    forage=total_forage,
    window=50
)