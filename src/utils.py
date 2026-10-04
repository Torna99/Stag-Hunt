import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec

def moving_avg(data, w):
    if len(data) < w:
        return data
    return np.convolve(data, np.ones(w) / w, mode='valid')

def plot_training_results(rewards, stags, maulings, forage, window=50):

    episodes = np.arange(1, len(rewards) + 1)
    ma_episodes = np.arange(window, len(rewards) + 1) if len(rewards) >= window else episodes

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Multi-Agent: Stag Hunt Training Results", fontsize=14, fontweight='bold')

    ax1.plot(episodes, rewards, alpha=0.25, color="steelblue", label="Raw")
    ax1.plot(ma_episodes, moving_avg(rewards, window), color="navy", linewidth=2, label=f"MA {window}")
    ax1.set_title("Total Reward")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(ma_episodes, moving_avg(stags, window), color="forestgreen", linewidth=2, label="Stags (Coop)")
    ax2.plot(ma_episodes, moving_avg(forage, window), color="darkorange", linewidth=2, label="Forage (Safe)")
    ax2.plot(ma_episodes, moving_avg(maulings, window), color="crimson", linewidth=2, label="Maulings (Risk)")
    ax2.set_title(f"Game Dynamics (MA {window})")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Events / Episode")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.show()

def plot_escalation_training_results(rewards, max_streak, cooperation_steps, window=50):

    episodes = np.arange(1, len(rewards) + 1)
    ma_episodes = np.arange(window, len(rewards) + 1) if len(rewards) >= window else episodes

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Multi-Agent: Escalation Training Results", fontsize=14, fontweight='bold')

    # Tot reward
    ax1.plot(episodes, rewards, alpha=0.25, color="steelblue", label="Raw")
    ax1.plot(ma_episodes, moving_avg(rewards, window), color="navy", linewidth=2, label=f"MA {window}")
    ax1.set_title("Total Reward")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(ma_episodes, moving_avg(max_streak, window), color="lightblue", linewidth=2, label="Max Streak")
    ax2.plot(ma_episodes, moving_avg(cooperation_steps, window), color="lightgreen", linewidth=2, label="Cooperation Steps")
    ax2.set_title(f"Game Dynamics (MA {window})")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Events / Episode")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.show()

def plot_harvest_training_results(rewards, young_plants_harvested, mature_plants_harvested, window=50):

    episodes = np.arange(1, len(rewards) + 1)
    ma_episodes = np.arange(window, len(rewards) + 1) if len(rewards) >= window else episodes

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Multi-Agent: Harvest Training Results", fontsize=14, fontweight='bold')

    ax1.plot(episodes, rewards, alpha=0.25, color="steelblue", label="Raw")
    ax1.plot(ma_episodes, moving_avg(rewards, window), color="navy", linewidth=2, label=f"MA {window}")
    ax1.set_title("Total Reward")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(ma_episodes, moving_avg(young_plants_harvested, window), color="lightblue", linewidth=2, label="Young Plants")
    ax2.plot(ma_episodes, moving_avg(mature_plants_harvested, window), color="lightgreen", linewidth=2, label="Mature Plants")
    ax2.set_title(f"Game Dynamics (MA {window})")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Plants / Episode")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.show()

def plot_epsilon_trend(epsilon_values, epsilon_min, epsilon_decay):

    episodes = np.arange(1, len(epsilon_values) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, epsilon_values, color="darkblue", linewidth=2, label="Epsilon Value")
    plt.axhline(y=epsilon_min, color="red", linestyle="--", label=f"Epsilon Min ({epsilon_min})")
    plt.title(f"Epsilon Decay Trend (Decay Rate: {epsilon_decay})", fontsize=14, fontweight='bold')
    plt.xlabel("Episode")
    plt.ylabel("Epsilon Value")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_entropy_trend(entropy_values, entropy_min, entropy_decay):

    episodes = np.arange(1, len(entropy_values) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, entropy_values, color="darkblue", linewidth=2, label="Entropy Value")
    plt.axhline(y=entropy_min, color="red", linestyle="--", label=f"Entropy Min ({entropy_min})")
    plt.title(f"Entropy Decay Trend (Decay Rate: {entropy_decay})", fontsize=14, fontweight='bold')
    plt.xlabel("Episode")
    plt.ylabel("Entropy Value")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_q_value_trend(q_values, window=50):
    episodes = np.arange(1, len(q_values) + 1)
    ma_episodes = np.arange(window, len(q_values) + 1) if len(q_values) >= window else episodes

    plt.figure(figsize=(10, 5))
    plt.plot(ma_episodes, moving_avg(q_values, window), color="purple", linewidth=2, label=f"MA {window}")
    plt.title(f"Mean Q-Value Trend (MA {window})", fontsize=14, fontweight='bold')
    plt.xlabel("Episode")
    plt.ylabel("Mean Q-Value")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def save_train_metrics(rewards, stags, maulings, forage, q_values, filepath="../results/train_metrics.csv"):
    data = {
        "Episode": np.arange(1, len(rewards) + 1),
        "totRewards": rewards,
        "Stags": stags,
        "Forage": forage,
        "Maulings": maulings,
        "Q_values": q_values
    }
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"Training metrics saved to {filepath}")

def save_escalation_train_metrics(rewards, max_streak, cooperation_steps, filepath="../results/escalation_train_metrics.csv"):
    data = {
        "Episode": np.arange(1, len(rewards) + 1),
        "totRewards": rewards,
        "MaxStreak": max_streak,
        "CooperationSteps": cooperation_steps
    }
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"Training metrics saved to {filepath}")

def save_harvest_train_metrics(rewards, young_plants_harvested, mature_plants_harvested, filepath="../results/harvest_train_metrics.csv"):
    data = {
        "Episode": np.arange(1, len(rewards) + 1),
        "totRewards": rewards,
        "YoungPlantsHarvested": young_plants_harvested,
        "MaturePlantsHarvested": mature_plants_harvested
    }
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"Training metrics saved to {filepath}")

def compare_experiments_stag_hunt(file_dict, title, window=50):
    '''
    Compare the training metrics of different experiments.
    file_dict: 
        es. {
                "Standard DQN": "results/standard_dqn.csv", 
                "Target Net DQN": "results/target_net.csv"
            }
    '''
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=15, fontweight="bold")
    axes = axes.flatten()
    
    ax_rewards, ax_stags, ax_maulings, ax_forage = axes

    for label, filepath in file_dict.items():
        df = pd.read_csv(filepath)
        
        rewards = df["totRewards"].values
        stags = df["Stags"].values
        maulings = df["Maulings"].values
        forage = df["Forage"].values

        ma_episodes = np.arange(window, len(rewards) + 1) if len(rewards) >= window else np.arange(1, len(rewards) + 1)
        ax_rewards.plot(ma_episodes, moving_avg(rewards, window), label=label, linewidth=2)
        ax_stags.plot(ma_episodes, moving_avg(stags, window), label=label, linewidth=2)
        ax_maulings.plot(ma_episodes, moving_avg(maulings, window), label=label, linewidth=2)
        ax_forage.plot(ma_episodes, moving_avg(forage, window), label=label, linewidth=2)

    # Subplot 1
    ax_rewards.set_title(f"Total Reward")
    ax_rewards.set_xlabel("Episode")
    ax_rewards.set_ylabel("Reward")
    ax_rewards.grid(True, linestyle="--", alpha=0.5)
    ax_rewards.legend()

    # Subplot 2
    ax_stags.set_title(f"Stags Hunted (Coop.)")
    ax_stags.set_xlabel("Episode")
    ax_stags.set_ylabel("Stags / Episode")
    ax_stags.grid(True, linestyle="--", alpha=0.5)
    ax_stags.legend()

    # Subplot 3
    ax_maulings.set_title(f"Maulings (Failed Coop.)")
    ax_maulings.set_xlabel("Episode")
    ax_maulings.set_ylabel("Maulings / Episode")
    ax_maulings.grid(True, linestyle="--", alpha=0.5)
    ax_maulings.legend()

    # Subplot 4
    ax_forage.set_title(f"Foraging (Risk-free low reward)")
    ax_forage.set_xlabel("Episode")
    ax_forage.set_ylabel("Forage / Episode")
    ax_forage.grid(True, linestyle="--", alpha=0.5)
    ax_forage.legend()

    plt.tight_layout()
    plt.show()

    ## compare Q-value
    fig, ax_q = plt.subplots(figsize=(10, 5))

    for label, filepath in file_dict.items():
        df = pd.read_csv(filepath)
        q_values = df["Q_values"].values
        ma_episodes = np.arange(window, len(q_values) + 1) if len(q_values) >= window else np.arange(1, len(q_values) + 1)
        ax_q.plot(ma_episodes, moving_avg(q_values, window), label=label, linewidth=2)

    ax_q.set_title(f"Mean Q-Value Trend (MA {window})", fontsize=14, fontweight='bold')
    ax_q.set_xlabel("Episode")
    ax_q.set_ylabel("Mean Q-Value")
    ax_q.grid(True, linestyle="--", alpha=0.5) 
    ax_q.legend()
    plt.show()

def compare_experiments_harvest(file_dict, title, window=50):
    '''
    Compare the training metrics of different experiments for the Harvest environment.
    file_dict: 
        es. {
                "Standard DQN": "results/standard_dqn.csv", 
                "Target Net DQN": "results/target_net.csv"
            }
    '''
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")
    
    gs = gridspec.GridSpec(2, 4)
    
    ax_rewards = fig.add_subplot(gs[0, 0:2])    
    ax_mature = fig.add_subplot(gs[0, 2:4])    
    ax_young = fig.add_subplot(gs[1, 1:3])      # colonne centrali (centrato)

    for label, filepath in file_dict.items():
        df = pd.read_csv(filepath)
        
        rewards = df["totRewards"].values
        young = df["YoungPlantsHarvested"].values
        mature = df["MaturePlantsHarvested"].values

        ma_episodes = np.arange(window, len(rewards) + 1) if len(rewards) >= window else np.arange(1, len(rewards) + 1)

        ax_rewards.plot(ma_episodes, moving_avg(rewards, window), label=label, linewidth=2)
        ax_mature.plot(ma_episodes, moving_avg(mature, window), label=label, linewidth=2)
        ax_young.plot(ma_episodes, moving_avg(young, window), label=label, linewidth=2)

    ax_rewards.set_title("Total Reward")
    ax_rewards.set_xlabel("Episode")
    ax_rewards.set_ylabel("Reward")
    ax_rewards.grid(True, linestyle="--", alpha=0.5)
    ax_rewards.legend()

    ax_mature.set_title("Mature Plants Harvested")
    ax_mature.set_xlabel("Episode")
    ax_mature.set_ylabel("Plants / Episode")
    ax_mature.grid(True, linestyle="--", alpha=0.5)
    ax_mature.legend()

    ax_young.set_title("Young Plants Harvested")
    ax_young.set_xlabel("Episode")
    ax_young.set_ylabel("Plants / Episode")
    ax_young.grid(True, linestyle="--", alpha=0.5)
    ax_young.legend()

    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.show()

def compare_experiments_escalation(file_dict, title, window=50):
    '''
    Compare the training metrics of different experiments for the Escalation environment.
    file_dict: 
        es. {
                "Standard DQN": "results/standard_dqn.csv", 
                "MAPPO": "results/mappo.csv"
            }
    '''
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")
    
    gs = gridspec.GridSpec(2, 4)
    
    ax_rewards = fig.add_subplot(gs[0, 0:2])      
    ax_max_streak = fig.add_subplot(gs[0, 2:4])   
    ax_coop_steps = fig.add_subplot(gs[1, 1:3])   # colonne centrali (centrato)

    for label, filepath in file_dict.items():
        df = pd.read_csv(filepath)
        
        rewards = df["totRewards"].values
        max_streak = df["MaxStreak"].values
        coop_steps = df["CooperationSteps"].values

        ma_episodes = np.arange(window, len(rewards) + 1) if len(rewards) >= window else np.arange(1, len(rewards) + 1)

        ax_rewards.plot(ma_episodes, moving_avg(rewards, window), label=label, linewidth=2)
        ax_max_streak.plot(ma_episodes, moving_avg(max_streak, window), label=label, linewidth=2)
        ax_coop_steps.plot(ma_episodes, moving_avg(coop_steps, window), label=label, linewidth=2)

    ax_rewards.set_title("Total Reward")
    ax_rewards.set_xlabel("Episode")
    ax_rewards.set_ylabel("Reward")
    ax_rewards.grid(True, linestyle="--", alpha=0.5)
    ax_rewards.legend()

    ax_max_streak.set_title("Max Cooperation Streak (Continuous Scaling)")
    ax_max_streak.set_xlabel("Episode")
    ax_max_streak.set_ylabel("Max Streak Length")
    ax_max_streak.grid(True, linestyle="--", alpha=0.5)
    ax_max_streak.legend()

    ax_coop_steps.set_title("Total Cooperation Steps")
    ax_coop_steps.set_xlabel("Episode")
    ax_coop_steps.set_ylabel("Steps / Episode")
    ax_coop_steps.grid(True, linestyle="--", alpha=0.5)
    ax_coop_steps.legend()

    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.show()