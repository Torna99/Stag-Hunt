### File used to plot the same algorithm on stag-hunt games with different configuration for the report

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

window = 100
dir_path = "results/"
game_dir = "StagHunt/"
algorithm = "A2C_2agents_train_metrics.csv"
title = f""
sub_dirs = [
    "standard1M3_5-1/",
    "standard1M3_5-3/",
    "standard1M3_10-1/",
    "standard1M3_10-3/",
]
labels = [
    "Reward 5, Penalty -1",
    "Reward 5, Penalty -3",
    "Reward 10, Penalty -1",
    "Reward 10, Penalty -3",
]

def moving_avg(data, w):
    if len(data) < w:
        return data
    return np.convolve(data, np.ones(w) / w, mode='valid')

if __name__ == "__main__":
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=15, fontweight="bold")
    
    ax_rewards, ax_stags, ax_maulings, ax_forage = axes.flatten()

    for i, sub_dir in enumerate(sub_dirs):
        filepath = dir_path + game_dir + sub_dir + algorithm
        df = pd.read_csv(filepath)
        
        rewards = df["totRewards"].values
        stags = df["Stags"].values
        maulings = df["Maulings"].values
        forage = df["Forage"].values

        ma_episodes = (
            np.arange(window, len(rewards) + 1)
            if len(rewards) >= window
            else np.arange(1, len(rewards) + 1)
        )
        label = labels[i]

        # 1. Total Reward
        ax_rewards.plot(ma_episodes, moving_avg(rewards, window), label=label, linewidth=2)
        # 2. Stags Hunted (Coop)
        ax_stags.plot(ma_episodes, moving_avg(stags, window), label=label, linewidth=2)
        # 3. Maulings 
        ax_maulings.plot(ma_episodes, moving_avg(maulings, window), label=label, linewidth=2)
        # 4. Forage 
        ax_forage.plot(ma_episodes, moving_avg(forage, window), label=label, linewidth=2)

    # Subplot 1
    ax_rewards.set_title("Total Reward")
    ax_rewards.set_xlabel("Episode")
    ax_rewards.set_ylabel("Reward")
    ax_rewards.grid(True, linestyle="--", alpha=0.5)
    ax_rewards.legend()

    # Subplot 2
    ax_stags.set_title("Stags Hunted (Coop.)")
    ax_stags.set_xlabel("Episode")
    ax_stags.set_ylabel("Stags / Episode")
    ax_stags.grid(True, linestyle="--", alpha=0.5)
    ax_stags.legend()

    # Subplot 3
    ax_maulings.set_title("Maulings (Failed Coop.)")
    ax_maulings.set_xlabel("Episode")
    ax_maulings.set_ylabel("Maulings / Episode")
    ax_maulings.grid(True, linestyle="--", alpha=0.5)
    ax_maulings.legend()

    # Subplot 4
    ax_forage.set_title("Foraging (Risk-free low reward)")
    ax_forage.set_xlabel("Episode")
    ax_forage.set_ylabel("Forage / Episode")
    ax_forage.grid(True, linestyle="--", alpha=0.5)
    ax_forage.legend()

    plt.tight_layout()
    plt.show()