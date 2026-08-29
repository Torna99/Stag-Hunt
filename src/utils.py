### Enrico Tornabene (0001191764)
###

import matplotlib.pyplot as plt
import numpy as np

def moving_avg(data, w):
    if len(data) < w:
        return data
    return np.convolve(data, np.ones(w) / w, mode='valid')

def plot_training_results(rewards, stags, maulings, forage, window=50):

    episodes = np.arange(1, len(rewards) + 1)
    ma_episodes = np.arange(window, len(rewards) + 1) if len(rewards) >= window else episodes

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Multi-Agent DQN: Stag Hunt Training Results", fontsize=14, fontweight='bold')

    # 1. Total Reward per Episode
    ax1.plot(episodes, rewards, alpha=0.25, color="steelblue", label="Raw")
    ax1.plot(ma_episodes, moving_avg(rewards, window), color="navy", linewidth=2, label=f"MA {window}")
    ax1.set_title("Total Reward")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # 2. Game Dynamics (Moving Average)
    ax2.plot(ma_episodes, moving_avg(stags, window), color="forestgreen", linewidth=2, label="Stags (Coop)")
    ax2.plot(ma_episodes, moving_avg(forage, window), color="darkorange", linewidth=2, label="Forage (Safe)")
    ax2.plot(ma_episodes, moving_avg(maulings, window), color="crimson", linewidth=2, label="Maulings (Risk)")
    ax2.set_title(f"Game Dynamics (MA {window})")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Events / Episode")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    #plt.savefig("training_results.png", dpi=300)
    plt.show()

