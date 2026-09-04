import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    #plt.savefig("epsilon_trend.png", dpi=300)
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
    #plt.savefig("q_value_trend.png", dpi=300)
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

def compare_experiments(file_dict, title, window=50):
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

    # Carica i file e traccia le linee per ciascun esperimento
    for label, filepath in file_dict.items():
        df = pd.read_csv(filepath)
        
        rewards = df["totRewards"].values
        stags = df["Stags"].values
        maulings = df["Maulings"].values
        forage = df["Forage"].values

        ma_episodes = np.arange(window, len(rewards) + 1) if len(rewards) >= window else np.arange(1, len(rewards) + 1)

        # 1. Total Reward
        ax_rewards.plot(ma_episodes, moving_avg(rewards, window), label=label, linewidth=2)
        # 2. Stags Hunted (Coop)
        ax_stags.plot(ma_episodes, moving_avg(stags, window), label=label, linewidth=2)
        # 3. Maulings (Risk/Failures)
        ax_maulings.plot(ma_episodes, moving_avg(maulings, window), label=label, linewidth=2)
        # 4. Forage (Safe Actions)
        ax_forage.plot(ma_episodes, moving_avg(forage, window), label=label, linewidth=2)

    # Configurazione Subplot 1: Rewards
    ax_rewards.set_title(f"Total Reward (MA {window})")
    ax_rewards.set_xlabel("Episode")
    ax_rewards.set_ylabel("Reward")
    ax_rewards.grid(True, linestyle="--", alpha=0.5)
    ax_rewards.legend()

    # Configurazione Subplot 2: Stags
    ax_stags.set_title(f"Stags Hunted (Cooperative, MA {window})")
    ax_stags.set_xlabel("Episode")
    ax_stags.set_ylabel("Stags / Episode")
    ax_stags.grid(True, linestyle="--", alpha=0.5)
    ax_stags.legend()

    # Configurazione Subplot 3: Maulings
    ax_maulings.set_title(f"Maulings (Failed Coop, MA {window})")
    ax_maulings.set_xlabel("Episode")
    ax_maulings.set_ylabel("Maulings / Episode")
    ax_maulings.grid(True, linestyle="--", alpha=0.5)
    ax_maulings.legend()

    # Configurazione Subplot 4: Forage
    ax_forage.set_title(f"Foraging (Safe Defection, MA {window})")
    ax_forage.set_xlabel("Episode")
    ax_forage.set_ylabel("Forage / Episode")
    ax_forage.grid(True, linestyle="--", alpha=0.5)
    ax_forage.legend()

    plt.tight_layout()
    #plt.savefig("results/comparison_plot.png", dpi=300)
    plt.show()

    ## compare Q-value trends
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

# def compare_experiments(file_dict, title, window=50, save_fig=False):
#     '''
#     Compare the training metrics of different experiments.
#     '''
#     fig, axes = plt.subplots(2, 2, figsize=(14, 10))
#     fig.suptitle(title, fontsize=15, fontweight="bold")
#     axes = axes.flatten()
    
#     ax_rewards, ax_stags, ax_maulings, ax_forage = axes

#     # Cache per evitare di rileggere i file da disco
#     dataframes = {}

#     # 1. Plot metriche di performance e cooperazione
#     for label, filepath in file_dict.items():
#         df = pd.read_csv(filepath)
#         dataframes[label] = df
        
#         rewards = df["totRewards"].values
#         stags = df["Stags"].values
#         maulings = df["Maulings"].values
#         forage = df["Forage"].values

#         ma_episodes = np.arange(window, len(rewards) + 1) if len(rewards) >= window else np.arange(1, len(rewards) + 1)

#         ax_rewards.plot(ma_episodes, moving_avg(rewards, window), label=label, linewidth=2)
#         ax_stags.plot(ma_episodes, moving_avg(stags, window), label=label, linewidth=2)
#         ax_maulings.plot(ma_episodes, moving_avg(maulings, window), label=label, linewidth=2)
#         ax_forage.plot(ma_episodes, moving_avg(forage, window), label=label, linewidth=2)

#     # Configurazione grafici 2x2
#     ax_rewards.set_title(f"Total Reward (MA {window})")
#     ax_rewards.set_xlabel("Episode")
#     ax_rewards.set_ylabel("Reward")
#     ax_rewards.grid(True, linestyle="--", alpha=0.5)
#     ax_rewards.legend()

#     ax_stags.set_title(f"Stags Hunted (Cooperative, MA {window})")
#     ax_stags.set_xlabel("Episode")
#     ax_stags.set_ylabel("Stags / Episode")
#     ax_stags.grid(True, linestyle="--", alpha=0.5)
#     ax_stags.legend()

#     ax_maulings.set_title(f"Maulings (Failed Coop, MA {window})")
#     ax_maulings.set_xlabel("Episode")
#     ax_maulings.set_ylabel("Maulings / Episode")
#     ax_maulings.grid(True, linestyle="--", alpha=0.5)
#     ax_maulings.legend()

#     ax_forage.set_title(f"Foraging (Safe Defection, MA {window})")
#     ax_forage.set_xlabel("Episode")
#     ax_forage.set_ylabel("Forage / Episode")
#     ax_forage.grid(True, linestyle="--", alpha=0.5)
#     ax_forage.legend()

#     plt.tight_layout()
#     if save_fig:
#         plt.savefig("results/comparison_metrics.png", dpi=300)

#     # 2. Plot dedicato per l'analisi dei Q-Values (Overestimation Bias)
#     fig_q, ax_q = plt.subplots(figsize=(10, 5))
#     has_q_data = False

#     for label, df in dataframes.items():
#         if "Q_values" in df.columns:
#             has_q_data = True
#             q_values = df["Q_values"].values
#             ma_episodes = np.arange(window, len(q_values) + 1) if len(q_values) >= window else np.arange(1, len(q_values) + 1)
#             ax_q.plot(ma_episodes, moving_avg(q_values, window), label=label, linewidth=2)

#     if has_q_data:
#         ax_q.set_title(f"Mean Q-Value Trend (MA {window})", fontsize=14, fontweight='bold')
#         ax_q.set_xlabel("Episode")
#         ax_q.set_ylabel("Mean Q-Value")
#         ax_q.grid(True, linestyle="--", alpha=0.5) 
#         ax_q.legend()
#         if save_fig:
#             plt.savefig("results/comparison_q_values.png", dpi=300)
#     else:
#         plt.close(fig_q)

#     plt.show()