# Autonomous and Adaptive Systems – Stag-Hunt

This repository contains the code and the report of the project for the Autonomous and Adaptive Systems course at the University of Bologna. It compares four DQN variants (Vanilla, Standard, Double, Dueling), A2C and MAPPO on the three environments of the [Gymnasium-Stag-Hunt](https://github.com/giorgiofranceschelli/Gymnasium-Stag-Hunt) environment.

| Environment | Challenge |
|---|---|
| **Hunt** (Stag-Hunt) | Hunting the stag gives the highest reward but requires both agents coordination, hunting it alone leads to a mauling penalty. Foraging is safe but low-reward.  |
| **Harvest** | Plants grow over time: mature plants give higher reward than young plants (learn to delay harvesting) |
| **Escalation** | Agents must coordinate step by step towards the stag, if one breaks the streak a penalty proportional to the streak length is applied. |

The full analysis of the tests and results are in [Report](Report.pdf).


## Repository structure

```
.
├── Report.pdf                  # Project report 
└── src
    ├── agents                  # One agent per algorithm
    │   ├── vanilladqn_agent.py
    │   ├── standarddqn_agent.py
    │   ├── doubledqn_agent.py
    │   ├── duelingdqn_agent.py
    │   ├── a2c_agent.py
    │   └── mappo_agent.py
    ├── buffers                 # Replay / rollout memories
    │   ├── replay.py           
    │   ├── a2c_memory.py       
    │   └── mappo_memory.py     
    ├── models                  # Neural network architectures
    │   ├── dqn.py
    │   ├── dueling_dqn.py
    │   ├── a2c.py
    │   └── mappo.py
    ├── train_staghunt.py      
    ├── train_harvest.py        
    ├── train_escalation.py     
    ├── plot_comparison.py      # Plots
    ├── plot_same_algorithm.py  # Plots
    └── utils.py                
```

## Installation and usage

1. Install the Stag-Hunt environments:

```bash
git clone https://github.com/giorgiofranceschelli/Gymnasium-Stag-Hunt.git
cd Gymnasium-Stag-Hunt
pip install .
cd ..
```
### Run a single experiment

```bash
# Stag-Hunt
python src/train_staghunt.py   

# Harvest
python src/train_harvest.py

# Escalation
python src/train_escalation.py 
```

**Common arguments**

| Argument | Default | Description |
|---|---|---|
| `--algorithm` | `duelingDQN` | `vanillaDQN`, `standardDQN`, `doubleDQN`, `duelingDQN`, `a2c`, `mappo` |
| `--max_episodes` | 2000 | Number of training episodes |
| `--max_steps_per_episode` | 200 | Episode length |
| `--grid_size` | 5 | Environment's grid size |
| `--discount_factor` | 0.99 | Discount factor γ |
| `--learning_rate` | 1e-3 | Learning rate |
| `--batch_size` | 64 | Batch size |
| `--replay_buffer_size` | 10000 | Replay buffer size (DQN) |
| `--epsilon_start` | 1.0 | Initial epsilon for epsilon-greedy (DQN) |
| `--epsilon_decay` | 0.995 | Multiplicative decay |
| `--epsilon_min` | 0.01 | Minimum epsilon |
| `--target_update_freq` | 500 | Target network update period C |
| `--gae_lambda` | 0.95 | GAE lambda |
| `--ppo_epochs` | 5 | PPO epochs per update |
| `--epsilon_clip` | 0.2 | PPO clipping value |
| `--critic_coeff` | 0.5 | Critic loss coefficient |
| `--entropy_coeff` | 0.01 | Entropy coefficient |
| `--save_dir` | `general` | Subfolder of `results/<Game>/` to save metrics |

**Stag-Hunt only**

| Argument | Default | Description |
|---|---|---|
| `--forage_qta` | 2 | Quantity of forage |
| `--forage_reward` | 1 | Reward for foraging |
| `--stag_reward` | 5 | Reward for hunting the stag together |
| `--mauling_penalty` | −3 | Penalty for hunting the stag alone |

**Harvest only**

| Argument | Default | Description |
|---|---|---|
| `--max_plants` | 5 | Maximum number of plants |
| `--chance_mature` | 0.1 | Probability that a plant grows |
| `--chance_die` | 0.1 | Probability that a plant dies |
| `--young_reward` | 1 | Reward for harvesting a young plant |
| `--mature_reward` | 3 | Reward for harvesting a mature plant |

**Escalation only**

| Argument | Default | Description |
|---|---|---|
| `--punishment_factor` | 0.5 | Penalty factor multiplied by the streak length when an agent breaks the coordination |

## Experiments

- **Algorithm comparison** on all three environments (Hunt, Harvest and Escalation).
- **Learning-rate study** for Vanilla DQN and MAPPO on Stag-Hunt.
- **Reward/penalty sensitivity** on Stag-Hunt.
- **Grid-size scaling** on Stag-Hunt

⚠️The seed is fixed to 42 for reproducibility.
