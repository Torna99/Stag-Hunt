import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils 

MA = 100

if __name__ == "__main__":

    experiments = {
        "Vanilla DQN (No Target Net)": "results/stdDQN_2agents_train_metrics.csv",
        "Standard DQN (with Target Net)": "results/advDQN_2agents_train_metrics.csv",
        "Double DQN (with Target Net)": "results/doubleDQN_2agents_train_metrics.csv",
        "Dueling DQN": "results/duelingDQN_2agents_train_metrics.csv"
    }
    utils.compare_experiments(experiments, title="Training Comparison: Standard vs Target DQN", window=MA)