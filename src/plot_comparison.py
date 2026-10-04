### File used to plot the comparison of different algorithms on the same game for the report

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils 

MA = 100

dir_path = "results/"
# game_dir = "StagHunt/"
# game_dir = "Harvest/"
game_dir = "Escalation/"
sub_dir = "g5x5_lr1M3/"
#title = sub_dir.replace("/", "")
title = ""
if __name__ == "__main__":

    # experiments = {
    #     "Vanilla DQN" : dir_path + game_dir + sub_dir + "vanillaDQN_2agents_train_metrics.csv",
    #     "Standard DQN" : dir_path + game_dir + sub_dir + "standardDQN_2agents_train_metrics.csv",
    #     "Double DQN" : dir_path + game_dir + sub_dir + "doubleDQN_2agents_train_metrics.csv",
    #     "Dueling DQN" : dir_path + game_dir + sub_dir + "duelingDQN_2agents_train_metrics.csv",
    #     "MAPPO" : dir_path + game_dir + sub_dir + "mappo_2agents_train_metrics.csv",
    #     "A2C": dir_path + game_dir + sub_dir + "a2c_2agents_train_metrics.csv"
    # }
    # utils.compare_experiments_stag_hunt(experiments, title=f"{title}", window=MA)


    # experiments = {
    #     "Vanilla DQN" : dir_path + game_dir + sub_dir + "vanillaDQN_2agents_harvest_train_metrics.csv",
    #     "Standard DQN" : dir_path + game_dir + sub_dir + "standardDQN_2agents_harvest_train_metrics.csv",
    #     "Double DQN" : dir_path + game_dir + sub_dir + "doubleDQN_2agents_harvest_train_metrics.csv",
    #     "Dueling DQN" : dir_path + game_dir + sub_dir + "duelingDQN_2agents_harvest_train_metrics.csv",
    #     "MAPPO" : dir_path + game_dir + sub_dir + "mappo_2agents_harvest_train_metrics.csv",
    #     "A2C": dir_path + game_dir + sub_dir + "a2c_2agents_harvest_train_metrics.csv"
    # }
    # utils.compare_experiments_harvest(experiments, title=f"{title}", window=MA)


    experiments = {
        "Vanilla DQN" : dir_path + game_dir + sub_dir + "vanillaDQN_2agents_escalation_train_metrics.csv",
        "Standard DQN" : dir_path + game_dir + sub_dir + "standardDQN_2agents_escalation_train_metrics.csv",
        "Double DQN" : dir_path + game_dir + sub_dir + "doubleDQN_2agents_escalation_train_metrics.csv",
        "Dueling DQN" : dir_path + game_dir + sub_dir + "duelingDQN_2agents_escalation_train_metrics.csv",
        "MAPPO" : dir_path + game_dir + sub_dir + "mappo_2agents_escalation_train_metrics.csv",
        "A2C": dir_path + game_dir + sub_dir + "a2c_2agents_escalation_train_metrics.csv"
    }
    utils.compare_experiments_escalation(experiments, title=f"{title}", window=MA)