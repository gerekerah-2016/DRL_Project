# Deep Reinforcement Learning – Assignment 1: From Q-learning to Deep Q-learning (DQN)

**Ben-Gurion University of the Negev**
**Faculty of Computer and Information Science**
**Course:** Deep Reinforcement Learning

**Students:** Nechi Berhe Weldu (ID 850164070), Gebreslassie Teklu Reda (ID 850170440)

## Project Overview

This project implements and evaluates reinforcement learning algorithms using Python and Gymnasium environments. The assignment explores the transition from tabular Q-learning to Deep Q-learning (DQN), focusing on training performance, hyperparameter optimization, and algorithmic improvements.

The project is divided into three sections:

1. **Tabular Q-learning:** Implementation of Q-learning on the FrozenLake environment, including epsilon-greedy exploration, hyperparameter tuning, and performance visualization.
2. **Deep Q-learning (DQN):** Implementation of DQN using PyTorch on the CartPole environment, including experience replay, target networks, and comparison of neural network architectures.
3. **Improved DQN:** Implementation and evaluation of an enhanced DQN algorithm (Dueling DQN, with Double DQN as an additional experiment), comparing its performance with the basic DQN implementation.

Each section includes Python scripts, experimental results, and visualizations. The accompanying Word report contains theoretical answers, experimental analysis, and comparisons.

## Project Structure

```
├── report.docx               Report: theory answers, results, analysis
├── section1_qlearning.py     Section 1 – Tabular Q-learning (FrozenLake)
├── section2_dqn.py           Section 2 – DQN (CartPole), 3 or 5 hidden layers
├── section3_dueling_dqn.py   Section 3 – Dueling DQN (main improvement)
├── section3_double_dqn.py    Section 3 – Double DQN (additional experiment)
├── make_comparison.py        Summary tables and comparison figure
├── requirements.txt          Python dependencies
└── results/                  Plots, data (CSV/JSON) and configurations
```

## Installation

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

## Running the Code

```bash
# Section 1 – hyperparameter search, then training
python section1_qlearning.py --search

# Section 2 – DQN with 3 and 5 hidden layers
python section2_dqn.py --layers 3
python section2_dqn.py --layers 5

# Section 3 – Dueling DQN and Double DQN
python section3_dueling_dqn.py --layers 3
python section3_dueling_dqn.py --layers 5
python section3_double_dqn.py --layers 3
python section3_double_dqn.py --layers 5

# Comparison figure and summary tables
python make_comparison.py
```

Useful options: `--seed N` sets the random seed (default 0), `--render` shows the trained agent playing, and `--load <model file>` tests a saved agent. `section2_dqn.py --constant-lr` runs the learning-rate experiment described in the report.

Training curves can be viewed in TensorBoard:

```bash
tensorboard --logdir runs
```

Note: the assignment specifies `FrozenLake-v0`, which is no longer available in Gymnasium; `FrozenLake-v1` is the same environment and is used instead.

## Results

| Method | Episodes to reach an average reward of 475 (over 100 episodes) |
|---|---|
| Q-learning, FrozenLake | 72.8% success rate on 1000 test episodes |
| DQN, 3 hidden layers | 704 (seed 0); 622 ± 90 over 3 seeds |
| DQN, 5 hidden layers | 916 (seed 0) |
| Dueling DQN, 3 hidden layers | 617 (seed 0); 631 ± 140 over 3 seeds |
| Dueling DQN, 5 hidden layers | 897 (seed 0) |
| Double DQN, 3 hidden layers | 942 (seed 0) |

Full results, plots and discussion are in `report.docx`.
