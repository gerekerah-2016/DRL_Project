# Deep Reinforcement Learning – Assignment 1: From Q-learning to DQN

Ben-Gurion University of the Negev, Stein Faculty of Computer and Information Science.

| Section | Script | Environment | Method |
|---|---|---|---|
| 1 | `section1_qlearning.py` | FrozenLake-v1 | Tabular Q-learning, decaying ε-greedy |
| 2 | `section2_dqn.py` | CartPole-v1 | DQN with experience replay and a target network (3 or 5 hidden layers) |
| 3 | `section3_dueling_dqn.py` | CartPole-v1 | Dueling DQN (main improvement) |
| 3 (extra) | `section3_double_dqn.py` | CartPole-v1 | Double DQN |

The assignment names `FrozenLake-v0`, which no longer exists in Gymnasium. `FrozenLake-v1` is the same 4×4 slippery lake.

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
python section1_qlearning.py              # add --search to repeat the hyper-parameter search
python section2_dqn.py --layers 3
python section2_dqn.py --layers 5
python section3_dueling_dqn.py --layers 3
python section3_dueling_dqn.py --layers 5
python section3_double_dqn.py --layers 3   # extra experiment
python section3_double_dqn.py --layers 5
python make_comparison.py                 # comparison figure + results/summary.json
```

Add `--render` to the DQN scripts to watch the trained agent play. Plots are written to `results/`, and TensorBoard logs to `runs/`:

```bash
tensorboard --logdir runs
```

Each DQN run takes roughly 5–15 minutes on a CPU. All runs use seed 0.

## Results

| Run | Episodes until 100-episode average ≥ 475 | Greedy test mean (100 episodes) |
|---|---|---|
| DQN, 3 layers | 704 | 498.3 |
| DQN, 5 layers | 916 | 500.0 |
| Dueling DQN, 3 layers | 617 | 500.0 |
| Dueling DQN, 5 layers | 897 | 500.0 |
| Double DQN, 3 layers | 942 | 500.0 |
| Double DQN, 5 layers | not reached in 1000 | 140.9 |

FrozenLake Q-learning: 72.8% greedy success rate (≈74% is optimal on the slippery lake).


See `report.docx` for answers, hyper-parameters, results and figures.
