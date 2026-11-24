# Deep Reinforcement Learning – Assignment 1: From Q-learning to DQN

Ben-Gurion University of the Negev, Department of Software and Information Systems Engineering.

| Section | Script | Environment | Method |
|---|---|---|---|
| 1 | `section1_qlearning.py` | FrozenLake-v1 | Tabular Q-learning, decaying ε-greedy |
| 2 | `section2_dqn.py` | CartPole-v1 | DQN with experience replay and a target network (3 or 5 hidden layers) |
| 3 | `section3_dueling_dqn.py` | CartPole-v1 | **Dueling DQN – the Section 3 submission** |
| 3 (extra) | `section3_double_dqn.py` | CartPole-v1 | Double DQN, an additional experiment |

The assignment names `FrozenLake-v0`, which no longer exists in Gymnasium. `FrozenLake-v1` is the same 4×4 slippery lake.

## Setup

Python 3.10+:

```bash
pip install -r requirements.txt
```

## Reproducing every result

Run from this folder. Each DQN run takes roughly 5–15 minutes on one CPU core.

```bash
# Section 1: grid search (validation seeds) -> results/section1/best_config.json, then final training
python section1_qlearning.py --search

# Section 2: DQN
python section2_dqn.py --layers 3
python section2_dqn.py --layers 5
python section2_dqn.py --layers 3 --constant-lr      # learning-rate ablation in the report

# Section 3: Dueling DQN (primary) and Double DQN (extra)
python section3_dueling_dqn.py --layers 3
python section3_dueling_dqn.py --layers 5
python section3_double_dqn.py --layers 3
python section3_double_dqn.py --layers 5

# Extra seeds for the Section 3 comparison
python section2_dqn.py --layers 3 --seed 1
python section3_dueling_dqn.py --layers 3 --seed 1
python section2_dqn.py --layers 3 --seed 2
python section3_dueling_dqn.py --layers 3 --seed 2

# Summary tables (results/summary.json, summary.csv) and comparison figure
python make_comparison.py

# Optional: rebuild report.docx from the saved results (needs Node.js and `npm install docx`)
node build_report.js
```

Test a saved agent (optionally watching it play):

```bash
python section2_dqn.py --load results/section3_dueling/3L_seed0_model.pt --render
```

TensorBoard logs (reward, loss per training step, 100-episode average, ε, learning rate):

```bash
tensorboard --logdir runs
```

## Definitions and seeds

- **Environment step**: one `env.step` call. **Training step**: one gradient update (one per environment step once the replay holds 1,000 transitions). The target network is synced every **C = 500 environment steps**.
- **Episodes to solve**: the last episode of the first window of 100 consecutive *training* episodes with average reward ≥ 475 (checked independently in code).
- **Greedy test**: ε = 0 on 100 new episodes with seeds 123456–123555, never used in training.
- Only a real failure (`terminated`) is treated as terminal; the time limit (`truncated`) still bootstraps.
- Seeds: CartPole training episode *i* of seed *k* uses env seed *k*·10000 + *i*. FrozenLake uses *k*·100000 + *i*, validation seeds 50000+ and test seeds 10000+.

## Output files

Per CartPole run (`results/<section>/<L>L_seed<k>[_constlr]`): `_reward.png`, `_loss.png`, `_episodes.csv` (reward, 100-episode average, ε, learning rate), `_loss.csv`, `_data.json`, `_config.json` (all settings, parameter count, versions) and `_model.pt` (weights with architecture metadata).

Section 1 (`results/section1`): Q-table colormaps, plots, `hp_search.csv/json`, `best_config.json`, `episodes.csv`, `metrics.json`.

## Results

See `report.docx` and `results/summary.csv`.
