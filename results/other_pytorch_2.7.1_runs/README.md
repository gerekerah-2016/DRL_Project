# Runs made with a different PyTorch build (2.7.1+cu118)

These five runs were repeated with PyTorch 2.7.1+cu118. The report's numbers come from PyTorch 2.15.0, and
re-running with 2.15.0 reproduces them exactly (those files are in `results/section2` and
`results/section3_dueling`). The same seed gives different training runs on different PyTorch builds, so
these files are kept here for transparency. `make_comparison.py` ignores this folder.

| Run | PyTorch 2.15.0 (report) | PyTorch 2.7.1 (this folder) |
|---|---|---|
| DQN, 3 hidden layers, seed 1 | 525 | 634 |
| DQN, 3 hidden layers, seed 2 | 636 | 698 |
| Dueling DQN, 3 hidden layers, seed 1 | 498 | 598 |
| Dueling DQN, 3 hidden layers, seed 2 | 777 | 906 |
| DQN, constant learning rate 5e-4, seed 0 | not solved in 1000 episodes | 601 |

Values are episodes until the 100-episode average reward first reaches 475.
