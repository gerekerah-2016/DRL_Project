"""
Section 3 - Improved DQN (main improvement): Dueling DQN (Wang et al., 2016) on CartPole-v1.

The Q-network is split into two streams after a shared trunk:
    V(s)   - how good the state is,
    A(s,a) - how much better each action is than the others,
combined as Q(s,a) = V(s) + A(s,a) - mean_a' A(s,a').
Everything else (experience_replay, sample_batch, sample_action, train_agent, test_agent,
standard DQN target and all hyper-parameters) is identical to Section 2, so the comparison
isolates the effect of the dueling architecture.

The number of hidden layers matches Section 2: with --layers 3 there are 2 shared hidden layers
followed by 1 hidden layer in each stream (3 hidden layers on every path from input to output).

Usage:
    python section3_dueling_dqn.py --layers 3
    python section3_dueling_dqn.py --layers 5
Outputs in ./results/section3_dueling and TensorBoard logs in ./runs/.
"""
import argparse
import os

import torch
import torch.nn as nn

from section2_dqn import dqn_target, run


class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, n_hidden_layers, hidden_size):
        super().__init__()
        trunk, in_dim = [], state_dim
        for _ in range(n_hidden_layers - 1):                 # shared hidden layers
            trunk += [nn.Linear(in_dim, hidden_size), nn.ReLU()]
            in_dim = hidden_size
        self.trunk = nn.Sequential(*trunk)
        self.value = nn.Sequential(nn.Linear(in_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))
        self.advantage = nn.Sequential(nn.Linear(in_dim, hidden_size), nn.ReLU(),
                                       nn.Linear(hidden_size, n_actions))

    def forward(self, x):
        h = self.trunk(x)
        v, a = self.value(h), self.advantage(h)
        return v + a - a.mean(dim=1, keepdim=True)          # identifiable aggregation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=3, choices=[3, 5])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run("Dueling DQN", args.layers, os.path.join("results", "section3_dueling"), dqn_target,
        args.seed, args.render, net_cls=DuelingQNetwork)


if __name__ == "__main__":
    main()
