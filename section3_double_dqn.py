"""
Section 3 - Improved DQN: Double DQN (van Hasselt, Guez & Silver, 2016) on CartPole-v1.

Everything (network, experience_replay, sample_batch, sample_action, train_agent, test_agent and
hyper-parameters) is identical to Section 2; the only change is how the target is computed, so the
comparison with Section 2 isolates the effect of Double DQN.

Usage:
    python section3_double_dqn.py --layers 3
    python section3_double_dqn.py --layers 3 --render
Outputs in ./results/section3 and TensorBoard logs in ./runs/section3_<layers>L.
"""
import argparse
import os

import torch

from section2_dqn import run


def double_dqn_target(q_net, target_net, r, s2, done, gamma):
    """Double DQN target:
        a* = argmax_a Q(s', a; theta)          (online network SELECTS the action)
        y  = r + gamma * Q(s', a*; theta^-)    (target network EVALUATES it)
    and y = r for terminal s'. Decoupling selection from evaluation removes the
    over-estimation bias of the max operator in the standard DQN target."""
    with torch.no_grad():
        best_actions = q_net(s2).argmax(dim=1, keepdim=True)
        next_q = target_net(s2).gather(1, best_actions)
    return r + gamma * (1.0 - done) * next_q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=3, choices=[3, 5])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run("Double DQN", args.layers, os.path.join("results", "section3"), double_dqn_target,
        args.seed, args.render)


if __name__ == "__main__":
    main()
