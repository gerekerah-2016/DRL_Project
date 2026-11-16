"""
Section 2 - Deep Q-Network (DQN) with experience replay and a target network on CartPole-v1.

Usage:
    python section2_dqn.py --layers 3          # network with 3 hidden layers
    python section2_dqn.py --layers 5          # network with 5 hidden layers
    python section2_dqn.py --layers 3 --render # watch the trained agent after training
Outputs (in ./results/section2): loss / reward plots, saved data, trained weights,
and TensorBoard logs in ./runs/section2_<layers>L.
"""
import argparse
import json
import os
import random
import time
from collections import deque

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

ENV_ID = "CartPole-v1"
SOLVED_SCORE = 475.0

HP = dict(
    hidden_size=128,          # units in every hidden layer
    lr=1e-3,                  # initial Adam learning rate
    lr_decay=0.995,           # learning rate multiplied by this after every episode
    lr_min=1e-5,              # floor for the learning rate
    gamma=0.99,               # discount factor
    batch_size=64,            # minibatch size
    replay_size=50_000,       # capacity of experience_replay
    min_replay=1_000,         # transitions collected before learning starts
    target_update=500,        # C: copy theta -> theta^- every C steps
    eps_start=1.0,
    eps_min=0.01,
    eps_decay=0.98,           # multiplicative decay of epsilon per episode
    grad_clip=10.0,           # max gradient norm
    max_episodes=1000,
)


class QNetwork(nn.Module):
    """MLP: state -> Q-value of every action. Used for both the Q-network and the target network."""

    def __init__(self, state_dim, n_actions, n_hidden_layers, hidden_size):
        super().__init__()
        layers, in_dim = [], state_dim
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(in_dim, hidden_size), nn.ReLU()]
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def make_experience_replay(capacity):
    """experience_replay: fixed-size deque of (s, a, r, s', done) transitions."""
    return deque(maxlen=capacity)


def sample_batch(experience_replay, batch_size):
    """Sample a random minibatch (uniform, without replacement) from experience_replay."""
    batch = random.sample(experience_replay, batch_size)
    s, a, r, s2, d = zip(*batch)
    return (torch.as_tensor(np.array(s), dtype=torch.float32),
            torch.as_tensor(a, dtype=torch.int64).unsqueeze(1),
            torch.as_tensor(r, dtype=torch.float32).unsqueeze(1),
            torch.as_tensor(np.array(s2), dtype=torch.float32),
            torch.as_tensor(d, dtype=torch.float32).unsqueeze(1))


def sample_action(q_net, state, epsilon, n_actions):
    """Decaying epsilon-greedy action selection."""
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        q = q_net(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
    return int(q.argmax(dim=1).item())


def dqn_target(q_net, target_net, r, s2, done, gamma):
    """Standard DQN target: y = r + gamma * max_a Q(s', a; theta^-), and y = r for terminal s'."""
    with torch.no_grad():
        max_next_q = target_net(s2).max(dim=1, keepdim=True)[0]
    return r + gamma * (1.0 - done) * max_next_q


def train_agent(n_hidden_layers, hp=HP, seed=0, target_fn=dqn_target, log_dir=None, verbose=True,
                net_cls=None):
    """Train with the DQN algorithm of figure 4. Stops once the 100-episode average reaches 475."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = gym.make(ENV_ID)
    state_dim, n_actions = env.observation_space.shape[0], env.action_space.n

    net_cls = net_cls or QNetwork
    q_net = net_cls(state_dim, n_actions, n_hidden_layers, hp["hidden_size"])
    target_net = net_cls(state_dim, n_actions, n_hidden_layers, hp["hidden_size"])
    target_net.load_state_dict(q_net.state_dict())      # theta^- = theta
    optimizer = torch.optim.Adam(q_net.parameters(), lr=hp["lr"])
    loss_fn = nn.MSELoss()
    experience_replay = make_experience_replay(hp["replay_size"])
    writer = SummaryWriter(log_dir) if log_dir else None

    episode_rewards, losses = [], []
    epsilon, total_steps, solved_at = hp["eps_start"], 0, None
    start = time.time()

    for ep in range(hp["max_episodes"]):
        state, _ = env.reset(seed=seed * 10_000 + ep)
        ep_reward, done = 0.0, False
        while not done:
            action = sample_action(q_net, state, epsilon, n_actions)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # Only a real failure is terminal; the 500-step time limit (truncation) is not.
            experience_replay.append((state, action, reward, next_state, float(terminated)))
            state = next_state
            ep_reward += reward
            total_steps += 1

            if len(experience_replay) >= hp["min_replay"]:
                s, a, r, s2, d = sample_batch(experience_replay, hp["batch_size"])
                y = target_fn(q_net, target_net, r, s2, d, hp["gamma"])
                q_sa = q_net(s).gather(1, a)
                loss = loss_fn(q_sa, y)                  # (y_j - Q(s_j, a_j; theta))^2
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), hp["grad_clip"])
                optimizer.step()
                losses.append(loss.item())
                if writer:
                    writer.add_scalar("loss_per_step", loss.item(), len(losses))

            if total_steps % hp["target_update"] == 0:
                target_net.load_state_dict(q_net.state_dict())   # every C steps

        episode_rewards.append(ep_reward)
        epsilon = max(hp["eps_min"], epsilon * hp["eps_decay"])
        for g in optimizer.param_groups:              # decay the learning rate per episode
            g["lr"] = max(hp["lr_min"], g["lr"] * hp["lr_decay"])
        avg100 = np.mean(episode_rewards[-100:])
        if writer:
            writer.add_scalar("reward_per_episode", ep_reward, ep)
            writer.add_scalar("avg_reward_last100", avg100, ep)
            writer.add_scalar("epsilon", epsilon, ep)
            writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], ep)
        if verbose and (ep + 1) % 25 == 0:
            print(f"ep {ep+1:4d} | reward {ep_reward:5.0f} | avg100 {avg100:6.1f} | "
                  f"eps {epsilon:.3f} | steps {total_steps} | {time.time()-start:5.0f}s", flush=True)
        if len(episode_rewards) >= 100 and avg100 >= SOLVED_SCORE:
            solved_at = ep + 1
            if verbose:
                print(f"Solved: average reward {avg100:.1f} over episodes {solved_at-99}-{solved_at}")
            break

    env.close()
    if writer:
        writer.close()
    return q_net, episode_rewards, losses, solved_at


def test_agent(q_net, n_episodes=100, render=False, seed=123_456):
    """Run the trained agent greedily on new episodes. Returns the list of episode rewards."""
    env = gym.make(ENV_ID, render_mode="human" if render else None)
    rewards = []
    for i in range(n_episodes):
        state, _ = env.reset(seed=seed + i)
        total, done = 0.0, False
        while not done:
            action = sample_action(q_net, state, 0.0, env.action_space.n)
            state, r, term, trunc, _ = env.step(action)
            total += r
            done = term or trunc
        rewards.append(total)
    env.close()
    return rewards


def plot_results(rewards, losses, title, out_prefix):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rewards, alpha=0.5, label="reward per episode")
    avg = [np.mean(rewards[max(0, i - 99):i + 1]) for i in range(len(rewards))]
    ax.plot(avg, color="C3", lw=2, label="average of last 100 episodes")
    ax.axhline(SOLVED_SCORE, color="gray", ls="--", label="475")
    ax.set_xlabel("Episode"); ax.set_ylabel("Total reward"); ax.set_title(f"{title} - reward per episode")
    ax.legend(loc="upper left"); fig.tight_layout(); fig.savefig(out_prefix + "_reward.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, alpha=0.3, lw=0.5, label="loss per training step")
    if len(losses) > 500:
        k = 500
        ax.plot(np.arange(k - 1, len(losses)), np.convolve(losses, np.ones(k) / k, mode="valid"),
                color="C3", lw=1.5, label="moving average (500 steps)")
    ax.set_yscale("log")
    ax.set_xlabel("Training step"); ax.set_ylabel("MSE loss (log scale)"); ax.set_title(f"{title} - loss per training step")
    ax.legend(loc="upper left"); fig.tight_layout(); fig.savefig(out_prefix + "_loss.png", dpi=150)
    plt.close(fig)


def run(name, n_hidden_layers, out_dir, target_fn, seed, render, net_cls=None):
    os.makedirs(out_dir, exist_ok=True)
    q_net, rewards, losses, solved_at = train_agent(
        n_hidden_layers, seed=seed, target_fn=target_fn, net_cls=net_cls,
        log_dir=os.path.join("runs", f"{os.path.basename(out_dir)}_{n_hidden_layers}L_seed{seed}"))
    test_rewards = test_agent(q_net, render=False)
    print(f"Episodes to reach avg >= 475 over 100 episodes: {solved_at}")
    print(f"Greedy test over 100 new episodes: mean {np.mean(test_rewards):.1f}, min {np.min(test_rewards):.0f}")
    prefix = os.path.join(out_dir, f"{n_hidden_layers}L_seed{seed}")
    plot_results(rewards, losses, f"{name} ({n_hidden_layers} hidden layers)", prefix)
    torch.save(q_net.state_dict(), prefix + "_model.pt")
    with open(prefix + "_data.json", "w") as f:
        json.dump(dict(rewards=rewards, losses=losses, solved_at=solved_at,
                       test_rewards=test_rewards), f)
    if render:
        test_agent(q_net, n_episodes=3, render=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=3, choices=[3, 5])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run("DQN", args.layers, os.path.join("results", "section2"), dqn_target, args.seed, args.render)


if __name__ == "__main__":
    main()
