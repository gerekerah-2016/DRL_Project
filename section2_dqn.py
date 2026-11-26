"""
Section 2 - Deep Q-Network (DQN) with experience replay and a target network on CartPole-v1.

Usage:
    python section2_dqn.py --layers 3                    # 3 hidden layers, seed 0
    python section2_dqn.py --layers 5 --seed 1           # 5 hidden layers, seed 1
    python section2_dqn.py --layers 3 --constant-lr      # ablation: no learning-rate decay
    python section2_dqn.py --layers 3 --lr 1e-3 --lr-decay 1.0   # learning-rate experiment (any lr / decay)
    python section2_dqn.py --load results/section2/3L_seed0_model.pt --render   # test a saved agent
Outputs per run (prefix results/section2/<L>L_seed<seed>[tag]):
    _reward.png, _loss.png    reward per training episode / loss per training step
    _episodes.csv, _loss.csv  the same data as tables
    _data.json                rewards, losses, solved_at, greedy test rewards
    _config.json              every setting needed to reproduce the run
    _model.pt                 weights + architecture metadata (see load_agent)
TensorBoard logs: runs/<name>  (tensorboard --logdir runs)

Definitions used everywhere:
    step            one environment step (env.step call)
    training step   one gradient update; one per environment step once the replay holds min_replay
    C               target_update counts ENVIRONMENT steps (theta^- <- theta when total_steps % C == 0)
    solved_at       last episode of the FIRST 100-episode training window whose average reward >= 475
    test            greedy play (epsilon = 0) on 100 new episodes with seeds never used in training
"""
import argparse
import csv
import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass

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
TEST_SEED = 123_456          # test episodes use seeds 123456..123555 (training uses seed*10000 + episode)

HP = dict(
    hidden_size=128,          # units in every hidden layer
    lr=1e-3,                  # initial Adam learning rate
    lr_decay=0.995,           # learning rate multiplied by this after every episode
    lr_min=1e-5,              # floor for the learning rate
    gamma=0.99,               # discount factor
    batch_size=64,            # minibatch size
    replay_size=50_000,       # capacity of experience_replay
    min_replay=1_000,         # transitions collected before learning starts
    target_update=500,        # C, in environment steps
    eps_start=1.0,
    eps_min=0.01,
    eps_decay=0.98,           # multiplicative decay of epsilon per episode
    grad_clip=10.0,           # max gradient norm
    max_episodes=1000,
)


@dataclass
class Experience:
    """One transition. `terminated` is 1.0 only for a real failure, never for the time-limit truncation."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: float


class QNetwork(nn.Module):
    """MLP: state -> Q-value of every action. One class is used for both the Q-network and the
    target network (same structure, different weights)."""

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
    """experience_replay: fixed-size deque of Experience; the oldest transition is dropped when full."""
    return deque(maxlen=capacity)


def sample_batch(experience_replay, batch_size):
    """Sample a random minibatch (uniform, without replacement) from experience_replay as tensors."""
    batch = random.sample(experience_replay, batch_size)
    return (torch.as_tensor(np.array([e.state for e in batch]), dtype=torch.float32),
            torch.as_tensor([e.action for e in batch], dtype=torch.int64).unsqueeze(1),
            torch.as_tensor([e.reward for e in batch], dtype=torch.float32).unsqueeze(1),
            torch.as_tensor(np.array([e.next_state for e in batch]), dtype=torch.float32),
            torch.as_tensor([e.terminated for e in batch], dtype=torch.float32).unsqueeze(1))


def sample_action(q_net, state, epsilon, n_actions):
    """Decaying epsilon-greedy action selection (epsilon = 0 gives the greedy policy)."""
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        q = q_net(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
    return int(q.argmax(dim=1).item())


def dqn_target(q_net, target_net, r, s2, terminated, gamma):
    """Standard DQN target: y = r + gamma * (1 - terminated) * max_a Q(s', a; theta^-).
    Computed without gradients, so only the online network is trained."""
    with torch.no_grad():
        max_next_q = target_net(s2).max(dim=1, keepdim=True)[0]
    return r + gamma * (1.0 - terminated) * max_next_q


def optimize_batch(q_net, target_net, optimizer, loss_fn, batch, target_fn, hp):
    """One gradient step on MSE(y, Q(s, a; theta)); updates the online network only."""
    s, a, r, s2, d = batch
    y = target_fn(q_net, target_net, r, s2, d, hp["gamma"])
    q_sa = q_net(s).gather(1, a)
    loss = loss_fn(q_sa, y)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), hp["grad_clip"])
    optimizer.step()
    return loss.item()


def train_agent(n_hidden_layers, hp=HP, seed=0, target_fn=dqn_target, log_dir=None, verbose=True,
                net_cls=None):
    """Train with the DQN algorithm of figure 4. Stops once the 100-episode average reaches 475."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = gym.make(ENV_ID)
    env.action_space.seed(seed)
    state_dim, n_actions = env.observation_space.shape[0], env.action_space.n

    net_cls = net_cls or QNetwork
    q_net = net_cls(state_dim, n_actions, n_hidden_layers, hp["hidden_size"])
    target_net = net_cls(state_dim, n_actions, n_hidden_layers, hp["hidden_size"])
    target_net.load_state_dict(q_net.state_dict())      # theta^- = theta
    optimizer = torch.optim.Adam(q_net.parameters(), lr=hp["lr"])
    loss_fn = nn.MSELoss()
    experience_replay = make_experience_replay(hp["replay_size"])
    writer = SummaryWriter(log_dir) if log_dir else None

    episode_rewards, losses, eps_log, lr_log = [], [], [], []
    epsilon, total_steps, solved_at = hp["eps_start"], 0, None
    start = time.time()

    try:
        for ep in range(hp["max_episodes"]):
            state, _ = env.reset(seed=seed * 10_000 + ep)
            eps_log.append(epsilon)
            lr_log.append(optimizer.param_groups[0]["lr"])
            ep_reward, done = 0.0, False
            while not done:
                action = sample_action(q_net, state, epsilon, n_actions)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                # Only a real failure is terminal; the 500-step time limit (truncation) still bootstraps.
                experience_replay.append(Experience(state, action, reward, next_state, float(terminated)))
                state = next_state
                ep_reward += reward
                total_steps += 1

                if len(experience_replay) >= hp["min_replay"]:
                    loss = optimize_batch(q_net, target_net, optimizer, loss_fn,
                                          sample_batch(experience_replay, hp["batch_size"]), target_fn, hp)
                    losses.append(loss)
                    if writer:
                        writer.add_scalar("loss_per_training_step", loss, len(losses))

                if total_steps % hp["target_update"] == 0:
                    target_net.load_state_dict(q_net.state_dict())   # every C environment steps

            episode_rewards.append(ep_reward)
            epsilon = max(hp["eps_min"], epsilon * hp["eps_decay"])
            for g in optimizer.param_groups:              # decay the learning rate per episode
                g["lr"] = max(hp["lr_min"], g["lr"] * hp["lr_decay"])
            avg100 = np.mean(episode_rewards[-100:])
            if writer:
                writer.add_scalar("reward_per_episode", ep_reward, ep)
                writer.add_scalar("avg_reward_last100", avg100, ep)
                writer.add_scalar("epsilon", eps_log[-1], ep)
                writer.add_scalar("learning_rate", lr_log[-1], ep)
            if verbose and (ep + 1) % 25 == 0:
                print(f"ep {ep+1:4d} | reward {ep_reward:5.0f} | avg100 {avg100:6.1f} | "
                      f"eps {epsilon:.3f} | steps {total_steps} | {time.time()-start:5.0f}s", flush=True)
            if len(episode_rewards) >= 100 and avg100 >= SOLVED_SCORE:
                solved_at = ep + 1
                if verbose:
                    print(f"Solved: average reward {avg100:.1f} over episodes {solved_at-99}-{solved_at}")
                break
    finally:
        env.close()
        if writer:
            writer.close()
    log = dict(epsilon=eps_log, lr=lr_log, env_steps=total_steps, train_seconds=time.time() - start)
    return q_net, episode_rewards, losses, solved_at, log


def test_agent(q_net, n_episodes=100, render=False, seed=TEST_SEED):
    """Run the trained agent greedily on new episodes (seeds never used in training)."""
    env = gym.make(ENV_ID, render_mode="human" if render else None)
    rewards = []
    try:
        for i in range(n_episodes):
            state, _ = env.reset(seed=seed + i)
            total, done = 0.0, False
            while not done:
                action = sample_action(q_net, state, 0.0, env.action_space.n)
                state, r, term, trunc, _ = env.step(action)
                total += r
                done = term or trunc
            rewards.append(total)
    finally:
        env.close()
    return rewards


def save_agent(q_net, path, n_hidden_layers, hp, seed, algorithm):
    torch.save(dict(state_dict=q_net.state_dict(), architecture=type(q_net).__name__,
                    state_dim=4, n_actions=2, n_hidden_layers=n_hidden_layers,
                    hidden_size=hp["hidden_size"], hp=hp, seed=seed, algorithm=algorithm), path)


def load_agent(path):
    """Rebuild a saved network from its checkpoint metadata."""
    ckpt = torch.load(path, weights_only=False)
    if ckpt["architecture"] == "DuelingQNetwork":
        from section3_dueling_dqn import DuelingQNetwork as cls
    else:
        cls = QNetwork
    net = cls(ckpt["state_dim"], ckpt["n_actions"], ckpt["n_hidden_layers"], ckpt["hidden_size"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt


def first_solved_episode(rewards):
    """Independent check of solved_at: first window of 100 consecutive episodes with mean >= 475."""
    for end in range(100, len(rewards) + 1):
        if np.mean(rewards[end - 100:end]) >= SOLVED_SCORE:
            return end
    return None


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


def export_csv(prefix, rewards, losses, eps=None, lrs=None):
    with open(prefix + "_episodes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward", "avg_last100", "epsilon", "learning_rate"])
        for i, r in enumerate(rewards):
            w.writerow([i + 1, r, round(float(np.mean(rewards[max(0, i - 99):i + 1])), 2),
                        eps[i] if eps else "", lrs[i] if lrs else ""])
    with open(prefix + "_loss.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["training_step", "loss"])
        w.writerows(enumerate(losses, start=1))


def run(name, n_hidden_layers, out_dir, target_fn, seed, render, net_cls=None, hp=None, tag=""):
    hp = dict(hp or HP)
    os.makedirs(out_dir, exist_ok=True)
    run_id = f"{n_hidden_layers}L_seed{seed}{tag}"
    prefix = os.path.join(out_dir, run_id)
    q_net, rewards, losses, solved_at, log = train_agent(
        n_hidden_layers, hp=hp, seed=seed, target_fn=target_fn, net_cls=net_cls,
        log_dir=os.path.join("runs", f"{os.path.basename(out_dir)}_{run_id}"))
    assert solved_at == first_solved_episode(rewards), "solved_at does not match the 100-episode window"
    test_rewards = test_agent(q_net, render=False)
    print(f"Episodes to reach avg >= 475 over 100 episodes: {solved_at}")
    print(f"Greedy test over 100 new episodes: mean {np.mean(test_rewards):.1f}, min {np.min(test_rewards):.0f}")
    plot_results(rewards, losses, f"{name} ({n_hidden_layers} hidden layers)", prefix)
    save_agent(q_net, prefix + "_model.pt", n_hidden_layers, hp, seed, name)
    with open(prefix + "_data.json", "w") as f:
        json.dump(dict(rewards=rewards, losses=losses, solved_at=solved_at,
                       test_rewards=test_rewards), f)
    with open(prefix + "_config.json", "w") as f:
        json.dump(dict(algorithm=name, architecture=type(q_net).__name__, n_hidden_layers=n_hidden_layers,
                       n_parameters=sum(p.numel() for p in q_net.parameters()), seed=seed,
                       target=target_fn.__name__, hp=hp, env=ENV_ID, test_seeds=[TEST_SEED, TEST_SEED + 99],
                       env_steps=log["env_steps"], training_steps=len(losses), solved_at=solved_at,
                       test_mean=float(np.mean(test_rewards)), test_min=float(np.min(test_rewards)),
                       torch=torch.__version__, gymnasium=gym.__version__), f, indent=2)
    export_csv(prefix, rewards, losses, log["epsilon"], log["lr"])
    if render:
        test_agent(q_net, n_episodes=3, render=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=3, choices=[3, 5])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true", help="watch the agent after training")
    parser.add_argument("--constant-lr", action="store_true",
                        help="ablation: constant learning rate 5e-4 (no decay)")
    parser.add_argument("--lr", type=float, default=None,
                        help="learning-rate experiment: initial learning rate (default 1e-3)")
    parser.add_argument("--lr-decay", type=float, default=None,
                        help="learning-rate experiment: decay per episode (1.0 = constant; default 0.995)")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--load", help="test a saved _model.pt instead of training")
    args = parser.parse_args()
    torch.set_num_threads(1)

    if args.load:
        net, ckpt = load_agent(args.load)
        rewards = test_agent(net, render=args.render, n_episodes=3 if args.render else 100)
        print(f"{ckpt['algorithm']}, {ckpt['n_hidden_layers']} hidden layers: "
              f"greedy test mean {np.mean(rewards):.1f}, min {np.min(rewards):.0f}")
        return

    hp, tag = dict(HP), ""
    if args.constant_lr:
        hp.update(lr=5e-4, lr_decay=1.0)
        tag = "_constlr"
    if args.lr is not None or args.lr_decay is not None:
        if args.lr is not None:
            hp["lr"] = args.lr
        if args.lr_decay is not None:
            hp["lr_decay"] = args.lr_decay
        tag += f"_lr{hp['lr']:g}_decay{hp['lr_decay']:g}"
    if args.max_episodes:
        hp["max_episodes"] = args.max_episodes
        tag += f"_max{args.max_episodes}"
    run("DQN", args.layers, os.path.join("results", "section2"), dqn_target, args.seed, args.render,
        hp=hp, tag=tag)


if __name__ == "__main__":
    main()
