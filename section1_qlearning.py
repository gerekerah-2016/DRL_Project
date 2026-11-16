"""
Section 1 - Tabular Q-learning on FrozenLake (Gymnasium).

Note: the assignment names "FrozenLake-v0", which no longer exists in Gymnasium.
"FrozenLake-v1" is the same 4x4 slippery lake and is used here.

Usage:
    python section1_qlearning.py            # train with the final hyper-parameters
    python section1_qlearning.py --search   # run the hyper-parameter search first
Outputs (in ./results/section1): Q-table colormaps, reward plot, steps plot,
and TensorBoard logs in ./runs/section1.
"""
import argparse
import itertools
import os

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

N_EPISODES = 5000
MAX_STEPS = 100
OUT_DIR = os.path.join("results", "section1")
ACTION_NAMES = ["Left", "Down", "Right", "Up"]

# Final hyper-parameters (chosen with --search, see report)
FINAL_HP = dict(alpha=0.1, gamma=0.99, eps_start=1.0, eps_min=0.01, eps_decay=0.999)


def sample_action(q_table, state, epsilon, rng):
    """Decaying epsilon-greedy: random action w.p. epsilon, else greedy (random tie-break)."""
    if rng.random() < epsilon:
        return int(rng.integers(q_table.shape[1]))
    best = np.flatnonzero(q_table[state] == q_table[state].max())
    return int(rng.choice(best))


def train(alpha, gamma, eps_start, eps_min, eps_decay, seed=0,
          snapshot_steps=(500, 2000), snapshot_episodes=(500, 2000), writer=None):
    env = gym.make("FrozenLake-v1", max_episode_steps=MAX_STEPS)
    rng = np.random.default_rng(seed)
    n_s, n_a = env.observation_space.n, env.action_space.n
    q_table = np.zeros((n_s, n_a))                    # initialize with zeros

    rewards, steps_to_goal = [], []
    snaps_steps, snaps_eps = {}, {}
    epsilon, total_steps = eps_start, 0

    for ep in range(N_EPISODES):
        state, _ = env.reset(seed=seed + ep)
        ep_reward, reached_goal, t = 0.0, False, 0
        for t in range(1, MAX_STEPS + 1):
            action = sample_action(q_table, state, epsilon, rng)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Q-learning update (figure 3): target = r if s' terminal, else r + gamma * max_a Q(s', a)
            target = reward if terminated else reward + gamma * np.max(q_table[next_state])
            q_table[state, action] += alpha * (target - q_table[state, action])

            total_steps += 1
            if total_steps in snapshot_steps:
                snaps_steps[total_steps] = q_table.copy()
            ep_reward += reward
            state = next_state
            if terminated or truncated:
                reached_goal = reward > 0
                break

        rewards.append(ep_reward)
        steps_to_goal.append(t if reached_goal else MAX_STEPS)   # 100 if goal not reached
        if ep + 1 in snapshot_episodes:
            snaps_eps[ep + 1] = q_table.copy()
        epsilon = max(eps_min, epsilon * eps_decay)                # decay per episode

        if writer is not None:
            writer.add_scalar("reward_per_episode", ep_reward, ep)
            writer.add_scalar("epsilon", epsilon, ep)
            if (ep + 1) % 100 == 0:
                writer.add_scalar("avg_steps_to_goal_last100", np.mean(steps_to_goal[-100:]), ep + 1)
    env.close()
    return q_table, np.array(rewards), np.array(steps_to_goal), snaps_steps, snaps_eps


def evaluate_greedy(q_table, n=1000, seed=10_000):
    """Success rate of the greedy policy (no exploration)."""
    env = gym.make("FrozenLake-v1", max_episode_steps=MAX_STEPS)
    wins = 0
    for i in range(n):
        s, _ = env.reset(seed=seed + i)
        while True:
            s, r, term, trunc, _ = env.step(int(np.argmax(q_table[s])))
            if term or trunc:
                wins += r > 0
                break
    env.close()
    return wins / n


def hp_search():
    grid = dict(alpha=[0.05, 0.1, 0.2], gamma=[0.95, 0.99], eps_decay=[0.995, 0.999])
    results = []
    for alpha, gamma, decay in itertools.product(*grid.values()):
        scores = []
        for seed in range(3):
            q, *_ = train(alpha, gamma, 1.0, 0.01, decay, seed=seed)
            scores.append(evaluate_greedy(q, n=500))
        results.append((np.mean(scores), alpha, gamma, decay))
        print(f"alpha={alpha:<5} gamma={gamma:<5} decay={decay:<6} greedy success={np.mean(scores):.3f}")
    results.sort(reverse=True)
    print("Best:", results[0])
    return results


def plot_q_table(q_table, title, path):
    fig, ax = plt.subplots(figsize=(5.5, 7))
    im = ax.imshow(q_table, cmap="viridis", aspect="auto")
    for s in range(q_table.shape[0]):
        for a in range(q_table.shape[1]):
            v = q_table[s, a]
            ax.text(a, s, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if v < q_table.max() * 0.6 else "black")
    ax.set_xticks(range(4), ACTION_NAMES)
    ax.set_yticks(range(q_table.shape[0]))
    ax.set_xlabel("Action")
    ax.set_ylabel("State")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.05)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", action="store_true", help="run hyper-parameter search first")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.search:
        hp_search()

    writer = SummaryWriter(os.path.join("runs", "section1"))
    q, rewards, steps, snaps_steps, snaps_eps = train(**FINAL_HP, seed=args.seed, writer=writer)
    writer.close()

    for k, tbl in snaps_steps.items():
        plot_q_table(tbl, f"Q-table after {k} steps", os.path.join(OUT_DIR, f"qtable_{k}_steps.png"))
    for k, tbl in snaps_eps.items():
        plot_q_table(tbl, f"Q-table after {k} episodes", os.path.join(OUT_DIR, f"qtable_{k}_episodes.png"))
    plot_q_table(q, "Final Q-table (5000 episodes)", os.path.join(OUT_DIR, "qtable_final.png"))

    # Reward per episode (raw 0/1 + 100-episode moving average for readability)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rewards, ".", ms=1.5, alpha=0.4, label="reward per episode")
    ma = np.convolve(rewards, np.ones(100) / 100, mode="valid")
    ax.plot(np.arange(99, len(rewards)), ma, color="C3", lw=2, label="moving average (100)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Reward"); ax.set_title("FrozenLake - reward per episode")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "reward_per_episode.png"), dpi=150)
    plt.close(fig)

    # Average steps to goal over last 100 episodes, plotted every 100 episodes
    x = np.arange(100, N_EPISODES + 1, 100)
    avg_steps = [steps[i - 100:i].mean() for i in x]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, avg_steps, "o-")
    ax.set_xlabel("Episode"); ax.set_ylabel("Avg. steps to goal (last 100)")
    ax.set_title("FrozenLake - average steps to goal (100 if goal not reached)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "avg_steps_to_goal.png"), dpi=150)
    plt.close(fig)

    success = evaluate_greedy(q)
    print(f"Success rate in last 100 training episodes: {rewards[-100:].mean():.2f}")
    print(f"Avg steps to goal, last 100 episodes: {avg_steps[-1]:.1f}")
    print(f"Greedy policy success rate over 1000 test episodes: {success:.3f}")
    np.save(os.path.join(OUT_DIR, "q_table_final.npy"), q)


if __name__ == "__main__":
    main()
