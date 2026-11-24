"""
Collects every CartPole run in results/, writes results/summary.json and results/summary.csv
(per run and mean/std over seeds), and draws results/comparison_all.png (seed-0 runs).
Missing runs are reported as warnings, not errors.

Usage: python make_comparison.py
"""
import csv
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ALGOS = {"section2": "DQN", "section3_dueling": "Dueling DQN", "section3": "Double DQN"}
STYLES = {("DQN", 3): "C0-", ("DQN", 5): "C1-", ("Double DQN", 3): "C0:", ("Double DQN", 5): "C1:",
          ("Dueling DQN", 3): "C2-", ("Dueling DQN", 5): "C3-"}


def load_runs():
    runs = []
    for folder, algo in ALGOS.items():
        for path in sorted(glob.glob(f"results/{folder}/*_data.json")):
            m = re.match(r"(\d)L_seed(\d+)(.*)_data\.json", os.path.basename(path))
            if not m:
                continue
            d = json.load(open(path))
            avg = [np.mean(d["rewards"][max(0, i - 99):i + 1]) for i in range(len(d["rewards"]))]
            runs.append(dict(algorithm=algo, layers=int(m.group(1)), seed=int(m.group(2)),
                             variant=m.group(3).lstrip("_") or "final", solved_at=d["solved_at"],
                             episodes_run=len(d["rewards"]), training_steps=len(d["losses"]),
                             max_avg100=float(max(avg)), max_avg100_episode=int(np.argmax(avg)) + 1,
                             test_mean=float(np.mean(d["test_rewards"])),
                             test_min=float(np.min(d["test_rewards"])), rewards=d["rewards"]))
    return runs


def main():
    runs = load_runs()
    if not runs:
        print("No CartPole results found in results/. Run section2_dqn.py / section3_*.py first.")
        return
    for algo in ("DQN", "Dueling DQN", "Double DQN"):
        for L in (3, 5):
            if not any(r["algorithm"] == algo and r["layers"] == L and r["variant"] == "final" for r in runs):
                print(f"Warning: no results for {algo}, {L} hidden layers")

    groups = {}
    for r in runs:
        if r["variant"] == "final":
            groups.setdefault((r["algorithm"], r["layers"]), []).append(r)
    aggregate = []
    for (algo, L), rs in sorted(groups.items()):
        solved = [r["solved_at"] for r in rs if r["solved_at"] is not None]
        aggregate.append(dict(
            algorithm=algo, layers=L, seeds=[r["seed"] for r in rs], n_seeds=len(rs),
            n_solved=len(solved), solved_at=[r["solved_at"] for r in rs],
            solved_mean=float(np.mean(solved)) if solved else None,
            solved_std=float(np.std(solved, ddof=1)) if len(solved) > 1 else None,
            test_mean=float(np.mean([r["test_mean"] for r in rs]))))

    per_run = [{k: v for k, v in r.items() if k != "rewards"} for r in runs]
    json.dump(dict(runs=per_run, aggregate=aggregate), open("results/summary.json", "w"), indent=2)
    with open("results/summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_run[0].keys()))
        w.writeheader(); w.writerows(per_run)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for r in runs:
        if r["seed"] != 0 or r["variant"] != "final":
            continue
        avg = [np.mean(r["rewards"][max(0, i - 99):i + 1]) for i in range(len(r["rewards"]))]
        status = f"solved at episode {r['solved_at']}" if r["solved_at"] else "not solved in 1000 episodes"
        ax.plot(avg, STYLES[(r["algorithm"], r["layers"])], lw=2,
                label=f"{r['algorithm']}, {r['layers']} hidden layers ({status})")
    ax.axhline(475, color="gray", ls=":", label="475")
    ax.set_xlabel("Episode"); ax.set_ylabel("Average reward of last 100 episodes")
    ax.set_title("DQN vs Dueling DQN vs Double DQN (seed 0) - 100-episode average reward")
    ax.legend(loc="upper left", fontsize=8); fig.tight_layout()
    fig.savefig("results/comparison_all.png", dpi=150)
    plt.close(fig)

    for a in aggregate:
        print(a)
    for r in per_run:
        if r["variant"] != "final":
            print("variant:", r)


if __name__ == "__main__":
    main()
