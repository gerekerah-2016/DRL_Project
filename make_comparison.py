"""Builds the Section 2 vs Section 3 comparison figure and a summary table (results/summary.json)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

runs = {
    "DQN, 3 hidden layers": "results/section2/3L_seed0_data.json",
    "DQN, 5 hidden layers": "results/section2/5L_seed0_data.json",
    "Double DQN, 3 hidden layers": "results/section3/3L_seed0_data.json",
    "Double DQN, 5 hidden layers": "results/section3/5L_seed0_data.json",
    "Dueling DQN, 3 hidden layers": "results/section3_dueling/3L_seed0_data.json",
    "Dueling DQN, 5 hidden layers": "results/section3_dueling/5L_seed0_data.json",
}
styles = ["C0-", "C1-", "C0:", "C1:", "C2-", "C3-"]
summary = {}
fig, ax = plt.subplots(figsize=(8, 4.5))
for (name, path), st in zip(runs.items(), styles):
    if not os.path.exists(path):
        continue
    d = json.load(open(path))
    r = d["rewards"]
    avg = [np.mean(r[max(0, i - 99):i + 1]) for i in range(len(r))]
    ax.plot(avg, st, lw=2, label=f"{name} (" + (f"solved at episode {d['solved_at']}" if d['solved_at'] else 'not solved in 1000 episodes') + ")")
    summary[name] = dict(solved_at=d["solved_at"], episodes_run=len(r),
                         training_steps=len(d["losses"]),
                         test_mean=float(np.mean(d["test_rewards"])),
                         test_min=float(np.min(d["test_rewards"])),
                         final_loss_avg=float(np.mean(d["losses"][-1000:])))
ax.axhline(475, color="gray", ls=":", label="475")
ax.set_xlabel("Episode"); ax.set_ylabel("Average reward of last 100 episodes")
ax.set_title("DQN vs Dueling DQN vs Double DQN - 100-episode average reward")
ax.legend(loc="upper left", fontsize=8); fig.tight_layout()
fig.savefig("results/comparison_all.png", dpi=150)
json.dump(summary, open("results/summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
