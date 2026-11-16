const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, WidthType, BorderStyle, ShadingType, PageBreak,
  LevelFormat, Footer, PageNumber,
} = require("docx");

const S = JSON.parse(fs.readFileSync("results/summary.json"));
const F = "Calibri";
const fmt = (v) => (v === null || v === undefined ? "not reached in 1000" : String(v));
const r = (name) => S[name] || {};

// ---------- helpers ----------
function runs(text) {
  // **bold** and *italic* markup inside a string
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: F, size: 24 }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: F, size: 24 }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, font: F, size: 24 }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: F, size: 24 }));
  return out;
}
const P = (text, opts = {}) => new Paragraph({ children: runs(text), spacing: { after: 100 }, ...opts });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: F })] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: F })] });
const Q = (label, question, answer) => [
  new Paragraph({ children: [new TextRun({ text: `${label} `, bold: true, font: F, size: 24 }),
    new TextRun({ text: question, bold: true, font: F, size: 24 })], spacing: { before: 120, after: 60 }, keepNext: true }),
  P(answer),
];
const bullet = (text) => new Paragraph({ children: runs(text), numbering: { reference: "bullets", level: 0 }, spacing: { after: 40 } });

const border = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(header, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (t, i, head) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    shading: head ? { fill: "D9E2F3", type: ShadingType.CLEAR } : undefined,
    margins: { top: 40, bottom: 40, left: 100, right: 100 },
    children: [new Paragraph({ children: [new TextRun({ text: String(t), bold: head, font: F, size: 22 })] })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, true)) }),
      ...rows.map((row) => new TableRow({ cantSplit: true, children: row.map((c, i) => cell(c, i, false)) }))],
  });
}
const gap = () => new Paragraph({ children: [], spacing: { after: 60 } });

function pngSize(path) {
  const b = fs.readFileSync(path);
  return [b.readUInt32BE(16), b.readUInt32BE(20)];
}
function figure(path, caption, widthIn) {
  const [w, h] = pngSize(path);
  const wpx = widthIn * 96, hpx = (wpx * h) / w;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, children: [new ImageRun({
      type: "png", data: fs.readFileSync(path), transformation: { width: wpx, height: hpx },
      altText: { title: caption, description: caption, name: caption } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: caption, italics: true, font: F, size: 20 })] }),
  ];
}

// ---------- Section 1 HP search table ----------
const searchRows = fs.readFileSync("s1_log.txt", "utf8").split("\n")
  .filter((l) => l.startsWith("alpha="))
  .map((l) => { const m = l.match(/alpha=(\S+)\s+gamma=(\S+)\s+decay=(\S+)\s+greedy success=(\S+)/); return [m[1], m[2], m[3], m[4]]; });
const s1 = fs.readFileSync("s1_log.txt", "utf8");
const s1Greedy = s1.match(/Greedy policy success rate over 1000 test episodes: (\S+)/)[1];
const s1Last100 = s1.match(/Success rate in last 100 training episodes: (\S+)/)[1];
const s1Steps = s1.match(/Avg steps to goal, last 100 episodes: (\S+)/)[1];

const D3 = r("DQN, 3 hidden layers"), D5 = r("DQN, 5 hidden layers");
const DD3 = r("Double DQN, 3 hidden layers"), DD5 = r("Double DQN, 5 hidden layers");
const DU3 = r("Dueling DQN, 3 hidden layers"), DU5 = r("Dueling DQN, 5 hidden layers");
const DUELING_5L_TEXT = process.env.DUELING_5L_TEXT || "";

// ---------- body ----------
const body = [
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Deep Reinforcement Learning", bold: true, font: F, size: 32 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Assignment 1 \u2013 From Q-learning to Deep Q-learning (DQN)", bold: true, font: F, size: 28 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: "Ben-Gurion University of the Negev \u2013 Department of Software and Information Systems Engineering", font: F, size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Student 1: [Full name], ID [000000000]     Student 2: [Full name], ID [000000000]", font: F, size: 24, highlight: "yellow" })] }),
  gap(),

  H1("How to run the code"),
  P("Python 3.10+ is required. Install the dependencies with **pip install -r requirements.txt** (gymnasium, torch, numpy, matplotlib, tensorboard). Run each script from the folder that contains it:"),
  bullet("**python section1_qlearning.py** \u2013 Q-learning on FrozenLake (add **--search** to repeat the hyper-parameter search)."),
  bullet("**python section2_dqn.py --layers 3** and **python section2_dqn.py --layers 5** \u2013 DQN on CartPole-v1 with the 3- and 5-hidden-layer networks."),
  bullet("**python section3_dueling_dqn.py --layers 3** (or **--layers 5**) \u2013 Dueling DQN on CartPole-v1, the Section 3 improvement."),
  bullet("**python section3_double_dqn.py --layers 3** (or **--layers 5**) \u2013 Double DQN, an extra Section 3 experiment."),
  bullet("**python make_comparison.py** \u2013 comparison figure of all CartPole runs. Add **--render** to the DQN scripts to watch the trained agent."),
  P("Plots are saved in **results/** and TensorBoard logs in **runs/** (view them with **tensorboard --logdir runs**). All runs use seed 0."),

  H1("Section 1 \u2013 Tabular Q-learning"),
  ...Q("1.", "Why can\u2019t Value-Iteration be used when the dynamics are unknown?",
    "Value-Iteration repeatedly applies the Bellman optimality backup V(s) \u2190 max_a \u03a3_{s\u2032} P(s\u2032|s,a)[R(s,a,s\u2032) + \u03b3V(s\u2032)]. This backup needs the full model of the environment: the transition probabilities P(s\u2032|s,a) and the reward function R. In a model-free setting these are unknown (or too complex to write down), so the expectation over next states simply cannot be computed. In addition, it sweeps over every state in every iteration, which is infeasible for large state spaces."),
  ...Q("2.", "How do model-free methods resolve this problem?",
    "Instead of computing the expectation over next states with a model, they **estimate it from experience**: the agent interacts with the environment and every observed transition (s, a, r, s\u2032) is used as a sample of that expectation. The estimate is updated incrementally with a learning rate (temporal-difference update), so averaging over many samples converges to the true expected value. They also learn action values Q(s,a) rather than V(s), so a policy can be derived by argmax over actions without knowing the model."),
  ...Q("3.", "What is the main difference between SARSA and Q-learning?",
    "SARSA is **on-policy**: its target R + \u03b3Q(S\u2032,A\u2032) uses the action A\u2032 that the agent actually takes next with its behaviour (e.g., \u03b5-greedy) policy. Q-learning is **off-policy**: its target R + \u03b3max_a Q(S\u2032,a) uses the greedy action, regardless of which action is taken next. Therefore SARSA learns the value of the exploring policy it follows, including the cost of exploration, and tends to learn safer behaviour (e.g., staying away from a cliff). Q-learning learns the optimal greedy values Q* directly, while it still explores."),
  ...Q("4.", "Why is decaying \u03b5-greedy better than acting greedily?",
    "Acting greedily is pure exploitation: because the Q estimates start inaccurate (here all zeros), the agent locks onto the first actions that seem good and never tries the others, so it can converge to a sub-optimal policy. With \u03b5-greedy, every action keeps a non-zero probability of being tried, which guarantees continued exploration of all state-action pairs (a condition for Q-learning convergence). Decaying \u03b5 explores a lot at the start, when estimates are poor, and gradually exploits more as the estimates become reliable, approaching the greedy policy at the end."),

  H2("Implementation and hyper-parameters"),
  P("The agent keeps a 16\u00d74 Q-table initialized with zeros and follows the algorithm of figure 3: the target is r for a terminal next state and r + \u03b3max_a Q(s\u2032,a) otherwise. Actions are chosen with decaying \u03b5-greedy (random tie-breaking among equal Q-values). It trains for 5000 episodes with at most 100 steps per episode. The assignment names FrozenLake-v0, which no longer exists in Gymnasium; FrozenLake-v1 is the same 4\u00d74 slippery lake and was used. Hyper-parameters were optimized with a grid search (3 seeds per setting, scored by the success rate of the final greedy policy on 500 test episodes):"),
  table(["\u03b1 (learning rate)", "\u03b3 (discount)", "\u03b5 decay / episode", "Greedy success rate"], searchRows, [2250, 2250, 2250, 2250]),
  gap(),
  P(`**Final hyper-parameters:** \u03b1 = 0.1, \u03b3 = 0.99, \u03b5 starts at 1.0 and is multiplied by 0.999 after every episode down to 0.01. This setting is tied for the best score; the slow decay matters most, since fast decay (0.995) with a small \u03b1 stops exploring before the values propagate back from the goal. A high \u03b3 is also important, because the only reward is at the goal.`),
  H2("Results"),
  P(`The final greedy policy reaches the goal in **${(parseFloat(s1Greedy) * 100).toFixed(1)}%** of 1000 test episodes, close to the ~74% that is optimal for the slippery 4\u00d74 lake (the ice moves the agent in an unintended direction 2/3 of the time). In the last 100 training episodes (still with \u03b5 = 0.01 exploration) the success rate was ${s1Last100} and the average number of steps was ${s1Steps}. The Q-table after 500 steps, after 2000 steps and the final table are shown in Figures A1\u2013A3; Figures A6\u2013A7 add the tables after 500 and 2000 *episodes* in case episodes were intended. After 500 and 2000 steps most values are still zero because the goal is rarely reached early on; in the final table values grow toward the goal (state 15), and holes (5, 7, 11, 12) and the goal itself stay zero because they are terminal. The reward per episode is in Figure A4 and the average steps to the goal over the last 100 episodes is in Figure A5: the success rate rises from ~0 to ~0.6 and the average number of steps drops from ~100 to ~60.`),

  H1("Section 2 \u2013 Deep Q-learning"),
  ...Q("1.", "Why do we sample from the experience replay in random order?",
    "Consecutive transitions from the same trajectory are strongly correlated, while stochastic gradient descent assumes independent, identically distributed samples. Training on consecutive transitions gives highly correlated, biased gradients, makes the network overfit to the most recent part of the state space and forget the rest, and can cause oscillation or divergence. Sampling a random minibatch from a large buffer breaks these correlations, makes the data much closer to i.i.d., and averages over many past policies, which smooths changes in the data distribution. It also lets every transition be reused many times, improving data efficiency."),
  ...Q("2.", "How does using an older set of weights for the targets improve the model?",
    "Without it, the target r + \u03b3max_a Q(s\u2032,a;\u03b8) depends on the same weights that are being updated, so every gradient step also moves the target: the network chases a moving target, which creates feedback loops, oscillations and divergence. With a separate target network \u03b8\u207b that is copied from \u03b8 only every C steps, the targets stay fixed for C steps. Each period then becomes a stable regression problem, much like supervised learning, which makes training significantly more stable."),

  H2("Implementation"),
  P("The script contains the required components: **QNetwork** (state in, Q-value of each action out; one class is used for both the Q-network and the target network, with different weights), **experience_replay** (a fixed-size deque of (s, a, r, s\u2032, done) transitions), **sample_batch** (a uniformly random minibatch), **sample_action** (decaying \u03b5-greedy), **train_agent** (the algorithm of figure 4 with MSE loss between the target and Q(s,a;\u03b8), and \u03b8\u207b \u2190 \u03b8 every C steps) and **test_agent** (greedy play on new episodes, optionally rendered). Only real failures are treated as terminal; reaching the 500-step limit is a truncation, so its target still bootstraps. Training stops as soon as the average reward of the last 100 episodes reaches 475."),
  P("**Network structures** (ReLU after every hidden layer, linear output): the 3-hidden-layer network is 4 \u2192 128 \u2192 128 \u2192 128 \u2192 2 (33,922 parameters) and the 5-hidden-layer network is 4 \u2192 128 \u2192 128 \u2192 128 \u2192 128 \u2192 128 \u2192 2 (66,946 parameters)."),
  table(["Hyper-parameter", "Value", "Hyper-parameter", "Value"], [
    ["Optimizer", "Adam", "Batch size", "64"],
    ["Initial learning rate", "1e-3", "Replay capacity", "50,000"],
    ["Learning-rate decay", "\u00d70.995 per episode (min 1e-5)", "Learning starts after", "1,000 transitions"],
    ["Discount \u03b3", "0.99", "Target update C", "every 500 steps"],
    ["\u03b5 start / min", "1.0 / 0.01", "\u03b5 decay", "\u00d70.98 per episode"],
    ["Loss", "MSE", "Gradient clipping", "max norm 10"],
  ], [2400, 2100, 2400, 2100]),
  gap(),
  P("**Most influential hyper-parameter \u2013 the learning rate.** With a constant learning rate of 5e-4 the agent repeatedly reached the maximum of 500 but then collapsed (catastrophic forgetting: once the agent is good, the replay buffer fills with \u201cbalanced\u201d states only, and large updates destroy what it learned), so its 100-episode average peaked at ~428 and never reached 475 in 1000 episodes. Decaying the learning rate by 0.995 per episode let the agent learn quickly at the start and then settle on the good policy, which solved the task. The \u03b5 decay was the next most important: decaying too slowly wastes episodes on random actions."),
  H2("Results"),
  table(["Network", "Episodes until avg \u2265 475 over 100 episodes", "Greedy test (100 new episodes): mean / min"], [
    ["3 hidden layers", fmt(D3.solved_at), `${D3.test_mean?.toFixed(1)} / ${D3.test_min}`],
    ["5 hidden layers", fmt(D5.solved_at), `${D5.test_mean?.toFixed(1)} / ${D5.test_min}`],
  ], [2400, 3500, 3100]),
  gap(),
  P(`The 3-layer network solved the task after **${fmt(D3.solved_at)} episodes** and the 5-layer network after **${fmt(D5.solved_at)} episodes** (the 100-episode window ending at that episode has an average of at least 475). Both trained agents are near-perfect on new episodes. For a 4-dimensional problem like CartPole, the larger network adds parameters without adding useful capacity, and was slower and less stable in this run. The reward per episode is shown in Figures B1 and B3, and the loss at every training step in Figures B2 and B4. Unlike supervised learning, the DQN loss does not decrease steadily: as the agent survives longer, the target Q-values grow (toward ~1/(1\u2212\u03b3) = 100), so the size of the TD errors and the loss grow as well, and they jump every time the target network is updated.`),

  H1("Section 3 \u2013 Improved DQN: Dueling DQN"),
  H2("The improvement and its purpose"),
  P("Standard DQN outputs Q(s,a) for every action directly. **Dueling DQN** (Wang et al., 2016) splits the network, after a shared trunk, into two streams: a **value stream** V(s), which estimates how good the state is, and an **advantage stream** A(s,a), which estimates how much better each action is than the others. They are combined as Q(s,a) = V(s) + A(s,a) \u2212 mean_a\u2032 A(s,a\u2032); subtracting the mean makes V and A identifiable. The purpose is faster and more stable value learning: in many states the choice of action matters little (in CartPole, when the pole is nearly upright both pushes are almost equally good), and with a separate V(s) every update improves the value of the state for all actions at once, not only for the action that was taken. Only the network changes; the algorithm, the standard DQN target and all hyper-parameters are those of Section 2, so the comparison isolates the effect of the architecture."),
  H2("Hyper-parameters and network structure"),
  P("All hyper-parameters are identical to the Section 2 table (Adam, initial learning rate 1e-3 decayed \u00d70.995 per episode, \u03b3 = 0.99, batch 64, replay 50,000, C = 500 steps, \u03b5 1.0 \u2192 0.01 with \u00d70.98 per episode, MSE loss, gradient clipping 10). The networks use ReLU and keep the same number of hidden layers as in Section 2 on every path from input to output. **3 hidden layers:** shared trunk 4 \u2192 128 \u2192 128, then a value stream 128 \u2192 128 \u2192 1 and an advantage stream 128 \u2192 128 \u2192 2 (50,563 parameters). **5 hidden layers:** shared trunk of four 128-unit layers, then the same two streams (83,587 parameters)."),
  H2("Results and comparison with Section 2"),
  table(["Algorithm", "Network", "Episodes until avg \u2265 475", "Greedy test mean / min"], [
    ["DQN (Section 2)", "3 hidden layers", fmt(D3.solved_at), `${D3.test_mean?.toFixed(1)} / ${D3.test_min}`],
    ["Dueling DQN", "3 hidden layers", fmt(DU3.solved_at), `${DU3.test_mean?.toFixed(1)} / ${DU3.test_min}`],
    ["DQN (Section 2)", "5 hidden layers", fmt(D5.solved_at), `${D5.test_mean?.toFixed(1)} / ${D5.test_min}`],
    ["Dueling DQN", "5 hidden layers", fmt(DU5.solved_at), `${DU5.test_mean?.toFixed(1)} / ${DU5.test_min}`],
    ["Double DQN (extra)", "3 hidden layers", fmt(DD3.solved_at), `${DD3.test_mean?.toFixed(1)} / ${DD3.test_min}`],
    ["Double DQN (extra)", "5 hidden layers", fmt(DD5.solved_at), `${DD5.test_mean?.toFixed(1)} / ${DD5.test_min}`],
  ], [2300, 2000, 2500, 2200]),
  gap(),
  P(`With 3 hidden layers, Dueling DQN reached an average reward of at least 475 over 100 consecutive episodes after **${fmt(DU3.solved_at)} episodes**, compared with ${fmt(D3.solved_at)} for DQN (${D3.solved_at - DU3.solved_at} fewer episodes, ${Math.round(100 * (D3.solved_at - DU3.solved_at) / D3.solved_at)}% faster), and its greedy policy scored ${DU3.test_mean?.toFixed(1)} on every one of 100 new test episodes (DQN: ${D3.test_mean?.toFixed(1)}, minimum ${D3.test_min}). ${DUELING_5L_TEXT} The reward and loss per training step are in Figures C1\u2013C4, and Figure C5 compares the 100-episode average reward of all runs. As in Section 2, the loss grows during training because the target Q-values grow as the agent survives longer.`),
  P(`We also tried **Double DQN** (van Hasselt et al., 2016), which uses the online network to select the next action and the target network to evaluate it, y = r + \u03b3Q(s\u2032, argmax_a Q(s\u2032,a;\u03b8); \u03b8\u207b), to reduce the over-estimation caused by the max operator (Figures C6\u2013C9). It solved the task after ${fmt(DD3.solved_at)} episodes with 3 layers (test ${DD3.test_mean?.toFixed(1)}) and did not reach 475 within 1000 episodes with 5 layers, so it did not help here: over-estimation is mild in CartPole (two actions, +1 reward at every step) and its more conservative targets make the values grow more slowly. Each configuration was trained with one seed, so differences of a few dozen episodes are partly run-to-run variance; a firmer conclusion would need several seeds.`),
];

// ---------- appendix ----------
const appendix = [
  new Paragraph({ children: [new PageBreak()] }),
  H1("Appendix \u2013 Figures"),
  H2("A. Section 1 \u2013 Tabular Q-learning"),
  ...figure("results/section1/qtable_500_steps.png", "Figure A1 \u2013 Q-table after 500 steps", 3.6),
  ...figure("results/section1/qtable_2000_steps.png", "Figure A2 \u2013 Q-table after 2000 steps", 3.6),
  ...figure("results/section1/qtable_final.png", "Figure A3 \u2013 Final Q-table (after 5000 episodes)", 3.6),
  ...figure("results/section1/reward_per_episode.png", "Figure A4 \u2013 Reward per episode (dots) with 100-episode moving average", 6.2),
  ...figure("results/section1/avg_steps_to_goal.png", "Figure A5 \u2013 Average steps to the goal over the last 100 episodes (100 if the goal was not reached), every 100 episodes", 6.2),
  ...figure("results/section1/qtable_500_episodes.png", "Figure A6 \u2013 Q-table after 500 episodes (supplementary)", 3.6),
  ...figure("results/section1/qtable_2000_episodes.png", "Figure A7 \u2013 Q-table after 2000 episodes (supplementary)", 3.6),
  H2("B. Section 2 \u2013 DQN"),
  ...figure("results/section2/3L_seed0_reward.png", "Figure B1 \u2013 DQN, 3 hidden layers: total reward per training episode", 6.2),
  ...figure("results/section2/3L_seed0_loss.png", "Figure B2 \u2013 DQN, 3 hidden layers: loss at every training step", 6.2),
  ...figure("results/section2/5L_seed0_reward.png", "Figure B3 \u2013 DQN, 5 hidden layers: total reward per training episode", 6.2),
  ...figure("results/section2/5L_seed0_loss.png", "Figure B4 \u2013 DQN, 5 hidden layers: loss at every training step", 6.2),
  H2("C. Section 3 \u2013 Dueling DQN (and Double DQN, extra)"),
  ...figure("results/section3_dueling/3L_seed0_reward.png", "Figure C1 \u2013 Dueling DQN, 3 hidden layers: total reward per training episode", 6.2),
  ...figure("results/section3_dueling/3L_seed0_loss.png", "Figure C2 \u2013 Dueling DQN, 3 hidden layers: loss at every training step", 6.2),
  ...figure("results/section3_dueling/5L_seed0_reward.png", "Figure C3 \u2013 Dueling DQN, 5 hidden layers: total reward per training episode", 6.2),
  ...figure("results/section3_dueling/5L_seed0_loss.png", "Figure C4 \u2013 Dueling DQN, 5 hidden layers: loss at every training step", 6.2),
  ...figure("results/comparison_all.png", "Figure C5 \u2013 DQN, Dueling DQN and Double DQN: average reward of the last 100 episodes", 6.2),
  ...figure("results/section3/3L_seed0_reward.png", "Figure C6 \u2013 Double DQN, 3 hidden layers: total reward per training episode", 6.2),
  ...figure("results/section3/3L_seed0_loss.png", "Figure C7 \u2013 Double DQN, 3 hidden layers: loss at every training step", 6.2),
  ...figure("results/section3/5L_seed0_reward.png", "Figure C8 \u2013 Double DQN, 5 hidden layers: total reward per training episode", 6.2),
  ...figure("results/section3/5L_seed0_loss.png", "Figure C9 \u2013 Double DQN, 5 hidden layers: loss at every training step", 6.2),
];

const doc = new Document({
  styles: {
    default: { document: { run: { font: F, size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: F, size: 28, bold: true, color: "1F3864" }, paragraph: { spacing: { before: 240, after: 100 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: F, size: 24, bold: true, color: "2F5496" }, paragraph: { spacing: { before: 160, after: 80 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022",
    alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1417, bottom: 1417, left: 1417, right: 1417 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], font: F, size: 20 })] })] }) },
    children: [...body, ...appendix],
  }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync("report.docx", buf); console.log("report.docx written"); });
