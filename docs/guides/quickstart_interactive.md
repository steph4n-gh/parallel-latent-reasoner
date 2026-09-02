# Guide: Interactive Terminal Deliberation & Visualizer

The Parallel Latent Reasoner (PRLR) includes an interactive dual-pane visualizer designed to run directly in your terminal. It allows you to explore the dynamics of continuous thought vs. standard token streaming in real-time.

## 1. Quick Launch

Launch the interactive REPL with production weights loaded by default:

```bash
python3 demo.py --interactive
```

## 2. CLI Options & Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--prompt "<text>"` | string | Sample prompt | Run deliberation on an ad-hoc user prompt. |
| `--case <case_id>` | string | None | Execute a specific benchmark case (`mcs_01..05`, `wsd_01..05`, `sdn_01..05`, `cms_01..05`, `atr_01..05`). |
| `--model <preset>` | string | `compact_test` | Resident scale profile (`compact_test`, `gemma_2b`, `gemma_9b`, `gemma_12b_q4`, `gemma_26b_a4b`). |
| `--slots <M>` | int | `16` | Number of continuous working memory slots. |
| `--steps <T>` | int | `8` | Maximum recurrent unroll sweeps. |
| `--no-gate` | flag | False | Disable the 3-Signal Dynamic Consensus E-Gate (forces fixed depth $T$). |
| `--temperature <T>` | float | `0.0` | Sampling temperature for discrete Coda decoding. |

## 3. Interpreting Telemetry

During deliberation, the right pane renders live diagnostic telemetry for each unroll step $t$:

```text
 Step | Velocity | Rel Decay | erank | Coda Pred | Status
------+----------+-----------+-------+-----------+--------
  t=1 | 0.006779 |  1.0000   |  1.30 |  "Beta"   | Active
  t=2 | 0.004414 |  0.6511   |  1.38 |  "Beta"   | Active
  t=3 | 0.002936 |  0.4331   |  1.41 |  "Beta"   | Active
  t=4 | 0.001915 |  0.2825   |  1.44 |  "Beta"   | Active
  t=5 | 0.000850 |  0.0820   |  1.44 |  "Beta"   | HALT (E-Gate Consensus)
```

- **Velocity $v(t)$**: Euclidean step distance between continuous slot representations.
- **Rel Decay**: Ratio $v(t) / v(1)$. When this drops below $0.10$, $\ge 90\%$ of kinetic momentum has dissipated.
- **erank**: SVD Shannon entropy of the 16 memory slots. A healthy trajectory maintains $\text{erank} \ge 1.3$.
- **Coda Pred**: Greedy readout from the linear Coda head without token emission. When predictions stabilize across steps, the early-exit condition is satisfied.
