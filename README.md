# swarm-paper

Code, results and manuscript drafts for a study of decentralized formation control of an 8-robot swarm under partial observability, using a message-passing graph neural network (GNN).

One human gesture selects one of four formations (circle, line, cluster, split). Only some robots (the *observers*, 100 %, 50 % or 25 % of the swarm) see the gesture. The gesture token has to reach the other robots through the communication graph (range 0.5 m). The question is how many message-passing hops K the policy needs.

Everything is simulated in MuJoCo (headless, no GPU needed) with a custom 8-robot e-puck model. Every number in the paper is produced by a script in `src/` from the CSV files in `results/`.

**Status:** manuscript in preparation.

**Authors:** Yashvardhan Abhale, Sahil Panchavishe, Ishan Zadbuke, Vivek Khandelwal.

## 1. Setup

Tested with Python 3.12.3 on macOS.

```bash
git clone https://github.com/sahilpanchavishe/swarm-paper
cd swarm-paper
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
cd src
python test_mj_env.py        # smoke test: MuJoCo model against ideal unicycle kinematics
```

Run every command below from `src/` with the virtual environment active. On macOS, scripts that use multiprocessing need the `if __name__ == "__main__":` guard, which they already have.

## 2. Reproduce the numbers in the paper (no training, a few minutes)

The trained models (`runs/*.pt`) and all result files (`results/*.csv`) are committed, so you can recompute the statistics, tables, figures and every number quoted in the text:

```bash
python d6_analysis.py          # success table, 16 paired comparisons with Holm correction, figures/d6_main_success.png
python d8_make_figures.py      # paper/figs/*.png and paper/tables/*.csv + *.md
python d8_check_numbers.py     # recomputes every number in paper/results_discussion_draft.md, fails if any differ
python d9_check_numbers.py     # same for the exploratory Day 9 results and the constants in the Methods
```

To re-run the closed-loop evaluation of the committed models (slower; already-evaluated models are skipped, so delete `results/d6_main_eval.csv` to redo all):

```bash
python d6_main_eval.py         # 50 test seeds x 4 formations x 3 observability levels per model
python d6_analysis.py
python d8_make_figures.py
```

## 3. Run the whole pipeline from scratch

| Step | Command | Writes |
|---|---|---|
| 1. Validate the robot model | `python validate_model.py` | `results/model_validation.csv`, `figures/model_validation.png` |
| 2. Expert check | `python expert_check.py 30` | printed summary (about 2 to 4 minutes) |
| 3. Physics sensitivity | `python sensitivity.py` | `results/sensitivity.csv` |
| 4. Connectivity and token coverage | `python connectivity.py` | `results/connectivity_*.csv`, `results/token_coverage_vs_K.csv`, `figures/connectivity.png` (about a minute) |
| 5. No-learning baselines | `python d3_baselines.py 50` | `results/d3_baselines.csv` |
| 6. Demonstrations | `python d4_gen_data.py` | `data/raw/d4_demos.npz` (git-ignored, regenerate it) |
| 7. Behaviour cloning | `python d4_train.py --K 2 --seed 0 --epochs 30 --eval 0` | `runs/d4_K2_s0.pt` |
| 8. DAgger | `python d5_dagger.py --K 2 --seed 0 --rounds 3 --eps 240 --epochs 10 --dev 5 --eval 0` | `runs/d5_K2_s0.pt`, `results/d5_dagger_log.csv` |
| 9. Evaluation and statistics | `python d6_main_eval.py` then `python d6_analysis.py` | `results/d6_main_eval.csv`, `results/d6_table_*.csv` |
| 10. Token ablation | `python d7_token_ablation.py` | `results/d7_token_ablation.csv` |
| 11. Exploratory analyses (not in the frozen protocol) | `python d9_relay_default.py` and `python d9_token_cutoff.py` | `results/d9_relay_default*.csv`, `results/d9_token_cutoff.csv` |
| 12. Tables, figures, checks | `python d8_make_figures.py`, `python d8_check_numbers.py`, `python d9_check_numbers.py` | `paper/figs/`, `paper/tables/` |

Steps 7 and 8 are repeated for K = 0, 1, 2, 3 and training seeds 0, 1, 2, plus the latched-token variant (`--latched`, K = 2). Two helper scripts do this unattended:

```bash
caffeinate -i bash d6_more_seeds.sh 1 2        # K = 3, 2, 1, 0 for the given seeds (about 2 hours per seed)
caffeinate -i bash d10_latched_seeds.sh 1 2    # latched K = 2 for the given seeds
```

`caffeinate` keeps a Mac awake. Both scripts skip a model whose `runs/d5_*.pt` file already exists, so an interrupted run can be restarted with the same command.

Retraining can give slightly different models on a different machine or library version. The committed models are the ones behind the paper's numbers.

## 4. Experimental protocol (frozen on Day 6)

- 8 differential-drive e-pucks, fixed role assignment (robot *i* goes to slot *i*), start box [-0.5, 0.5]² m, horizon 1100 control steps (35.2 s), control step 0.032 s.
- Success: mean final slot error below 0.025 m (0.05 m also reported). Contact: centre distance below 0.074 m. Overlap: below 0.069 m.
- Seeds: demonstrations 0 to 1199, DAgger 1300 to 4299, development 5000 and up, **test 10000 to 10049**.
- Paired design: every policy sees the same (formation, seed, observability) episodes.
- Statistics: Wilson confidence intervals, paired Wilcoxon tests, bootstrap confidence intervals (5000 resamples), Holm correction over the 16 comparisons in `d6_analysis.py`. Primary hypothesis: GNN K = 2 against K = 1 at 25 % observability.
- Three training seeds for K = 0 to 3 and for the latched variant (the original plan was five).
- The Day 9 analyses (fairer relay baseline, token cut-off test) were added after the protocol was frozen. They are exploratory and are labelled that way in the paper.

The full protocol is in `01_protocol/protocol.md`; decisions are logged in `logs/decisions.md`.

## 5. Repository layout

| Path | Contents |
|---|---|
| `src/` | All code. Scripts are prefixed by the project day (`d4_`, `d5_`, ...). `swarm_sim.py` (kinematics, expert, formations) and `mj_env.py` (MuJoCo environment) are shared by everything. |
| `results/` | Raw and summary CSV files. Nothing here is edited by hand. |
| `runs/` | Trained model weights (`d4_*` after behaviour cloning, `d5_*` after DAgger). |
| `paper/` | `methods_draft.md`, `results_discussion_draft.md`, `figs/`, `tables/` |
| `figures/` | Working figures written by the analysis scripts |
| `01_protocol/` | Frozen protocol |
| `logs/` | Decision log, project status, training logs |
| `data/raw/` | Demonstration data (git-ignored, regenerate with `d4_gen_data.py`) |

## 6. Rules the code follows

- A changed script gets a new day-prefixed name. `swarm_sim.py` and `mj_env.py` are never overwritten.
- Every table, figure and number in the text comes from a script that reads the raw CSV files. If a number in the text disagrees with a checker, the text is wrong, not the data.

## 7. Licence and citation

TODO: choose a licence for the code (for example MIT) and add a `LICENSE` file; add the citation once the paper has a DOI.

## 3. Studies added on Day 15 (protocol: `01_protocol/protocol_day15_addendum.md`)

| Study | What | Seeds | Status |
|---|---|---|---|
| A | Disconnected starts (`d15_A_*` suites, `src/d15_scenarios.py`) | 50000-50024 | confirmatory |
| B | Safety filter under link loss, delay and noise (`d15_filter_dist`) | 40000-40024 | exploratory (results seen before predictions) |
| C | Token cut-off redo with position-only readout (`src/d15_cutoff.py`) | 60000-60019 | confirmatory |
| D | Command change replication (`d15_change`) | 70000-70024 | exploratory |
| Drift, frontier, failure replay | `d15_drift.py`, `d15_frontier` suite, `d15_replay.py` | 80000+, 5000-5009 | exploratory |

Static and velocity-dependent delay margins were tried on dev seeds 5000-5009 and failed (they remove contact by blocking the formation); the final runs use the plain filter. Delay safety is an open limitation.

Run from `src/`:
```bash
python d12_eval.py <suite>      # d15_A_conn1, d15_A_comp2_same, d15_A_comp2_split, d15_A_comp3_same, d15_filter_dist, d15_change, d15_frontier
python d15_cutoff.py --models ... --cuts ... --out ../results/<file>.csv    # see run_d15_rest.sh for the exact arguments
python d15_stats_A.py; python d15_stats_B.py; python d15_stats_C.py
python d15_drift.py; python d15_replay.py
python d15_tables.py            # paper/tables/t15_*, paper/figs/f15_*
python check_claims.py          # every number in the text against paper/claims.csv
```
`run_d15.sh`, `run_d15_B.sh`, `run_d15_rest.sh` and `run_d15_tail.sh` chain these in the order used.
