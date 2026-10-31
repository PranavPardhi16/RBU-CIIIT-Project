# Process-Aware Anomaly Detection for PLC-Controlled ICS
**Hybrid learned-reconstruction + mined-invariant detection, with transient-aware gating, evaluated event-wise**

A reproducible benchmark and method study for detecting attacks in PLC/SCADA-style telemetry.
Everything in `results/` is produced by the code here (10 seeds, `configs/full.json`).

## What is in the repo
| Component | File | Notes |
|---|---|---|
| Process simulator | `src/synth.py` | Two-tank plant, PI control, actuator lag, benign mode changes, unlogged demand, distractor sensor, dropouts, 5 attack types with ground-truth affected tags |
| Detectors | `src/models.py` | Isolation Forest, PCA residual, AR (ridge, lagged) residual, denoising MLP-AE (early-stopped), hybrid fusion, **GatedScore** |
| Invariant miner | `src/invariants.py` | Lasso relations; selected on a **held-out** attack-free block; per-tag attribution weights |
| Deep model | `src/deep.py` | LSTM-AE with AdamW, cosine LR, clipping, denoising, early stopping, best-checkpoint restore. **Not executed (no PyTorch in authoring sandbox)** |
| Metrics | `src/metrics.py` | Event recall/precision/F1, detection delay, false alarms/day, trade-off curves, alarm-budget calibration; PA-F1 only for contrast |
| Attribution | `src/attribution.py` | hit@1 / hit@3 of the true affected tag |
| Statistics | `src/stats.py` | Bootstrap CIs, paired Wilcoxon across seeds |
| Runner / tuning / tests | `run_experiments.py`, `tune_gate.py`, `tests/test_core.py` | 7 unit tests pass |
| HAI support | `src/data.py`, `--hai` flag | **Untested on real HAI data** |

## Protocol (what makes it rigorous)
- Temporal split per run: train 70% | early-stopping 10% | calibration 20% (all attack-free) | test (attacks).
- Thresholds, score normalisation and invariant R2/residual scales come from attack-free blocks only. No test-label tuning in operating-point metrics.
- Test half deliberately extrapolates beyond training (wider setpoint range, wider demand range, diurnal drift).
- Gate window W chosen on separate validation simulations (seeds 100-102); evaluation seeds 0-9 untouched.
- 10 independent simulated plants; bootstrap CIs over seeds; Wilcoxon for pairwise claims.
- Oracle trade-off curves sweep thresholds on test and are labelled as such.

## Results (synthetic benchmark, 10 seeds, 120 attack events; primary threshold = 1e-3 calibration quantile, 5 s debounce)
| Model | Event-F1 (95% CI) | Event recall | FA/day | Attribution hit@3 |
|---|---|---|---|---|
| Isolation Forest | 0.19 (0.12-0.25) | 0.23 | 67 | - |
| PCA residual | 0.43 (0.31-0.54) | 0.43 | 59 | 0.74 |
| **AR residual** | **0.74 (0.69-0.78)** | 0.73 | 14 | 0.70 |
| MLP-AE | 0.40 (0.36-0.43) | 0.59 | 40 | 0.76 |
| Invariants | 0.30 (0.25-0.34) | 0.34 | 25 | 0.60 |
| Invariants + gate (W=600) | 0.50 (0.38-0.61) | 0.58 | 109 | 0.59 |
| Hybrid AR + invariants | 0.62 (0.52-0.71) | 0.72 | 27 | 0.77 |
| Hybrid AR + gated invariants (W=1500, validation-chosen) | 0.66 (0.59-0.73) | 0.75 | 41 | **0.84** |

Per-attack-type recall, alarm-budget results, all ablations: `results/summary.csv`, `results/per_attack_type_recall.csv`, `results/significance_wilcoxon.csv`.

## Findings, including the ones that go against the original hypothesis
1. **The original hypothesis is NOT supported here.** Fusing invariants with a learned scorer did not beat a well-tuned AR residual baseline on event-F1 (hybrid AR+invariants 0.62 vs AR 0.74; Wilcoxon p=0.027, hybrid worse). A strong simple baseline matters more than a fancier model.
2. **Benign mode changes are a real failure mode.** Alarms from invariants and learned scorers cluster after setpoint changes (71-81% of false alarms vs ~28% base rate of test time).
3. **Transient-aware gating helps invariants substantially** (event-F1 0.30 to 0.50; share of false alarms near mode changes 0.81 to 0.22). Gating also gave the best attribution (hit@3 0.84 with AR). It does not make the hybrid beat plain AR on F1.
4. **Calibration does not transfer under shift.** Thresholds set for 5 alarm-runs/day on attack-free data realised 11 to 200 alarms/day on the shifted test half. Gated invariants were worst (partly because neutralised windows change the calibration score distribution; not yet fixed). This is the most promising open problem (drift-aware calibration).
5. **Stealthy ramp attacks are almost never detected** (recall <= 0.13 for every method). Honest open problem.
6. **PA-F1 overstates results**, e.g. MLP-AE 0.71 vs event-F1 0.40.
7. The invariant miner recovers sensible physics (level ~ setpoint/pump/pressure; flow ~ level via the square-root law).

## Limitations (state these in any write-up)
- All numbers are from a **simulated** plant. They show relative behaviour under controlled conditions, not performance on real PLC logs or HAI.
- The LSTM-AE is unexecuted; the HAI loader is untested on the real files.
- FA/day is high in absolute terms (even the best model); this benchmark is deliberately hostile (unseen operating points, dropouts).
- Attack parameters and the simulator are my design; conclusions about hybrid vs AR may change on real data.

## Run
    pip install -r requirements.txt
    python tests/test_core.py
    python tune_gate.py                                    # choose W on validation seeds
    python run_experiments.py --config configs/full.json   # ~3.5 min, writes results/
    python run_experiments.py --hai "data/hai/train*.csv" --hai_test "data/hai/test*.csv" --seeds 3 --models ar,pca,iforest,mlp_ae,invariant,lstm_ae

## Suggested next steps for the internship
1. Run on real HAI (and your Siemens VNIT logs if permitted); report where findings differ from simulation.
2. Fix gated-score calibration; add drift-aware thresholding (connects to concept-drift work).
3. Attack-specific work on stealthy ramps (CUSUM on residuals; slow-drift detectors).
4. Run `lstm_ae` and add ONNX export + latency benchmark.
