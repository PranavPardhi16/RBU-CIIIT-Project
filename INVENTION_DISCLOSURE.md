# Invention Disclosure (draft): ConformGuard

> **Status:** technical disclosure draft for discussion with a supervisor / IP cell / patent attorney.
> **This is not a patent application, not legal advice, and not a claim that the idea is novel or patentable.**
> Novelty can only be established by a professional prior-art search. Section 6 lists what must be checked.

## 1. Working title
Regime-conditional, alarm-budget-calibrated detection of attacks on industrial control telemetry, with a stability-selected CUSUM bank on invariant residuals.

## 2. Technical field and problem
Intrusion/anomaly detection on PLC/SCADA time series (cyber-physical systems). Three practical problems that block deployment:

1. **Operator commands cause false alarms.** After a setpoint change the plant is legitimately in transition. Detectors scoring all tags treat the command step and the control lag as anomalies (measured here: the AR score of a command column was ~100x larger in test than in calibration data).
2. **Stealthy persistent drift is invisible to one-step detectors.** Residual-based detectors adapt to slow ramps (a ramp is predictable from lagged values).
3. **Thresholds have no operational meaning.** Plant staff think in "alarm incidents per day", not in score quantiles, and calibrated thresholds do not transfer under benign operating-point shift.

## 3. Summary of the method (as implemented in `icsguard/engine.py`)
- **S1. Command discovery.** Tags that are non-constant but change in <0.5% of samples are auto-detected as commands/setpoints. Commands are treated as exogenous inputs: they condition the models but are *not scored* and are not targets of invariants.
- **S2. Data-estimated regime window.** A *transient* regime lasts W seconds after any command change; W is the 90th percentile of the fused-score settling time measured on attack-free calibration data (not hand-set).
- **S3. Regime-conditional (Mondrian) conformal p-values.** The fused residual score is compared only against calibration scores from the same regime, so one significance level gives comparable alarm rates in steady and transient operation.
- **S4. Alarm-budget mapping.** The operator gives alarm *runs per day*. The significance level alpha and the CUSUM threshold h are found by replaying attack-free calibration data (debounced run counting), with the budget split between the two channels.
- **S5. Stability-selected CUSUM bank.** A two-sided CUSUM runs over signed, normalised residuals of mined linear invariants, but **only for invariants passing two tests on calibration data**: near-white residuals (lag-1 autocorrelation < 0.8) and benign bias stability (block-mean wander <= 0.6 x CUSUM allowance). Inputs are Huber-clipped; accumulators **freeze during transients**; at each transient-to-steady transition the reference is **re-baselined** to the mean residual of the last 120 s of the transient.
- **S6. Envelope triage.** Alarms occurring while a command is outside its calibrated range are flagged "envelope" (novel operating point) so operators can separate them from attack-like alarms.
- **S7. Single code path.** The batch analysis and the streaming per-sample engine share the same scoring and decision code; a unit test asserts identical p-values, CUSUM statistics and alarms.
- **S8. Optional quarantined rolling calibration** (delayed acceptance with a +/-D window free of suspicious p-values). Implemented and evaluated; **it showed no measurable benefit** (see 4).

## 4. Evidence (simulated plant, 10 seeds x 12 attacks, budget 5 runs/day; `run_conformguard.py`)
| Variant | Event recall | Event precision | Event-F1 (95% CI) | Stealth-ramp recall | In-envelope FA/day (realised) |
|---|---|---|---|---|---|
| AR residual, global threshold (baseline) | 0.64 | 0.75 | 0.67 (0.54-0.77) | 0.07 | 9.7 |
| AR + invariants fused, global threshold | 0.62 | 0.48 | 0.47 (0.37-0.59) | 0.03 | 33.1 |
| AR + regime-conditional conformal | 0.72 | 0.78 | **0.75** (0.71-0.78) | 0.07 | 10.5 |
| **AR + regime-conditional + CUSUM bank** | **0.88** | 0.56 | 0.67 (0.61-0.73) | **0.60** | 14.6 |

Statistical tests (paired Wilcoxon over 10 seeds, budget 5): recall of the full variant vs the baseline +0.23 (p = 0.016); regime conditioning vs baseline on event-F1 +0.077 (p = 0.055, borderline); full variant vs baseline on event-F1 is a tie (p = 0.56).

**What did NOT work / remains open (important for any claim):**
- **No variant holds the alarm budget under distribution shift**: realised in-envelope alarm runs/day were ~2.2-2.7x the budget even for the best variants.
- Rolling calibration gave no benefit; the quarantine rule refuses exactly the shifted data it would need to adapt to.
- Fusing the invariant *score* with the AR score reduced precision; invariants helped only through the CUSUM channel.
- Results are on a **simulator designed by the author**; real-plant behaviour is unknown.

## 5. Candidate distinguishing features (hypotheses to test against prior art)
D1 (S1+S2+S3): regime derived automatically from command-tag change events with a data-estimated settling window, used as Mondrian categories for conformal alarm p-values on ICS residuals.
D2 (S5): CUSUM bank gated by calibration-time whiteness *and* benign-bias-stability selection, with regime freezing and operating-point re-baselining.
D3 (S4): mapping an operator-specified alarm-runs/day budget to joint (alpha, h) via debounced attack-free replay.
D4 (S6): command-envelope triage flag attached to alarms.
Individually, conformal anomaly detection, Mondrian conformal prediction, CUSUM and invariant mining are **known**. Any novelty would have to lie in the specific combination and the selection/gating rules above, and that is unverified.

## 6. Prior art that must be reviewed (starting list, not exhaustive)
- Conformal anomaly detection (e.g., Laxhammar & Falkman), Mondrian/class-conditional conformal prediction (Vovk et al., *Algorithmic Learning in a Random World*), adaptive conformal inference under shift (Gibbs & Candes, NeurIPS 2021).
- CUSUM (Page, 1954) and robust/Huber CUSUM variants; control-chart methods for CPS.
- Invariant-based attack detection in water treatment (e.g., Adepu & Mathur) and learned/sequence detectors on SWaT/WADI/HAI (USAD, TranAD and others).
- Evaluation critiques of point-adjusted metrics (Kim et al., AAAI 2022).
- Patent databases (Indian IPO InPASS, Espacenet, Google Patents, WIPO). Suggested search themes: "conformal" + "industrial control" + "alarm"; "setpoint change" + "anomaly detection" + "false alarm"; "CUSUM" + "invariant" + "PLC". Likely classification areas (verify): G05B 23/02 (monitoring of control systems), H04L 63/14 (network security, attack detection), G06N 20/00.

## 7. IP and disclosure cautions (please verify locally; this is general information)
- **Indian patent law:** Section 3(k) excludes "a mathematical or business method or a computer programme per se or algorithms". Software-based inventions generally need a demonstrable technical effect or hardware integration to be considered. A pure algorithm may be better protected by **publication + open-source licensing** than by a patent.
- **Public disclosure can destroy novelty.** Posting this repo publicly, an arXiv preprint, a conference talk or even a detailed professor pitch can count as disclosure in many jurisdictions. India's grace period (Section 31) is narrow and most other countries are stricter. Decide the route *before* publishing.
- **Ownership:** work done during an IIT internship, with institute resources or supervision, is typically governed by the institute's IP policy. Ask the supervisor and the IIT Bhubaneswar IP/technology-transfer cell before treating this as personally owned. Keep dated records showing this work pre-dates any internship.
- **Practical recommendation:** treat this as a research artefact (paper/workshop + open-source) unless a supervisor and IP cell see a hardware-linked, commercially valuable claim.

## 8. What would make it genuinely strong
1. Reproduce on real data (HAI/SWaT-class, and Siemens VNIT logs if permitted); report where simulation conclusions break.
2. Solve budget transfer under shift (drift-aware calibration with operator acknowledgement instead of blind quarantine).
3. Edge deployment evidence (latency/memory on an industrial gateway), which would also strengthen any technical-effect argument.
4. Independent replication of the CUSUM-bank result on a second simulator or testbed.
