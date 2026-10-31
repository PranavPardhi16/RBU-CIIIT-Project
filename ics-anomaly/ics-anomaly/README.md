# ICSGuard / ConformGuard
**Budget-calibrated, regime-aware attack detection for PLC/SCADA telemetry, packaged as a streaming service with a live dashboard.**

```
 PLC / historian ──► POST /api/ingest ─┐
                                       ▼
   ┌───────────────────────── icsguard.engine.Guard ─────────────────────────┐
   │ commands ➜ regime (steady | transient, W learned) ➜ AR residual score  │
   │ Mondrian conformal p-value  ─┐                                          │
   │ stability-selected CUSUM bank ┴► alarm (budget-calibrated α, h) + tags  │
   └──────────────────────────────────────────────────────────────────────────┘
        │ REST / SSE                          │ envelope triage flag
        ▼                                     ▼
   dashboard (/)                       alarm log + top-3 suspicious tags
```

## Quick start
```bash
pip install -r requirements.txt && pip install -e .
python -m icsguard train --demo --out models/demo --budget 5        # simulate, train, calibrate (~10 s)
python -m icsguard serve --model models/demo.npz --stream models/demo_stream.npz --port 8000
# open http://localhost:8000  ->  "Start live replay"
make test                                                           # 22 tests
```
Your own data: `python -m icsguard train --csv plant.csv --label-col label --command-cols SP --out models/plant --budget 5`
(trains on attack-free rows; command tags are auto-detected if not given), then `icsguard replay --model models/plant.npz --csv new.csv`.

## API
| Endpoint | Purpose |
|---|---|
| `GET /` | live dashboard |
| `POST /api/ingest` | `{"rows": [[...]]}` or `{"records": [{tag: value}]}` → per-sample p-value, alarm, regime, CUSUM stat, top tags |
| `GET /api/alerts` | closed alarm runs (+ open one) with tags, envelope flag, channel |
| `POST /api/budget` | change alarm-runs/day budget live |
| `GET /api/model`, `/health`, `POST /api/reset` | metadata, liveness, state reset |
| `GET /api/demo/stream` (SSE), `/api/demo/replay` | labelled demo replay |

Engine speed: ~140 µs/sample (single core, numpy), ~2,600 samples/s through HTTP+JSON in a local test.

## What is new here (and how strongly it is supported)
1. **Commands are exogenous and define regimes**; conformal p-values are computed per regime (Mondrian), with the transient window learned from data.
2. **Stability-selected CUSUM bank** on invariant residuals (near-white + benign-bias-stable only), regime-frozen and re-baselined at operating-point changes.
3. **Operator budget** (alarm runs/day) mapped to (α, h) by attack-free replay; **envelope triage** flags alarms at novel operating points.

Evidence (simulated plant, 10 seeds × 12 attacks, budget 5/day; full tables in `results_cg/`, figures in `docs/figures/`):

| Variant | Recall | Event-F1 | Stealth-ramp recall |
|---|---|---|---|
| AR, global threshold (baseline) | 0.64 | 0.67 | 0.07 |
| AR + regime-conditional conformal | 0.72 | **0.75** | 0.07 |
| **AR + regime + CUSUM bank (product default)** | **0.88** (p=0.016 vs baseline) | 0.67 | **0.60** |

![ablation](docs/figures/conformguard_ablation.png)

## Honest limitations
- **Simulated data only.** The plant, attacks and nuisances are my design; real-world performance is unknown. HAI/SWaT and the LSTM model (`src/deep.py`) are **untested** here.
- **Calibration is not solved:** realised false-alarm runs were ~2–3× the budget under distribution shift. The demo dashboard will visibly over-alarm on its seed.
- Rolling calibration (implemented, off by default) showed **no benefit**; fusing invariant scores with AR scores **hurt** precision.
- The UI was syntax-checked and its API tested, but **not visually verified in a browser**; the Dockerfile was **not built** (no Docker daemon in the authoring environment).
- No authentication/TLS: put it behind a gateway/VPN before touching a real plant network. It is a detection aid, not a safety system.

## Repository map
`icsguard/` product (engine, service, CLI, dashboard) · `src/` research code (simulator, metrics, baselines, LSTM-AE) · `run_conformguard.py` ablation benchmark · `run_experiments.py`, `tune_gate.py` phase-1 benchmark (`docs/PHASE1_BENCHMARK.md`) · `docs/INVENTION_DISCLOSURE.md` · `tests/` · `Dockerfile`, `Makefile`, `pyproject.toml`.
No license file is included on purpose: choose one (e.g., Apache-2.0) only after clarifying IP ownership (see the disclosure document).
