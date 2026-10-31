.PHONY: install demo-model serve test bench
install:        ; pip install -r requirements.txt && pip install -e .
demo-model:     ; python -m icsguard train --demo --out models/demo --budget 5
serve:          ; python -m icsguard serve --model models/demo.npz --stream models/demo_stream.npz --port 8000
test:           ; python tests/test_core.py && python tests/test_engine.py && python tests/test_api.py
bench:          ; python run_conformguard.py --seeds 10 --out results_cg      # ~4 min, ConformGuard ablation
bench-phase1:   ; python run_experiments.py --config configs/full.json --out results   # phase-1 detector benchmark
