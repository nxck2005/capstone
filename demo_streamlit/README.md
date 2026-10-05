# Streamlit demo

```bash
./run-streamlit-demo.sh
```

Run it from the repository root. The first run creates `demo_streamlit/.venv` from the hashed CPU lock plus `requirements.txt`, which needs internet and a few minutes. Later runs work offline. The page opens at `http://127.0.0.1:8501`. `--port` and `--no-browser` are available, and `?snr=-4` deep-links to a measured SNR.

This is a second, academic-style front end. It does not replace the React exhibition app in `demo/`.

- **Section 1** sends one bundled training-split image through both links at the selected SNR. It uses fresh CPU inference with the frozen *r* = 1/6 checkpoints, through the verified `LiveLearned` and `LiveClassical` classes in `demo/backend/app.py`. The weights are the same ignored files in `demo/assets/checkpoints/`, provisioned with `demo/scripts/provision_assets.py`. Without them, each arm says it is unavailable and nothing is substituted.
- **Section 2** shows the published **G-12 test-split** results: `presentation-results/data/g12_test_curves.csv` and `g12_test_differences.csv`. On every start, each plotted point is checked against `results/g12/results.csv`. The figure is drawn by the paper's own `deliverables/research-paper/figures/make_figures.py`, so it matches the paper's headline figure (DR-4).
- Nothing trains, fine-tunes or recomputes a reported metric (DR-6). The server binds to 127.0.0.1 only, and usage statistics are off (DR-5).

Checks: `demo_streamlit/.venv/bin/python -m pytest -q tests/test_streamlit_demo.py`.
