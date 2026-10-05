# Capstone exhibition · offline simulated wireless link

The dashboard places a task-aware DJSCC receiver beside adaptive JPEG 2000 + LDPC transmission on the **same Imagenette-160 image and simulated AWGN SNR**. Its plot is a read-only projection of the published W10 **validation** results (`presentation-results/data/w10_primary_252.csv` / W10 v11 closeout), **not** an accuracy measurement of the image on screen. It was built while the test split was sealed and still shows validation data only; for the final G-12 test results use `./run-g12-demo.sh` or `./run-streamlit-demo.sh` (see `demo_g12/README.md`). No demo call trains a model, changes scientific evidence or uses a network service.

## Start (WSL2 or CPU-only Linux laptop)

From the repository root, install dependencies **while online**, once:

```bash
python3 -m venv demo/.venv-demo
demo/.venv-demo/bin/python -m pip install --require-hashes -r requirements-cpu.lock
demo/.venv-demo/bin/python -m pip install -r demo/backend/requirements.txt
cd demo/frontend && npm ci && npm run build && cd ../..
```

Python 3 and Node/npm are required for installation; Node is **not** required at runtime after the build. The system JPEG 2000 library **OpenJPEG 2.5.4** is needed for classical codec execution (`pacman -S openjpeg2` on the reference machine). Check the installed library version before the exhibition. For WSL2 with the already-provisioned project `.venv`, you can use `DEMO_PYTHON=.venv/bin/python` in place of `demo/.venv-demo` after installing the demo backend packages. Do not run `uv` without its documented index flags or substitute a CUDA environment when validating a CPU-only installation.

**Shortcut:** once installed, `./run-demo.sh` from the repository root picks the Python environment (`DEMO_PYTHON`, then `demo/.venv-demo`, then the project `.venv`), checks the build and weights, starts the server and opens the browser. `--port 8080` and `--no-browser` are available. To open the dashboard at a chosen measured point, add a query string such as `http://127.0.0.1:8000/?snr=-4` or `?snr=18&ratio=r_1_24`; values outside the measured grid are ignored.

Then disconnect networking and start the **single localhost server**:

```bash
./demo/scripts/start.sh
```

Open `http://127.0.0.1:8000`. `DEMO_PORT=8080` changes the local port; `DEMO_PYTHON=/absolute/path/to/python` selects another installed environment. The launcher binds to **127.0.0.1 only** and serves the built UI and API from one process. Everything needed at runtime is local: source, static build, published CSV/evidence, chosen images and provisioned weights. No CDN or inference service is contacted.

## Frozen weights: provision before disconnecting

Weights are deliberately **not committed**. The W10-bound ordinary learned 1/6 checkpoint is not in the Git checkout; the artifact-aware classical scorer likewise lives outside it. A machine without a verified checkpoint must display that inference path as **unavailable**, never substitute pilot weights or the validation chart's accuracy for a prediction.

The selected W8 learned checkpoint is `/home/nick/w8-final-pascal-20260901-r1/run-01-r_1_6-train0-channel0/checkpoints/epoch-0092.pt` on Confessor, SHA-256 `b0f72a3e16c537984b6afd3dc93bdf3ea87a0cae8a5b49f3565c803750a8826a`. The artifact classifier is `checkpoints/artifact_classifier/epoch-17.pt` there, SHA-256 `468710ba5e6426d2daeaba50af331b498d5d079726476538d69e2fd3b6355ca1` (also published as a GitHub Release asset under tag `g8-f-f2-artifact-classifier-2026-08-25`). If you have authorized Confessor SSH access, stage the two files outside Git, then verify/install:

```bash
mkdir -p /tmp/capstone-demo-weights
scp confessor:/home/nick/w8-final-pascal-20260901-r1/run-01-r_1_6-train0-channel0/checkpoints/epoch-0092.pt /tmp/capstone-demo-weights/learned.pt
scp confessor:/home/nick/projects/capstone/checkpoints/artifact_classifier/epoch-17.pt /tmp/capstone-demo-weights/artifact.pt
python3 demo/scripts/provision_assets.py --learned /tmp/capstone-demo-weights/learned.pt --artifact /tmp/capstone-demo-weights/artifact.pt
python3 demo/scripts/provision_assets.py --verify
```

The installer derives expected digests from the **frozen W10 rehearsal authorization**, verifies source and copied bytes and writes only under ignored `demo/assets/checkpoints/`. On another laptop, copy the two verified files into this directory and run `--verify`; source locations from the original machine are not necessary at runtime. Transfer all dependencies and frozen assets before going offline. Do not commit the checkpoint directory or a generated inference cache.

For a smaller transfer than the whole research checkout, after `npm run build` and asset provisioning, create the bounded portable exhibit archive:

```bash
python3 demo/scripts/package_offline.py /tmp/opencode/capstone-exhibit.tar.gz
```

The ~103 MB archive contains the **two hash-verified weights**, four tiny source-bound example images, the built UI, the Python scientific modules needed for inference, the CPU lock and only the frozen aggregate/selection files used by this adapter. It excludes the multi-GB W10 runtime, source dataset and generated caches. The command prints an archive SHA-256. Extract it on the exhibition laptop, run the install commands above **inside** its `capstone-exhibit/` folder while still online, then disconnect and use `./demo/scripts/start.sh`. The archive intentionally does **not** vendor Python/OS packages or rely on an existing SSH connection.

## What the exhibition can and cannot claim

- The dashboard labels the two systems **Digital link** and **AI link** and shows only those two curves. The SNR control selects **one of 21 actually measured operating points**: −8 through +7 dB in 1-dB steps, then 9, 11, 13, 15 and 18 dB. The published plot is validation accuracy (`n_correct / 1000`) at 1/6 bandwidth, one frozen seed cell, not an image-level confidence or final-test result. Adjacent markers are joined for reading only; no intervening point was measured. The x-axis gives every measured point equal width, so the 1 dB steps from −8 to +7 dB take more room than the 2–3 dB steps above; every tick shows its real value and a break mark sits between +7 and +9 dB. Say so if someone reads distances off the axis.
- The bundled example JPEGs are **original Imagenette training-split bytes**. Their SHA-256 and stable sample IDs are recorded in `assets/examples/manifest.json`; they are useful illustrations, **not held-out accuracy evidence**. Any inference response is a demonstration output and cannot update W10 scientific metrics. No test image is included or opened.
- Arbitrary image upload is not exposed: the frozen learned model and artifact scorer are Imagenette-class specialists, and the offline gallery gives repeatable source-bound examples without accepting unbounded or out-of-vocabulary uploads. This is a deliberate limitation, not a silent sample substitution.
- Classical outage means *no recovered image*. It must not be presented as an image silently restored from the source, nor classified from the source in place of a decoded image. A missing checkpoint, unsupported path or error is displayed explicitly.
- The low-SNR learned result is a **system-level** difference in task-aware representation, receiver task head, source/channel strategy and outage behavior; it is not proof of an intrinsically superior channel code. At 1/6, classical delivery is zero at −8 through −5 dB and reaches 100% at −4 dB; it is more accurate than the ordinary learned curve at all measured points from −4 to +18 dB. Its coverage has some nonmonotone dips.

**Specification styling exception (DR-4):** `params.demo.figure_style_module` names `src/viz/style.py`, which does not exist in this frozen source epoch. The published `presentation-results/plot_results.py` owns its DejaVu Sans and colorblind-safe palette locally. The React dashboard mirrors those actual curve colors (`#0072B2` learned, `#D55E00` adaptive classical, `#009E73` randomized) but does **not** claim to render through the absent Python module. Neither protected sources nor published figures were altered for this cosmetic integration. The specification also lists Streamlit, whereas this user-requested standalone exhibition uses React + FastAPI. The later Streamlit demo (`demo_streamlit/`) does render its figure through the paper's own figure module.

## Three-minute walkthrough

1. **0:00–0:35 — Problem.** “Both links see the same image and simulated wireless noise. The digital path sends a compressed image through error-correcting codes; our DJSCC path learns a representation tuned for the final classification task.” Show the source image and the two receiver cards.
2. **0:35–1:20 — Weak channel.** Start at **−8 dB**. Explain that a lower SNR means noise is stronger relative to the signal. The classical receiver may have no image to decode; do not call the chart's 10% floor a successful delivery. The semantic task head can still produce a prediction, if a frozen checkpoint is available. Differentiate a live output from the population-level validation chart.
3. **1:20–2:10 — Transition.** Move to **−4 dB**; the graph's discrete marker and exact published percentages change immediately. Compare recovered-image status, when actual demo inference is available. The transition was sampled between −5 and −4 dB; there is no measured threshold between them.
4. **2:10–3:00 — Strong channel and limits.** Move to **+18 dB**. Explain why a delivered, well-tuned classical JPEG 2000 + LDPC chain can outperform the standard learned system at stronger SNR. The screen keeps these caveats off the page for visitors, so say them if asked: the chart is validation data from one frozen seed (1,000 images per point), the final test results (3,925 images, three seeds) are in the G-12 demo and the paper, and this comparison does not isolate channel coding alone.

## Checks

```bash
demo/.venv-demo/bin/python -m pytest -q tests/test_demo_backend.py
(cd demo/frontend && npm run test && npm run build)
# With ./demo/scripts/start.sh running in another terminal:
python3 demo/scripts/smoke_api.py
```

The app uses the original closed scientific implementation via imports where supported; it does not modify any source under `src/`, `tools/`, `spec/`, `configs/` or frozen result paths. See `backend/API.md` for endpoint shapes and availability semantics.
