# Exhibition demo · G-12 test edition

```bash
./run-g12-demo.sh
```

This is a copy of the exhibition dashboard in `demo/`, with the chart showing the **final G-12 test results** in place of W10 validation data. Run the launcher from the repository root. It opens `http://127.0.0.1:8001` (`--port`, `--no-browser`) and builds the page on first run if `dist/` is missing, which needs Node/npm. `?snr=-4&ratio=r_1_24` deep-links to a measured point.

The three demos:

| Command | Folder | Chart data | Live pictures | Style |
| --- | --- | --- | --- | --- |
| `./run-demo.sh` | `demo/` | W10 validation (1,000 images, one seed) | 4 training images | Visitor dashboard (React) |
| `./run-g12-demo.sh` | `demo_g12/` | G-12 test (3,925 images, mean of 3 seeds at 1/6) | 10 test images, or browse all 3,925 | Visitor dashboard (React) |
| `./run-streamlit-demo.sh` | `demo_streamlit/` | G-12 test, plus the ER-9 control | 10 test images, or browse all 3,925 | Academic, paper figures (Streamlit) |

## What differs from `demo/`

- **Backend.** `backend/app.py` builds the original `demo.backend.app` application and keeps its live inference classes, outage handling, checkpoint verification and input validation. It replaces the metadata, chart, gallery and inference routes, and adds `/api/test-split` and `/api/test-images/{id}` for browsing. The weights are shared, so provisioning `demo/` (see `demo/README.md`, "Frozen weights") provisions this demo too.
- **Test images (AM-101).** The default row is ten test images, one per class: for each class, the one whose stable sample ID sorts first. They are bundled in `demo_shared/test_examples/` with their G-12 records, so they work on any laptop. "Browse all test images" opens every one of the 3,925, filterable by class. Browsing needs the extracted dataset and the per-image records in `results/per_image/` (release `g12-test-per-image-2026-10-02`). On first use it caches the source JPEGs in the ignored `demo_shared/.cache/`, about 6 s; later starts take about 1 s. All test reads go through `data.test_access`. Each receiver card shows what G-12 recorded for that image at that SNR in seed pair 1 of 3, and whether the live result is the same. For the ten default images, all 420 image–SNR–system checks matched when this was written.
- **Chart data.** `presentation-results/data/g12_test_curves.csv`, read through `demo_shared/results.py`. That module checks every served point against `results/g12/results.csv` at start-up. At 1/6 the curves are means over three seed pairs and carry 95% image-bootstrap bands. At 1/24 they are one seed pair, so no band is drawn, as in the paper.
- **Frontend.** The interval band, the "Final test · 3,925 pictures" heading, and the low-SNR sentence now quotes the measured AI accuracy. The old wording, "most pictures right", is false at 1/24, where DJSCC scores 47% at −8 dB.

## Presenter notes

- The chart is the final test result: 3,925 held-out images, and at 1/6 the mean of three seed pairs. The pictures are test images too. Each card's "In the final test" line is the recorded G-12 outcome for that image, and the live run reproduces it. The default ten were picked by rule, not by result: DJSCC gets the gas pump wrong at almost every SNR, and that stays in.
- "Correct" for the digital link during an outage only means the fallback class happened to be right. The tench image is the example.
- At 1/6 the digital link delivers nothing from −8 to −5 dB and everything from −4 dB up. At −4 and −3 dB the two links are within a point of each other. From −2 dB up the digital link leads by about 2–4 points. The digital baseline is re-tuned per SNR, while DJSCC was trained once at 7 dB.
- Live inference and per-image records cover 1/6 only. At 1/24 both cards say "unavailable", and nothing is substituted.

## Checks

```bash
.venv/bin/python -m pytest -q tests/test_demo_g12_backend.py tests/test_demo_test_gallery.py
(cd demo_g12/frontend && npm run test && npm run build)
DEMO_PORT=8001 .venv/bin/python demo_g12/scripts/smoke_api.py   # with the demo running
```
