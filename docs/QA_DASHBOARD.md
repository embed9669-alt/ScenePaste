# QA Dashboard

ScenePaste provides two levels of dataset QA:

```bash
scenepaste analyze ./generated
scenepaste qa ./generated
```

`analyze` is a quick terminal-oriented integrity/statistics check. `qa` produces both machine-readable JSON and a standalone HTML dashboard.

## Generate the dashboard

```bash
scenepaste qa ./generated
```

Default outputs:

```text
qa_report.json
qa_dashboard.html
```

Custom paths:

```bash
scenepaste qa ./generated \
  --json reports/run_a.json \
  --html reports/run_a.html
```

## Checks

Depending on the available annotation formats, the report covers:

- image/label counts and annotation integrity;
- invalid or out-of-range annotations;
- unreadable images;
- exact duplicate-image hashes (bounded by `--duplicate-limit`);
- perceptual near-duplicates via 64-bit pHash plus a compact diversity ratio;
- cv-lite visual embedding uniqueness;
- cross train/val/test exact / pHash / high-embedding-similarity leakage;
- class distribution;
- object scale/position/crowding histograms, including overlap-IoU proxies when geometry is available;
- object source reuse and background reuse;
- target-vs-generated distribution drift when a target profile is available;
- per-run actual rendered visibility diagnostics for mask-aware generation modes, including per-class visible-ratio summaries and augmentation-effect counts.

When generation used a distribution profile, the relevant target profile is copied into the output dataset so QA can compare intended and observed distributions.

## Embedding backends

The default QA/curation descriptor is `cv-lite-v1`: local, deterministic and zero-download. For semantic similarity you can opt into CLIP or DINOv2:

```bash
python -m pip install -e ".[embeddings]"
scenepaste qa ./generated --embedding-backend clip
scenepaste curate leakage ./generated --embedding-backend dinov2
```

Foundation-model weights are loaded lazily and cached in-process. ScenePaste batches model inference and uses CUDA automatically when PyTorch reports an available GPU. If the optional dependencies or model weights cannot be loaded, the command fails explicitly instead of silently reporting an empty embedding result.

## Generation diagnostics

Mask-aware output modes (`seg`, `both`, `coco`, `semantic`, `obb`, `all`) record actual z-order-aware visible fractions during rendering. The latest run writes `latest_generation_diagnostics.json`, and the QA dashboard displays this data when present. Detect-only generation intentionally avoids this extra mask projection cost.

## Visual inspection still matters

A green QA report does not prove that a composite is visually realistic. Use:

```bash
scenepaste explore ./generated
```

or the **Dataset Explorer → QA Dashboard** actions in the GUI to combine automatic validation with human inspection.


ScenePaste keeps exact byte duplicates, pHash near-duplicates and cross-split embedding leakage as separate signals. `cv-lite-v1` is a lightweight offline descriptor for curation, not a semantic foundation-model embedding. `--near-duplicate-threshold` controls the maximum Hamming distance (0–7) used for the fast perceptual check.
