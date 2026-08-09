# Performance and large-run guide

ScenePaste uses deterministic per-sample plans, bounded multiprocessing and crash-safe run state. The goal is to scale without making results depend on worker scheduling.

## Multiprocessing

```bash
scenepaste generate ... --workers 0
```

- `--workers 1`: one worker process path, useful for debugging and low-memory systems.
- `--workers 0`: auto = roughly CPU count minus one.
- `--workers N`: explicit process count.
- `--queue-depth 0`: keeps at most about `workers * 2` tasks in flight, so a million-image job does not create a million futures.

Every sample index gets its own deterministic seed. A run using the same seed, inputs and parameters therefore uses the same sample plan regardless of worker count.

Each worker loads its own object library and owns its own decoded-background LRU cache. More workers increase throughput but also increase RAM use. For very large cutout libraries, start with 2–4 workers and increase while monitoring memory.

## Crash-safe resume

Run state is stored in:

```text
<output>/.scenepaste/runs/<run_id>.sqlite3
```

Completed sample indices are committed to SQLite. Per-sample metadata (and COCO fragments when needed) live under:

```text
<output>/.scenepaste/fragments/<run_id>/
```

Resume a specific run:

```bash
scenepaste generate ... --run-id tunnel_aug_01 --resume --workers 8
```

Or omit `--run-id` to resume the most recent incomplete run in that output directory.

ScenePaste verifies a stable configuration hash before resuming. Changing class maps, seed, count, profile/template content or synthesis geometry requires a new run ID.

## Distribution profile

Learning a profile is a one-time preprocessing pass:

```bash
scenepaste profile learn ./real_dataset -o real_profile.json
```

Use it during generation:

```bash
scenepaste generate ... \
  --distribution-profile real_profile.json \
  --profile-strength 1.0
```

`profile-strength=0.5` mixes profile-driven and generic ScenePaste planning.

## Background LRU cache

```bash
--background-cache-size 16
```

The cache is per worker. Set `0` to disable. Very high-resolution backgrounds may justify a smaller cache.

## Preview sampling

```bash
--preview-ratio 0.01
```

For 100k generated images this keeps roughly 1k QA previews instead of performing a second JPEG write for every image.

## COCO at scale

Workers write one small crash-safe fragment per completed image. At finalization, ScenePaste streams those fragments into standard COCO JSON and assigns deterministic image/annotation IDs without loading the whole dataset into a Python annotation list.

## rembg

The optional auto-cutout backend reuses its model session within a process. Multiprocessing means each worker may initialize its own model, so auto-cutout is usually best performed once before a very large generation run rather than inside every worker.

## Practical recommendations

- SSD/NVMe is strongly recommended.
- Keep validation/test data real and independent.
- Use `scenepaste qa <output>` after every important generation run.
- More synthetic images from a tiny object library do not create real object diversity.
- For millions of images, use separate output runs, QA them, then package approved splits with `scenepaste shard`.

## Appearance / QA cost controls

Image-only augmentation recipes add a predictable post-render CPU cost. Use `clean` for throughput benchmarking, then enable only the camera effects that model a plausible deployment domain. JPEG/downscale/motion-blur effects are more expensive than simple brightness/gamma operations.

Perceptual near-duplicate QA is bounded by `--duplicate-limit`. The pHash implementation uses banded candidate lookup rather than scanning every image pair, but very large corpora should still use a representative QA bound or sharded reports.


## Live run telemetry

During advanced generation ScenePaste writes:

```text
<output>/.scenepaste/status/<run_id>.json
```

The status includes completed/failed/requested counts, elapsed time, images/second, ETA and free disk bytes. The Qt large-generation dialog reads this file once per second. It is runtime state and is excluded from source releases.

## Curation cost controls

`cv-lite` embedding and leakage checks are intentionally bounded by CLI limits. For very large datasets, start with representative limits, use sharded reports, or curate per split/domain rather than embedding millions of images into RAM at once. WebDataset sharding computes a SHA-256 per completed tar, which adds one sequential read of each shard but gives a useful transfer-integrity manifest.

## v1.0 reference benchmark

ScenePaste ships `scripts/benchmark_generation.py` so results can be reproduced on your own machine:

```bash
python scripts/benchmark_generation.py --count 1000 --workers 1 4 8
```

Install the development extra (`pip install -e ".[dev]"`) to include `psutil` and report peak total RSS; without it the benchmark still reports throughput.

Reference result measured for the v1.0 release candidate on the OpenAI build container:

- CPU: Intel Xeon Platinum 8573C, **5 available vCPUs**
- RAM available to the container: about **5.9 GiB**
- Python: **3.13.5**
- input: bundled 800×600 public samples
- output: YOLO Detect only
- objects/image: 1–3
- previews: disabled
- count: 1000 images per worker setting

| Workers | images/s | Relative | Peak total RSS* |
|---:|---:|---:|---:|
| 1 | 96.42 | 1.00× | 161 MB |
| 4 | 172.63 | 1.79× | 880 MB |
| 8 | 143.39 | 1.49× | 1506 MB |

\*Peak total RSS is sampled for the launcher process plus live child processes with `psutil`; it is a practical process-memory estimate, not a platform-independent allocator metric.

This benchmark is intentionally published even though **8 workers are slower than 4 on this machine**. The container only exposes 5 vCPUs, and each worker loads its own cutout library/background cache. It demonstrates the intended tuning rule: start around 2–4 workers and increase only while throughput still improves and RAM remains comfortable. High-resolution images, `all` output, COCO/segmentation conversion, augmentation recipes and slower storage will change the numbers substantially.
