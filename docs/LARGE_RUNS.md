# Resumable multiprocessing and large runs

ScenePaste uses deterministic per-index planning for every generated sample. This makes one-worker and multi-worker runs reproducible while allowing interrupted runs to resume safely.

## Start a large run

```bash
scenepaste generate \
  --objects ./objects \
  --backgrounds ./backgrounds \
  --output ./generated \
  --class-map "person=0,truck=1" \
  --count 500000 \
  --workers 8 \
  --queue-depth 16 \
  --run-id production_a \
  --preview-ratio 0.01
```

`--workers 0` automatically chooses approximately `CPU count - 1` processes. Start conservatively when the cutout library is large because each worker loads its own object assets.

## Crash-safe state

ScenePaste writes run state to:

```text
<output>/.scenepaste/runs/<run_id>.sqlite3
<output>/.scenepaste/fragments/<run_id>/meta/
<output>/.scenepaste/fragments/<run_id>/coco/   # COCO mode only
```

A sample is marked complete only after its image/annotation output and metadata fragment have been written. If the parent process is interrupted after a worker wrote files but before completion is recorded, that index is safe to regenerate and overwrite during resume.

## Resume

Rerun the same command with:

```bash
--run-id production_a --resume
```

Or omit `--run-id` with `--resume` to use the latest incomplete run in that output directory.

ScenePaste validates a stable configuration hash before continuing. Changes to important generation settings, class mapping, profile contents or template contents prevent an unsafe resume.

## Determinism

Planning is derived from `seed + sample index`, not from process scheduling. With the same inputs/configuration, worker count does not change the planned sample sequence. Background sampling is also advanced deterministically across completed indices when resuming.

## COCO at scale

COCO annotations are written as per-sample fragments and assembled into the final JSON in one streaming finalization pass. This avoids repeatedly loading and rewriting an ever-growing `instances_coco.json` during generation.

## Recommended settings

For large jobs:

- use `--preview-ratio 0.01` or lower;
- keep `--queue-depth` near `workers * 2` unless profiling shows a reason to change it;
- use local SSD/NVMe output when possible;
- pre-create cutouts instead of running `--auto-cutout` inside many worker processes;
- lower worker count if RAM usage is high;
- periodically run `scenepaste qa` on completed data.

After QA, package approved splits with `scenepaste shard` into deterministic WebDataset-compatible tar files. Direct worker-to-shard generation remains optional future work.


## Live status

Long runs update `<output>/.scenepaste/status/<run_id>.json` with completed/failed/requested counts, images/second, ETA and free disk bytes. The Qt large-run dialog reads this file live.
