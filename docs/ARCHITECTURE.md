# ScenePaste architecture


## v1.0 project-centered workflow

`scenepaste.project.json` is the portable workflow anchor. It points to assets, backgrounds, generated output, real/validation datasets, model predictions, distribution profiles and scene templates. Both CLI generation and the Qt editor can consume the same manifest.

The Qt **Data Loop Center** is deliberately an orchestration layer over the CLI (`QProcess`), not a second implementation of QA/mining/curation logic. This keeps GUI and headless behavior identical.

Hard Mining now normalizes Detect boxes, Seg polygons and OBB quadrilaterals into a shared polygon representation and computes geometry IoU through mask rasterization. Distribution profiles separately model observable crowding (`max overlap IoU`) and polygon/mask shape occupancy; generated scenes record actual z-order-aware visible ratios in per-run diagnostics.

Similarity remains pluggable: `cv-lite-v1` is built in and fully offline; optional CLIP/DINOv2 backends are loaded lazily when explicitly selected.

ScenePaste separates **planning**, **execution**, **annotation writers**, **dataset tools**, and the **desktop UI**. v1.0 uses the same deterministic planner for one or many workers and adds a dataset-curation layer after generation/evaluation.

## Public interface

```text
scenepaste gui
scenepaste generate
scenepaste profile learn|show
scenepaste explore
scenepaste qa
scenepaste analyze
scenepaste split
scenepaste merge
scenepaste curate hardmine|leakage|diversity
scenepaste compare
scenepaste shard
```

The Python API is exported from `scenepaste.__init__`.

## Generation architecture

```text
GenerationConfig
     |
     +--> optional DistributionProfile
     +--> optional Scene Template v2
     |
     v
Deterministic Sample Planner
(index + seed -> background + object placement specs)
     |
     +-------------------------------+
     | bounded queue                 |
     v                               v
 Worker 1        Worker 2 ...      Worker N
 assets/cache    assets/cache      assets/cache
     |                               |
     +------------ outputs -----------+
                   |
             per-sample fragments
             images / labels
                   |
             SQLite RunState
                   |
         CSV / streamed COCO finalize
                   |
              QA Dashboard
```

A sample index has a deterministic seed, so worker scheduling does not change its planned scene. Completed indices are stored in SQLite. Pending indices are implicit, avoiding a million-row task table before a large run starts.

## Distribution profiles

`scenepaste.core.distribution` learns compact histograms rather than copying source images. Profiles contain:

- class frequencies;
- objects/image counts;
- center-X and center-Y;
- bbox bottom-Y;
- width / height / area;
- aspect ratio.

The learner supports LabelMe, YOLO Detect/Seg/OBB and COCO. Profiles can be compared by QA to detect synthetic distribution drift.

## Scene Template v2

`scenepaste.core.templates` owns the framework-neutral schema. `compose_app_qt.templates` is only a Qt Document adapter.

A template stores a nominal human-designed scene plus optional per-instance/global ranges for position, scale, angle, occurrence, flipping, same-class asset replacement and overlap.

## Mask-first annotation path

```text
source alpha
  -> resize / flip / rotate / translate
  -> per-instance canvas mask
  -> z-order occlusion
  -> final visible mask
       -> YOLO segmentation polygon
       -> COCO polygons + bbox + area
       -> semantic class mask
       -> oriented bounding box
```

The five-element core transform `(scale, x, y, flip, angle)` is reconstructable, so rotated parameterized templates remain consistent across pixel-aware exports.

## Crash-safe run state

`scenepaste.core.runstate` stores state in:

```text
<output>/.scenepaste/runs/<run_id>.sqlite3
```

Workers atomically write per-sample metadata and (for COCO) annotation fragments under `.scenepaste/fragments/<run_id>/`. The coordinator marks a task complete only after its fragments exist. Recovery can therefore regenerate an unconfirmed index without duplicating final CSV/COCO rows.

## Dataset tools

- `analyze.py` — fast CLI statistics/validation;
- `qa.py` — full JSON + standalone HTML QA dashboard;
- `split.py` — source-aware train/val splitting;
- `merge.py` — collision-safe merging;
- `hardmine.py` — model failure scoring and hard-profile feedback;
- `leakage.py` — cross-split exact/perceptual/embedding leakage checks;
- `diversity.py` — lightweight visual uniqueness and representative subset export;
- `compare.py` — real-vs-synthetic distribution/domain comparison;
- `shard.py` — deterministic WebDataset-compatible tar packaging.

## Desktop UI

`compose_app_qt/` contains the PySide6 editor, Dataset Explorer, parameterized template dialog and a large-generation dialog. Large generation launches the CLI through `QProcess`, isolating Qt from multiprocessing worker crashes and making stop/resume behavior clearer on Windows/macOS/Linux.

## Release packaging

`scripts/build_release.py` stages and verifies a clean source ZIP. `.git`, caches, coverage artifacts, egg-info, generated data and user asset directories are forbidden in a release.

## Extension rule

New behavior should become a planner policy, renderer/annotation policy, dataset tool or UI controller—not another standalone copy of the generator.

## Quality policies

ScenePaste keeps quality controls as explicit policies around the deterministic planner rather than mixing them into annotation writers:

- `core.recipes` owns image-only camera/domain augmentation recipes;
- `core.labelme` resolves generic and class-specific background placement masks;
- `core.templates` samples parameterized slots and validates cross-slot relations;
- `core.distribution` learns and mixes normalized real-data domain profiles;
- the advanced pipeline renders once, derives mask-first visible geometry, then may fan out to every supported writer with `output_format=all`;
- `tools.qa` separates exact byte duplicates from perceptual pHash near-duplicates.

This separation keeps scene geometry, appearance randomization and output serialization independently testable.


## Curation loop

```text
real validation set + model predictions
              |
              v
     Hard Example Mining
       |             |
 hard profile   hard negatives
       |             |
       +------> generator <------+ real distribution profile / scene template
                         |
                         v
                    synthetic set
                         |
          +--------------+----------------+
          |              |                |
          v              v                v
     QA/leakage    real-vs-synth     diversity select
          |              |                |
          +--------------+----------------+
                         |
                         v
                  shard / train / evaluate
```

`core.similarity` supplies the lightweight cv-lite descriptor. The curation tools deliberately stay outside annotation writers so similarity policy can later be swapped for an optional semantic embedding backend.
