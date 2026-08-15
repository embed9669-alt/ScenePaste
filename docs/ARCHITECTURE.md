# ScenePaste V10 architecture

ScenePaste focuses on **human-reviewed assets + deterministic Copy-Paste**.

## Main pipeline

```text
GenerationConfig / Project Manifest
        |
        +--> DistributionProfile
        +--> Scene Template
        +--> HardCase Recipe
        |
        v
Explicit Label-First Planner
(index + seed -> class + normalized position + scale + difficulty)
        |
        v
Scene-region policy
(explicit/class/forbidden/ground prior)
        |
        v
Deterministic Copy-Paste (paste_one)
        |
        v
Detect / Seg / OBB / COCO / Semantic writers
        |
        v
SQLite RunState + sample metadata
        |
        v
QA / hardmine / compare / curate / shard
```

## Determinism

A sample index has a deterministic task seed. Background sampling advances identically during resume. Label-first plans are generated before workers render pixels.

## Label path

For Copy-Paste, the alpha/transform mask remains the source of segmentation geometry. Z-order visibility is recomputed from canvas-aligned raw masks before Seg/OBB/COCO/Semantic export.

## Stable layers

- Project Manifest
- Distribution Profiles
- Scene Template v2 constraints
- crash-safe SQLite resume
- deterministic bounded multiprocessing
- multi-format mask-first writers
- Dataset Explorer and Data Loop Center
- Detect/Seg/OBB hard mining
- QA/leakage/real-vs-synthetic comparison/diversity selection/WebDataset sharding

Generative backends (SD / Qwen / HTTP / plugin) were removed in the copy-paste-only cleanup.
