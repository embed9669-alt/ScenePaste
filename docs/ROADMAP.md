# ScenePaste Roadmap

## v1.0.0 — Stable (current)

Release-complete baseline:

- portable Project Manifest and project-aware generation;
- unified Qt Data Loop Center;
- Detect / Seg / OBB Hard Example Mining;
- overlap/crowding-aware distribution profiles and rendered visibility diagnostics;
- optional CLIP / DINOv2 embedding backends with `cv-lite-v1` offline by default;
- deterministic resumable multiprocessing, relation-constrained templates, augmentation recipes, QA/leakage, real-vs-synthetic comparison, curation and WebDataset sharding;
- bundled wheel-installable sample resources;
- Windows / Linux / macOS Qt test matrix plus non-editable installed-wheel smoke workflow;
- documented 1/4/8-worker benchmark procedure and memory guidance.

## v1.1 candidates

- true virtualized/paginated asset browser for very large libraries;
- explicit layer panel with move-up / move-down / top / bottom controls;
- GUI-first automatic cutout workflow;
- direct-to-shard generation to avoid huge loose-file directories;
- shared-memory/read-only cutout store to reduce per-worker asset duplication;
- configurable/local foundation-model paths and richer GPU embedding extraction;
- model-specific adapters beyond YOLO-style TXT;
- richer joint scene distribution learning (class co-occurrence and pairwise spatial relations).

## Research / longer-term

- learned object-to-background compatibility scoring;
- depth/ground-plane assisted scale constraints;
- generative inpainting/harmonization as an optional post-process;
- active-learning connectors to training/annotation platforms.
