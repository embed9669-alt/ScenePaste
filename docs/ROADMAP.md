# ScenePaste Roadmap

## v1.0.0 — Stable baseline

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

## v1.1.0 — Object Appearance (current stable)

Phase 1 (landed in tree):

- Object Appearance Recipes (`off` / `legacy` / `mild` / `surveillance-object`);
- per-cutout brightness/contrast, sat/hue, gamma, color temperature, blur, noise, JPEG, motion blur, **resolution degrade**;
- class-aware `by_class` overrides;
- per-instance `object_effects` metadata + diagnostics counts;
- CLI `--object-appearance-recipe` and `scenepaste recipe --kind object …`;
- GUI large-generate recipe picker (blank keeps v1.0 compatibility; `mild` is recommended).

Phase 2 landed:

- project defaults for object appearance recipes, including portable relative JSON paths;
- QA dashboard coverage tables for object-level and scene-level effects;
- alpha-aware edge-safe blur / motion blur / resolution degradation;
- strict recipe validation and worker-level recipe preloading.

Still planned for a later UI-focused release:

- interactive per-instance appearance preview in the scene editor.

Phase 3 (optional, later):

- region-aware recolor (vehicle paint / clothing masks);
- generative editing as `scenepaste[generative]` — not core.

## v1.2 candidates

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
