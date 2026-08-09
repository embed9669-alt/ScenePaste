# Changelog

## [Unreleased]

### Added
- Scene editor **目标外观预览** panel: per-instance enable/recipe/sliders/resample with live canvas preview (RGB-only; labels unchanged).
- Main-window **批量生成默认** panel (scene recipe / object appearance / blend / empty-scene) shared by 批量套用, 大规模生成, project save/load, and project settings.

## [1.1.0] — 2026-08-09

### Added
- **Object Appearance Recipes** for per-cutout diversification (`mild`, `surveillance-object`, `legacy`, `off`) with brightness/contrast, saturation, hue, gamma, color temperature, blur, noise, JPEG, motion blur and resolution degrade.
- Class-aware `by_class` overrides in object appearance JSON.
- `--object-appearance-recipe` on `scenepaste generate`; `scenepaste recipe --kind object list|show|export`.
- Per-instance `object_effects` in generation metadata fragments and `object_effect_counts` in run diagnostics.
- Large-generate GUI picker for object appearance recipes; blank keeps v1.0-compatible legacy behavior, `mild` is the recommended opt-in starting point.
- Docs: `docs/OBJECT_APPEARANCE.md`.

### Changed
- Object recipes are preloaded once per generation worker instead of re-reading custom JSON for every pasted instance.
- Color matching now uses alpha-weighted object-footprint statistics rather than the whole bounding rectangle.
- Blur, motion blur and low-resolution degradation use alpha-aware filtering to avoid pulling mask-exterior pixels into object edges.
- Recipe validation rejects unknown effects/fields and validates ranges; `hue.range` is explicitly measured in degrees.
- QA Dashboard adds object-level and scene-level appearance coverage tables.
- Project-relative custom recipe paths resolve from `scenepaste.project.json`, and the large-generation GUI inherits project defaults.
- Added optional object-level sharpness variation.

### Compatibility
- Leaving `--object-appearance-recipe` unset keeps the historical light HSV + `blur_prob` path.

## [1.0.0] — 2026-08-09

### Stable release

- Promoted the v1.0 release line from RC to Production/Stable.
- Corrected README claims to match the current UI: responsive/searchable thumbnails and z-order-aware occlusion, without claiming a pagination or dedicated layer panel that is not present yet.
- Bundled the public sample dataset inside the installed Python package so **★ Load Sample** works after a normal wheel install.
- Added explicit `scripts/__init__.py` for reliable source-tree release-helper imports.
- Raised the supported Python baseline to 3.10 and aligned metadata/CI across Python 3.10–3.13.
- Added a non-editable installed-wheel smoke workflow covering bundled samples, offscreen Qt sample load, template save, all-format generation, Explorer, QA and sharding.
- Added a reproducible 1/4/8-worker benchmark helper and release performance guidance.

## [1.0.0rc1] — 2026-08-09

### Added
- Portable `scenepaste.project.json` Project Manifest with CLI + GUI open/save support.
- Unified Qt **Data Loop Center** for profile learning, Detect/Seg/OBB hard mining, QA/leakage, real-vs-synthetic comparison, diversity curation and WebDataset sharding.
- Detect, Segmentation and OBB Hard Example Mining through a shared polygon-IoU matcher.
- Distribution-profile crowding statistics (`overlap_iou`) and polygon/mask visible-shape occupancy.
- Per-run generation diagnostics with actual rendered visible ratios for mask-aware output modes.
- Optional CLIP and DINOv2 embedding backends; dependency-free `cv-lite-v1` remains the default.
- `scenepaste project ...` and `scenepaste loop ...` workflows.

### Changed
- `profile learn --geometry-source auto|detect|seg|obb` can prefer richer YOLO geometry.
- QA Dashboard now exposes crowding/visibility statistics and generation diagnostics.
- Hard-profile learning can use Seg/OBB ground-truth geometry.
- Project-aware `scenepaste generate --project ...` fills paths, class map, profile/template and selected defaults.

### Compatibility
- Existing v0.9 CLI commands, generation layouts and deterministic run/resume semantics remain supported.
- Detect-only generation avoids extra visible-mask projection cost; precise visible-ratio diagnostics are available when Seg/OBB/COCO/Semantic/all masks are already required.

### Verified
- 170 headless tests pass in the RC release environment; 11 PySide6-dependent Qt modules are skipped locally and remain CI-covered.
- Project-driven headless generation, all-format export, QA diagnostics, Detect/Seg/OBB Hard Mining and WebDataset sharding were exercised end-to-end.

## [0.9.0] — 2026-08-09

### Added
- **Hard Example Mining** for YOLO-style detection predictions: FN, FP, low-confidence TP and localization-IoU difficulty scoring.
- Hard-example JSON/CSV/list + standalone HTML dashboard, hard-negative background list and reusable hard DistributionProfile.
- Cross train/val/test leakage checks using exact SHA-1, pHash near duplicates and lightweight cv-lite high-similarity signals.
- Lightweight visual diversity/uniqueness analysis plus greedy representative subset export with matching labels/masks/COCO.
- Real-vs-synthetic comparison JSON + HTML for class/count/geometry distributions and visual-domain similarity.
- Deterministic WebDataset-compatible tar sharding with same-basename multimodal samples and per-shard SHA-256 manifest.
- Long-run telemetry JSON containing throughput, ETA, failed count and free disk space; Qt large-generation dialog displays it live.
- Unified `scenepaste curate`, `scenepaste compare`, and `scenepaste shard` commands.

### Changed
- Distribution-profile comparison now covers object-count, center, size, area and aspect-ratio histogram distances.
- QA Dashboard now reports cv-lite diversity and cross-split leakage in addition to v0.8 exact/pHash checks.
- Roadmap moves from adding random augmentation knobs toward a measurable model→data→model improvement loop.

### Scope
- v0.9 hard mining targets object detection predictions; segmentation/OBB hard mining remains future work.
- `cv-lite-v1` is the zero-download default descriptor and is not presented as a semantic foundation-model embedding.
- WebDataset export is a post-generation packaging step, preserving normal ScenePaste resume/QA semantics.

### Verified
- 155 headless tests pass in the release environment; 11 PySide6-dependent Qt modules are skipped locally and remain CI-covered.

## [0.8.0] — 2026-08-09

### Added
- Built-in and JSON-editable **Augmentation Recipes** (`clean`, `camera-mild`, `surveillance`, `low-light`) with deterministic per-sample metadata.
- Foreground blend modes: `alpha`, `hard`, and Gaussian alpha-edge blending.
- Pure-background negative scenes via `--empty-scene-prob`.
- Class-specific LabelMe placement zones such as `paste_zone:person`, `zone:truck`, and `ground:forklift`.
- Scene Template relation constraints: `left_of`, `right_of`, `above`, `below`, `min_distance`, and `max_distance`.
- Weighted multi-domain DistributionProfile mixing (`scenepaste profile mix`).
- One-pass multi-task export (`--output-format all`) for Detect + Seg + OBB + Semantic + COCO.
- pHash perceptual near-duplicate detection and a compact diversity signal in QA.
- `scenepaste recipe list/show/export` CLI and bundled recipe examples.
- OBB handling in Dataset Explorer, split, and merge utilities.

### Changed
- Placement planning now prefers a matching class-specific zone before the generic paste zone.
- Exact byte duplicates and perceptual near-duplicates are reported as separate QA signals.
- Scene Template v2 remains backward compatible while accepting stable slot IDs and relation constraints.
- v0.8 documentation now separates scene geometry randomization from image-only camera/domain recipes.

### Verified
- 147 headless tests pass in the release environment; Qt-dependent tests remain CI-covered when PySide6 is available.

## [0.7.0] — 2026-08-09

### Added
- Real-data distribution learning from LabelMe, YOLO Detect/Seg/OBB and COCO (`scenepaste profile learn`).
- Distribution-driven class/count/position/scale generation with configurable `--profile-strength`.
- Parameterized Scene Template v2 with position/scale/angle ranges, instance/flip probabilities, same-class asset variation and overlap control.
- Deterministic bounded multiprocessing (`--workers`, `--queue-depth`) with worker-count-independent per-index plans.
- Crash-safe SQLite run state and `--resume`; per-task metadata and COCO fragments make recovery idempotent.
- Streaming COCO finalization for large runs.
- Full QA JSON + HTML dashboard (`scenepaste qa`) with duplicates, integrity, class/scale/position distributions, reuse and target-vs-generated drift.
- Qt large-generation dialog, GUI distribution-profile learning, parameterized-template dialog and QA Dashboard action.
- Bundled parameterized scene-template example and advanced headless tests.

### Changed
- The v0.7 generator now uses one unified deterministic planner even with `workers=1`.
- Rotation is supported by the core mask-first paste transform, so parameterized templates preserve angle in Seg/COCO/Semantic/OBB geometry.
- Run metadata uses SQLite state while preserving the v0.5 CSV columns for compatibility.

### Verified
- 137 headless tests pass in the release environment; Qt-dependent tests remain CI-covered when PySide6 is available.

## [0.5.0] — 2026-08-09

### Public Beta
- Unified the supported user interface around the single `scenepaste` command:
  `gui`, `generate`, `explore`, `analyze`, `split`, and `merge`.
- Finalized the public package version as `0.5.0` and aligned README, package metadata,
  issue templates, release names and GUI version reporting.
- Kept top-level dataset utility scripts only as source-tree compatibility wrappers;
  implementations now live under `scenepaste.tools`.

### Dataset Explorer
- Added `scenepaste explore <dataset>` and a Dataset Explorer action in the desktop editor.
- Added a headless annotation parser/overlay renderer in `scenepaste.explorer` so visualization
  logic can be tested without Qt.
- Explorer recognizes YOLO Detect, YOLO Seg, Ultralytics OBB, COCO Instances and Semantic masks.

### Scene-first workflow
- Added bundled scene-template examples for distant-person, person-near-truck and mixed-traffic layouts.
- Added a generated before/after example to the README and dedicated scene-template documentation.
- Retained mask-first visible annotation export, balanced sampling, run-safe metadata, sampled previews,
  background LRU caching, QA, merge and leakage-aware split behavior from the 0.4 line.

### Packaging / release quality
- Reworked `pyproject.toml` package discovery so `scenepaste.tools` and future subpackages ship automatically.
- Simplified installed console entry points to the canonical `scenepaste` command.
- Rebuilt `scripts/build_release.py` to derive the version from project metadata and verify the archive.
- Release ZIP explicitly excludes `.git`, caches, coverage artifacts, egg-info, generated datasets and user asset folders.

### Verification
- Added headless tests for Dataset Explorer overlays and clean-release packaging.
- Current container verification: 131 tests pass; Qt-dependent modules are skipped when PySide6 is unavailable
  and are configured to run in GitHub Actions with `QT_QPA_PLATFORM=offscreen`.

## [0.4.1] — 2026-08-09

### ScenePaste branding / public interface
- Renamed the project and Python distribution to **ScenePaste / `scenepaste`**.
- Added a unified `scenepaste` command with `gui`, `generate`, `analyze`, `split` and `merge` subcommands.
- Added the public `scenepaste` Python package so new code can import `GenerationConfig` and `generate_dataset` without relying on the historical internal module name.
- Updated GUI titles, scene-template schema, install hints, release archive naming and all public documentation.
- Reworked README first-run guidance, project positioning, limitations and roadmap.
- Added `docs/ROADMAP.md`.


## [0.4.0] — 2026-08-09

### Correctness
- Added a mask-first visible-instance annotation pipeline. Occluded pixels are removed before segmentation-derived annotations are exported.
- Fixed semantic segmentation class `0` collision: mask value `0` is background and dataset class `N` maps to `N+1`; mapping is saved in `semantic_classes.json`.
- OBB output now follows Ultralytics four-corner format (`class x1 y1 ... x4 y4`) instead of internal `xywhr` notation.
- GUI and CLI use the same visible-mask rules for segmentation, COCO, semantic masks and OBB.

### Performance
- `CocoWriter` uses O(1) ID allocation, shared writers and configurable atomic checkpoints instead of rewriting the full JSON for every generated image.
- GUI COCO batch generation reuses one writer for the whole job.
- Background LRU decoding cache is now actually connected to generation.
- rembg reuses a model session across images.
- Thumbnail library is paginated/lazy and keeps only the current page's Tk `PhotoImage` objects alive.
- Preview output can be sampled with `--preview-ratio`.

### Controlled generation
- Added class-balanced asset sampling to avoid class frequency being determined by the number of cutout files.
- Added balanced shuffled background sampling to reduce background reuse skew.
- Added GUI scene-template save/load. Templates store relative layout, scale, flip, angle and source identity and can be reused across different background resolutions.

### Dataset tooling
- Rebuilt dataset analyzer into a QA tool for detection, segmentation, OBB, semantic masks and COCO.
- Rebuilt split tool to resolve the latest run, preserve source groups, and split accompanying labels/masks/COCO annotations.
- Rebuilt merge tool with class-map validation, collision-safe names and COCO ID remapping.
- Legacy `batch_generate*.py` scripts now delegate to the single core generator rather than maintaining duplicate generation algorithms.

### Project / release
- Previous development package name standardized as `copy-paste-dataset-studio`, version `0.4.0`.
- Added English/Chinese README, architecture/format/performance/user docs, CI, issue/PR templates, Windows/Linux launch helpers and a clean-release builder.
- Removed fake author/repository metadata and Unicode-dependent helper filenames from the release layout.
- Test suite expanded to 86 tests in this development snapshot.

## [0.3.1] — 2026-08-09
- Fixed real CLI auto-cutout integration and auto class IDs.
- Fixed segmentation transform geometry.
- Fixed batch cancellation race.
- Added run-specific metadata and initial reusable COCO writer.
- Removed hard-coded local analyzer paths.

## [0.3.0] — 2026-08-08
- Added segmentation/COCO output, optional auto-cutout, theme system, render caching and bilingual documentation.

## [0.2.0]
- Added interactive editor, undo/redo, batch apply and placement augmentation.

## [0.1.0]
- Initial LabelMe-to-YOLO Copy-Paste generator.
