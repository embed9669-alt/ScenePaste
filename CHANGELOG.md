# Changelog

## [10.0.0] — 2026-08-14

### Copy-Paste only cleanup
- Removed generative backends and tooling: Diffusers / light-inpaint, Qwen local+HTTP bridge, opencv-harmonize generative path, `scenepaste model` / `scenepaste qwen`, and the `[generative]` optional dependency.
- Generation pipeline is Copy-Paste only (`paste_one`); GUI batch factory no longer exposes AI schemes or model download/serve UI.
- `scenepaste factory` now only emits label-first plans (`factory plan`).
- UI polish: renamed misleading labels (`加载目标素材…`, menu `合成`, tab `批量默认`), merged empty factory scheme tab into data strategy, clarified cutout entry points, synced Asset Studio docs.

### Asset Studio mainline
- Added a new **Asset Studio** desktop workflow for human review/editing of existing segmentation data.
- Split editing into an authoritative **Foreground Mask** and an independent **Background Removal Mask**.
- Added brush add/erase, mask undo/redo, hole filling, dilation/erosion, de-noise, foreground preview, background-removal preview and clean-background preview.
- Added reviewed asset export: transparent foreground + LabelMe JSON, clean background, original/auto/edited/removal masks and provenance `metadata.json`.
- Promoted Asset Studio to the welcome page, toolbar and Assets menu.
- Reframed the batch factory around **Controlled Copy-Paste** as the only generation scheme.
- Added headless mask/background/export tests so the main asset pipeline is testable without PySide6.

## [9.9.0] — 2026-08-14

### Identity-preserving lightweight generation

- Changed `light-inpaint` from full-object regeneration to identity-preserving boundary harmonization: the deterministic project asset core is locked and only a narrow inner edit band is sent to Diffusers.
- `generator_strength` now defaults to `0.35`, is actually forwarded to the lightweight Diffusers pipeline, and is additionally capped by a conservative soft blend.
- Rewrote the default prompt/negative prompt to preserve the exact same input object rather than let the diffusion prior replace or redesign it.
- Added rectangle-annotation safety: data-factory runs default to OpenCV GrabCut foreground refinement for LabelMe detector boxes. Failed refinements are rejected instead of falling back to full rectangular source tiles.
- Added `--rectangle-mask-mode {grabcut,reject,legacy}` and per-asset `mask_source` provenance.
- Generator provenance now records source asset path, shape index and mask source.
- Debug comparisons can include a dedicated `AI Edit Band` beside the semantic Label Mask.
- Updated Qt copy and defaults to explain identity-preserving light inpainting and migrate old 0.70/0.75 defaults to 0.35.
- Added regression tests for project-asset core locking, strength forwarding, legacy rectangle rejection, GrabCut bbox refinement, and identity-preserving prompts.

## [9.8.2] — 2026-08-14

### Fixed
- Lightweight/Qwen model downloads now default to Hugging Face HTTP compatibility mode by setting `HF_HUB_DISABLE_XET=1` before importing `huggingface_hub`; this avoids common `cas-server.xethub.hf.co` / Xet CAS 401 failures on mirrors and restricted networks.
- Added `--use-xet` as an explicit opt-in for users who want hf-xet/CAS acceleration.
- Xet/CAS 401/403 errors are now classified separately from normal Hugging Face token authentication failures.
- The GUI explains that HTTP compatibility mode is enabled and keeps partial `local_dir` downloads resumable.

## [9.8.1] — 2026-08-14

### Fixed
- Fixed a GUI startup crash in the AI Data Factory caused by an undefined `needs_model` name in `_update_factory_hints()`.
- Added a headless AST regression test for undefined name references in the factory hint initializer so this class of GUI-only NameError is caught even when PySide6 is unavailable in the packaging environment.

## [9.8.0] — 2026-08-14

### Open-source desktop UX finalization
- Replaced the narrow empty-state banner with a dedicated Welcome Home centered on **Start AI Data Generation**, recent projects and a concise three-step onboarding path.
- Reorganized menus into File / Assets / Scene / Edit / Generate / Data / View / Help and expanded Help with quick start, model docs, shortcuts, GitHub and issue reporting.
- Simplified the main toolbar around project/assets, the primary AI Data Factory action, save and undo/redo.
- Added `QSettings` persistence for window geometry, splitter proportions, theme, side-panel visibility, recent projects and recently used asset folders.
- Renamed the desktop inspector tab to **Batch Generation** and the data-factory system tab to **Runtime & Performance** for clearer mental models.
- Simplified the model card: common users see model status/path/install first, while repository ID and download source move into an advanced disclosure section.
- Qwen-only inference controls are now dynamically hidden unless local Qwen is selected.
- Added **Try 10 samples**, producing an isolated `_trials/<run-id>` dataset before a long production run.
- Expanded the generation dashboard with Label-QA rejection and model-fallback metrics; final diagnostics now aggregate fallback count.
- Added source-level UI regression coverage for the welcome flow, menu structure, persistence, trial generation and dynamic advanced controls.

## [9.7.0] — 2026-08-14

### Lightweight AI as the default generative path
- Replaced local 20B Qwen as the recommended desktop AI workflow with `light-inpaint`, a mask-native local Diffusers inpainting backend intended for ordinary workstations.
- Added `scenepaste model download --preset light-inpaint` and `scenepaste model doctor --preset light-inpaint`.
- The lightweight downloader fetches only the FP16 Diffusers components ScenePaste needs and skips duplicate FP32 / legacy checkpoint formats.
- Added local model auto-discovery via `SCENEPASTE_INPAINT_MODEL` and common `models/stable-diffusion-inpainting` paths.
- Added ROI-cropped 512-side inference, attention/VAE slicing/tiling, automatic low-VRAM CPU offload, and hard restoration outside the label-first mask.
- Updated the Qt data-factory presets so lightweight SD Inpainting is recommended, while local Qwen is explicitly labeled as a 20B high-resource advanced mode.
- Added per-preset resource hints and strict disk-space preflight: 4.5 GB recommended free space for the compact lightweight download and 55 GB for full local Qwen.
- Local Qwen download now fails before network transfer when the selected disk cannot safely hold the checkpoint, with suggestions to use lightweight inpainting, a larger disk, or Qwen service mode.
- Rewrote README model guidance around the lightweight-first strategy and added `docs/LIGHT_INPAINT.md`.

## 9.6.3

- Documented official Qwen-Image-Edit-2511 download pages in README/README_zh.
- Added manual browser download, Hugging Face CLI, `hf` CLI, and Git LFS examples.
- Documented the recommended local model directory `./models/Qwen-Image-Edit-2511`.
- Clarified that the complete Hugging Face repository structure, including `model_index.json` and model weights/components, must be preserved.

## [Unreleased]

## [9.6.2] — 2026-08-14

### Qwen download diagnostics / recovery
- Fixed the GUI case where a failed/crashed Qwen download only displayed `code=-1` with no visible reason.
- QProcess start/crash errors now capture Qt's own `errorString()`, and model-download stderr is buffered for diagnosis.
- Download failures now show an inline selectable error card, actionable suggestions, a copy-details button, a detailed modal, and automatically expand the log panel.
- Added explicit download-source selection: official Hugging Face, optional `hf-mirror.com` third-party mirror, or the user's `HF_ENDPOINT`; ScenePaste never silently switches to a third-party mirror.
- Added download timeout/worker controls in the CLI path and structured JSON error output for dependency, auth, repository, network/proxy/SSL/timeout, disk-space and filesystem-permission failures.
- Incomplete/partial model directories are no longer treated as valid local models; `model_index.json` is required and interrupted downloads are reported as resumable/incomplete.

## [9.6.1] — 2026-08-14

### Open-source provenance and acknowledgements
- Added bilingual README acknowledgements covering NVIDIA Physical AI Data Factory, Qwen-Image/Edit, SAM 2, Grounded SAM 2, LabelMe, X-AnyLabeling, rembg and WebDataset.
- Added `THIRD_PARTY_NOTICES.md` separating runtime dependencies, optional model integrations, interoperability targets and design references.
- Explicitly documented that GPL-licensed LabelMe/X-AnyLabeling are interoperability/design references and are not copied or bundled into ScenePaste.
- Refreshed `docs/INSPIRATION.md` around the current V9 data-factory direction while retaining the earlier Copy-Paste engineering references.
- Added model-asset licensing guidance for separately downloaded Qwen/SAM/rembg assets and a no-endorsement clarification.

## [9.6.0] — 2026-08-14

### Open-source UX / desktop polish
- Reframed the desktop app as **ScenePaste — AI Vision Data Factory** and promoted the AI Data Factory to a first-class toolbar/onboarding entry.
- Rebuilt the large-generation dialog around a beginner-friendly workflow: Data & Task → AI Data Factory → Advanced Settings.
- Added high-level generation presets for Copy-Paste, local Qwen fusion, Qwen service fusion, Qwen generative objects, OpenCV harmonization and advanced custom backends.
- Model/service controls are now context-aware instead of permanently exposing implementation details.
- Added model readiness cards, local model browsing, download state, disk-space warning, GPU/VRAM/Pipeline diagnostics and clearer dependency errors.
- Reorganized quality controls around label consistency, retry/fallback, mask refinement, scene regions and hard-case generation.
- Replaced the log-first right pane with a task dashboard showing progress, completed images, object count, failures, throughput, ETA/disk state and recent before/mask/after comparison preview.
- Detailed logs and advanced Qwen tuning are collapsed by default; they remain accessible for debugging and research.
- Refined dark/light QSS with consistent cards, spacing, typography, status pills, scrollbars, hover states and preview surfaces.
- Preserved the CLI/config contract and existing core behavior; UI presets only map onto the existing `generation_mode` / `generator_backend` options.

## [9.5.0] — 2026-08-14

### Final Qwen native integration
- Replaced the provisional Qwen local path with native official Diffusers Qwen edit pipeline loading. ScenePaste detects `QwenImageEditPipeline` and `QwenImageEditPlusPipeline` from the local checkpoint.
- Added `scenepaste qwen download`, defaulting to `Qwen/Qwen-Image-Edit-2511`, for explicit local snapshot download.
- Added GPU/VRAM, dependency, pipeline-class and local-model diagnostics to `scenepaste qwen doctor`.
- Added Qwen controls for inference steps, true CFG, guidance scale, ROI crop padding, CPU offload and VAE tiling.
- Qwen edits now run on a padded local ROI and are composited back strictly inside the label-first mask, preserving all pixels outside the requested label region.
- Added optional runtime fallback (`opencv-harmonize` / `copy-paste`) and per-object provenance when a generative edit fails.
- Added optional before/composite/mask/after debug artifacts plus a four-panel comparison image for every generated object.
- Expanded the Qt data-factory page with official-model download, local model chooser, dependency/VRAM check, Qwen service controls, native parameters and quick access to the latest comparison image.
- Native local Qwen/Diffusers generation is forced to one worker by default to avoid loading a 20B-class model into multiple worker processes.

### Compatibility
- Traditional Copy-Paste remains the default and requires no generative dependencies.
- Qwen model weights are never bundled in the ScenePaste release archive; download is explicit and models remain in the user's chosen local directory.

## [9.0.0] — 2026-08-14

### VisionDataForge mainline
- Promoted ScenePaste from a Copy-Paste-first augmenter to a **Label-First controllable data factory** while keeping the mature ScenePaste data/QA infrastructure and legacy generation path.
- Added explicit scene plans with class, location, scale, occlusion/difficulty hints and reusable built-in hard-case recipes (`small-object`, `far-occluded`, `crowded`).
- Added scene-understanding placement regions with LabelMe class zones, forbidden zones and an automatic ground-prior fallback.
- Added pluggable generation backends: deterministic Copy-Paste, OpenCV harmonization baseline, Diffusers inpainting, generic HTTP services and Python plugins.
- Added `generative-paste` and `generative-object` generation modes with prompt construction, deterministic retry seeds and hard preservation outside the planned edit mask.
- Added label-consistency QA based on requested masks, observed image changes, leakage, IoU and area plausibility; warn/reject policies can retry bad generations.
- Added generated-mask refinement and V9 per-sample metadata containing the scene plan, generator trace, QA metrics and scene-region provenance.
- Added `scenepaste factory doctor` and `scenepaste factory plan` for backend diagnostics and reproducible Label-First planning.
- Added a V9 Data Factory tab to the large-generation Qt dialog.
- Moved LabelMe cutout saving into the core package so headless workflows no longer require PySide6.
- Added `docs/V9_DATA_FACTORY.md` plus refreshed architecture and roadmap documentation.

### Compatibility
- Existing Copy-Paste generation, Detect/Seg/OBB/Semantic/COCO export, Distribution Profiles, Scene Templates, QA, hard mining, curation and project workflows remain available.
- Heavy generative dependencies are optional through `pip install .[generative]`; the default install remains lightweight.


### Added
- Scene editor **目标外观预览** panel: per-instance enable/recipe/sliders/resample with live canvas preview (RGB-only; labels unchanged).
- Main-window **批量生成默认** panel (scene recipe / object appearance / blend / empty-scene) shared by 批量套用, 大规模生成, project save/load, and project settings.
- GUI **抠图工作室** (Cutout Studio): folder path + thumbnail browser, rembg / SAM2 click / GroundingSAM2 models with download progress, LabelMe save, batch auto-cutout.
- Docs assets: `docs/images/ui_cutout_studio.png`, `docs/images/gif_00_cutout.gif`; refreshed README / blog screenshots and workflow GIFs.

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
