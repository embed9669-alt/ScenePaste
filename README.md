# ScenePaste

[![CI](https://github.com/embed9669-alt/ScenePaste/actions/workflows/ci.yml/badge.svg)](https://github.com/embed9669-alt/ScenePaste/actions/workflows/ci.yml)

**Scene-first synthetic dataset generation for computer vision.**

ScenePaste is a local-first desktop studio and CLI for building **controllable synthetic datasets from real object cutouts and real backgrounds**. It turns Copy-Paste from a one-line random augmentation into a complete engineering workflow:

**assets → scene composition → reusable templates → batch synthesis → automatic annotations → visual inspection → dataset QA → model hard-mining → curation / sharding**

> **Status: v1.0.0 Stable.** ScenePaste has completed the release-consistency pass for packaging, bundled samples, deterministic generation, annotation formats and dataset tools. Synthetic data should still be visually reviewed and evaluated against a real held-out validation/test set.

![ScenePaste desktop editor](docs/images/ui_overview.png)

<details>
<summary>Light theme &amp; project settings</summary>

![Light theme](docs/images/ui_overview_light.png)

![Project settings](docs/images/ui_settings.png)

</details>

### What it produces

![Background and generated ScenePaste sample](docs/images/example_before_after.jpg)

Same scene, multiple annotation tasks from one render (`--output-format all`):

![Detect / Segmentation / OBB / Semantic](docs/images/example_formats.png)

| Detect | Segmentation | OBB | Semantic |
|---|---|---|---|
| axis-aligned boxes | instance polygons | oriented boxes | per-pixel class masks |

<details>
<summary>Dataset Explorer · Data Loop Center · QA</summary>

![Dataset Explorer with multi-format overlays](docs/images/ui_explorer.png)

![Data Loop Center](docs/images/ui_data_loop.png)

![QA summary card](docs/images/example_qa.png)

</details>

The source repository and installed wheel both ship a tiny public sample dataset, so the GUI can be tried without preparing your own assets first.

## Why ScenePaste?

ScenePaste is for cases where you need more than random training-time augmentation:

- deliberately create rare, safety-critical or long-tail scenes;
- compose a realistic layout once and reuse it as a **scene template** across many backgrounds;
- control position, scale, perspective, rotation, flip, overlap and z-order-aware occlusion;
- derive Detect / Segment / Semantic / OBB / COCO labels from the same rendered scene;
- generate deterministic offline datasets without a cloud service;
- visually inspect generated annotations in the built-in **Dataset Explorer**;
- measure class balance, source reuse and annotation health before training;
- perform source-aware train/val splitting and cross-split visual leakage checks;
- feed model prediction failures back into a hard-example generation profile;
- compare real vs synthetic domains and export curated diverse subsets;
- package multi-million-image datasets into WebDataset-compatible tar shards.

ScenePaste does **not** claim a new Copy-Paste research algorithm. Its goal is to make controllable synthetic-data generation practical, inspectable and reproducible for real computer-vision projects.

## Highlights

- **PySide6 scene editor** — drag, resize, rotate, flip and duplicate real cutouts, with z-order-aware occlusion.
- **Parameterized Scene Template v2** — human-designed nominal layouts plus position/scale/angle/probability ranges, same-class asset variation and relation constraints (`left_of`, `right_of`, `above`, `below`, distance bounds).
- **Dataset Explorer** — browse generated images with Detect / Seg / OBB / COCO / Semantic overlays.
- **Asset search and responsive thumbnail browsing** for cutout libraries.
- **LabelMe / X-AnyLabeling input** with polygon extraction.
- **Optional rembg auto-cutout** (`pip install -e ".[auto]"`).
- **Constraint-aware placement** — generic or class-specific `paste_zone:<class>` regions, perspective sizing and relation-constrained templates.
- **Mask-first annotation pipeline** — final visible masks respect z-order occlusion.
- **YOLO Detect / Segment / OBB**, **Semantic Segmentation**, and **COCO Instances**.
- **Balanced sampling** for classes and backgrounds, plus intentional pure-background negative scenes.
- **Distribution-driven generation** — learn class/count/position/scale plus crowding/overlap statistics from LabelMe, YOLO Detect/Seg/OBB or COCO, or mix several domain profiles with explicit weights.
- **Resumable multiprocessing** — deterministic per-index plans, bounded process queues and SQLite run state.
- **Deterministic runs**, run IDs, crash-safe per-sample metadata, background LRU cache and sampled previews.
- **Augmentation Recipes + blend modes** — reproducible camera/surveillance/low-light appearance pipelines with alpha, hard-edge or Gaussian-edge blending.
- **QA + curation Dashboard** — integrity, exact/pHash/embedding leakage, visual uniqueness, class/scale/position/crowding distributions, actual rendered visibility diagnostics, reuse and target-vs-generated drift.
- **Detect / Seg / OBB hard-example mining** — consume YOLO-style prediction TXT with confidence, rank FN/FP/low-confidence/geometry failures and emit a reusable hard distribution profile.
- **Real vs synthetic comparison** — compare class/count/geometry distributions plus lightweight visual-domain similarity.
- **Diversity curation** — select a maximally diverse subset and optionally export it as a complete labeled dataset.
- **WebDataset sharding** — deterministic tar shards with per-shard SHA-256 manifests for large-scale training.
- **Project Manifest + Data Loop Center** — reopen a complete workflow from `scenepaste.project.json`, then run profile learning, Detect/Seg/OBB hard mining, QA, comparison, curation and sharding from one GUI.
- **Optional CLIP / DINOv2 embeddings** — keep `cv-lite-v1` as the offline default or opt into foundation-model similarity.
- **Live large-run telemetry** — progress, img/s, ETA, failures and free disk under `.scenepaste/status/`.
- Dataset analyze, merge and leakage-aware split tools.
- **Dark/light themes** and multi-platform GitHub Actions CI.

## Install

**Requires Python 3.10 or newer** (3.10–3.13 are covered by CI).

### Recommended desktop setup

```bash
cd ScenePaste
python -m pip install -e ".[dev,gui-qt]"
```

If building against a flaky PyPI mirror, install with local build tools already present:

```bash
python -m pip install --no-build-isolation -e ".[dev,gui-qt]"
```

Optional automatic foreground extraction:

```bash
python -m pip install -e ".[auto]"
```

Common desktop features (Qt GUI + rembg):

```bash
python -m pip install -e ".[all,dev]"
```

Optional foundation-model embeddings:

```bash
python -m pip install -e ".[embeddings]"
```

Core generator only:

```bash
python -m pip install -e .
```

## One command, fourteen workflows

```text
scenepaste gui          # interactive scene editor
scenepaste generate     # batch synthetic-data generation
scenepaste explore      # visual dataset / annotation browser
scenepaste analyze      # quick dataset QA and statistics
scenepaste qa           # full JSON + HTML QA dashboard
scenepaste profile      # learn/show/mix real-data distribution profiles
scenepaste recipe       # inspect/export augmentation recipes
scenepaste split        # source-aware train/val split
scenepaste merge        # merge generated datasets
scenepaste curate       # hard mining / leakage / diversity selection
scenepaste compare      # real-vs-synthetic comparison dashboard
scenepaste shard        # WebDataset-compatible tar sharding
scenepaste project      # portable project manifest
scenepaste loop         # unified GUI data-loop center
```

```bash
scenepaste --help
scenepaste --version
```

## 30-second GUI demo

The installed wheel contains a tiny sample dataset as package resources. Launch:

```bash
scenepaste gui
```

Then click **★ 加载示例**. This works after a normal wheel install and does not require a source checkout.

### Your own assets (when not using the sample)

Prepare two folders:

```text
objects/                 # cutouts
  person_001.jpg
  person_001.json        # LabelMe / X-AnyLabeling polygon (required)
  truck_002.jpg
  truck_002.json

backgrounds/
  street_01.jpg
  street_01.json         # optional: paste_zone / paste_zone:person / ground / road
  street_02.jpg
```

- Objects need a same-stem image + LabelMe JSON with the class polygon.
- Backgrounds can be plain images; add a same-name JSON only when you want placement zones.
- Class IDs are declared once via `--class-map` / project settings (unknown labels are auto-extended in the GUI loader).

When running from the repository, the same public samples are also available under `samples/`, so CLI examples can use paths such as:

```bash
scenepaste generate \
  --objects ./samples/objects \
  --backgrounds ./samples/backgrounds \
  --output ./generated \
  --count 20 \
  --class-map "person=0,truck=1,motorcycle=2"
```

Reusable source-tree layouts are included under `samples/templates/`, including parameterized and relation-constrained examples:

- `distant_person.json`
- `person_near_truck.json`
- `mixed_traffic_scene.json`
- `parameterized_mixed_traffic.json`
- `constrained_person_truck.json`

See [docs/SCENE_TEMPLATES.md](docs/SCENE_TEMPLATES.md) and [docs/PLACEMENT_CONSTRAINTS.md](docs/PLACEMENT_CONSTRAINTS.md).

## Project Manifest and Data Loop Center

Create one portable project file instead of rebuilding paths and defaults on every machine:

```bash
scenepaste project init . \
  --objects ./objects --backgrounds ./backgrounds --output ./generated \
  --class-map "person=0,truck=1"

scenepaste generate --project ./scenepaste.project.json --count 100000 --workers 8
```

The main editor can open/save the same project file. **🧠 Data Loop** (or `scenepaste loop --project ...`) brings Profile learning, Detect/Seg/OBB Hard Mining, QA/leakage, real-vs-synthetic comparison, diversity curation and WebDataset sharding into one window.

See [docs/PROJECTS.md](docs/PROJECTS.md) and [docs/DATA_LOOP_CENTER.md](docs/DATA_LOOP_CENTER.md).

## Batch generation

```bash
scenepaste generate \
  --objects ./samples/objects \
  --backgrounds ./samples/backgrounds \
  --output ./generated \
  --count 1000 \
  --min-objects 1 \
  --max-objects 3 \
  --class-map "person=0,truck=1,motorcycle=2" \
  --output-format detect \
  --asset-sampling balanced \
  --background-sampling balanced \
  --preview-ratio 0.01 \
  --background-cache-size 16
```

### Learn the real data distribution

```bash
scenepaste profile learn ./real_dataset --geometry-source auto -o distribution_profile.json
```

The compact profile learns class frequency, objects/image, normalized position/scale, width/area/aspect-ratio and **crowding/overlap IoU**. With polygon/mask geometry it also records visible-shape occupancy. Input can be **LabelMe, YOLO Detect/Seg/OBB or COCO**.

Mix several domains without allowing the largest source dataset to dominate by size:

```bash
scenepaste profile mix factory.json tunnel.json -w 0.7 0.3 -o mixed_profile.json
```

Use a learned or mixed profile during generation:

```bash
scenepaste generate \
  --objects ./objects \
  --backgrounds ./backgrounds \
  --output ./generated \
  --class-map "person=0,truck=1" \
  --count 100000 \
  --distribution-profile distribution_profile.json \
  --profile-strength 1.0 \
  --workers 0 \
  --preview-ratio 0.01
```

### Appearance recipes, blending and negative scenes

Use a built-in image-only recipe after scene composition. Geometry and labels remain aligned because these transforms do not move pixels spatially:

```bash
scenepaste recipe list
scenepaste recipe show surveillance

scenepaste generate \
  --objects ./objects \
  --backgrounds ./backgrounds \
  --output ./generated \
  --class-map "person=0,truck=1" \
  --count 20000 \
  --output-format all \
  --augmentation-recipe surveillance \
  --blend-mode gaussian \
  --blend-sigma 1.5 \
  --empty-scene-prob 0.10
```

Built-ins are `clean`, `camera-mild`, `surveillance`, and `low-light`. Export one to JSON and tune probabilities/ranges for your camera:

```bash
scenepaste recipe export camera-mild -o my_camera_recipe.json
```

See [docs/AUGMENTATION_RECIPES.md](docs/AUGMENTATION_RECIPES.md).

### Resumable multiprocessing

```text
--workers 0                  auto = about CPU count - 1
--workers 8                  explicit process count
--queue-depth 0              bounded in-flight queue (default workers*2)
--run-id my_experiment       stable run identifier
--resume                     resume only unfinished sample indices
--preview-ratio 0.01         save only a QA sample of previews
--background-cache-size 16   per-worker decoded-background LRU
--seed 42                    deterministic per-index planning
```

Run state is stored under `.scenepaste/runs/<run_id>.sqlite3`. If interrupted, rerun the same command with `--resume`; ScenePaste validates a configuration hash before continuing.

### Parameterized Scene Template

A v2 template can define X/Y ranges, scale ranges, angle ranges, instance probability, flip probability, same-class random asset replacement, intentional overlap and cross-slot relations such as `right_of`, `above` or distance bounds. Save one from the GUI or use the bundled `samples/templates/parameterized_mixed_traffic.json`:

```bash
scenepaste generate ... \
  --scene-template samples/templates/parameterized_mixed_traffic.json \
  --count 10000 --workers 8
```

## Output formats

| Mode | Main output | Intended use |
|---|---|---|
| `detect` | `labels/train/*.txt` | Ultralytics YOLO Detect |
| `seg` | `labels-seg/train/*.txt` | Ultralytics YOLO Segment |
| `both` | Detect in `labels/train`, Seg in `labels-seg/train` | dual export |
| `obb` | `labels-obb/train/*.txt` | Ultralytics YOLO OBB (`class x1 y1 ... x4 y4`) |
| `semantic` | `masks/train/*.png` + `semantic_classes.json` | semantic segmentation |
| `coco` | `instances_coco.json` + Detect labels | COCO instance segmentation |
| `all` | Detect + Seg + OBB + Semantic + COCO | one-pass multi-task export |

Semantic masks reserve value `0` for background. Dataset class `0` is stored as mask value `1`, class `1` as `2`, and so on. See [docs/FORMATS.md](docs/FORMATS.md).

Regenerate the multi-format collage and Explorer/QA shots used above with:

```bash
python scripts/capture_docs_screenshots.py
```

## Dataset Explorer

After generation:

```bash
scenepaste explore ./generated
```

The explorer automatically finds images under `images/train|val|test` and overlays the annotation modalities available for each image:

- YOLO Detect boxes;
- YOLO segmentation polygons;
- Ultralytics OBB four-corner polygons;
- COCO instance polygons / boxes;
- semantic segmentation masks.

You can also open it from the main editor using **🔎 数据集浏览**.

## Placement zones and perspective

A background may have a same-name LabelMe JSON:

```text
bg_001.jpg
bg_001.json
```

Draw a polygon named one of:

```text
paste_zone
ground
road
地面
可粘贴区域
```

Object feet are sampled inside that region. You can also define **class-specific zones** such as `paste_zone:person`, `paste_zone:truck`, `zone:forklift`, or `ground:person`; a matching class-specific zone takes precedence over the generic zone. Without a zone, generation uses the configured vertical ground range. Perspective-aware sizing makes nearer/lower targets larger and farther/upper targets smaller.

For scene-level relationships, Scene Template v2 supports `left_of`, `right_of`, `above`, `below`, `min_distance` and `max_distance` constraints with bounded retry sampling.

## Automatic cutout

```bash
scenepaste generate \
  --objects ./raw_object_images \
  --backgrounds ./backgrounds \
  --output ./generated \
  --class-map "auto=0" \
  --auto-cutout
```

`rembg` is optional and its model session is reused within the process. SAM is currently an extension point, not a completed backend. See [docs/AUTO_MASK.md](docs/AUTO_MASK.md).

## Dataset QA and Dashboard

Quick terminal QA:

```bash
scenepaste analyze ./generated
scenepaste analyze ./generated --json
```

Full dashboard:

```bash
scenepaste qa ./generated
```

This writes `qa_report.json` and a standalone `qa_dashboard.html` covering annotation integrity, unreadable images, exact duplicates, **perceptual near-duplicates (pHash)**, lightweight visual-embedding uniqueness, **cross train/val/test leakage candidates**, class distribution, scale/position histograms, source/background reuse and—when a target distribution profile was used—target-vs-generated drift. The Dataset Explorer toolbar also exposes **📊 QA Dashboard**.

## Close the loop with model hard examples

Run your detector on a real held-out split with Ultralytics `save_txt=True, save_conf=True`, then point ScenePaste at the prediction label directory:

```bash
scenepaste curate hardmine ./real_yolo_dataset \
  --predictions ./runs/detect/predict/labels \
  --task detect --split val \
  --top 300 \
  -o ./hardmine
```

ScenePaste class-aware matches predictions to ground truth and scores **false negatives, false positives, low-confidence true positives and poor localization**. Outputs include:

```text
hardmine_dashboard.html
hard_examples.json / .csv / .txt
hard_negative_backgrounds.txt
hard_distribution_profile.json
```

Feed the last file directly into the next synthesis run:

```bash
scenepaste generate ... \
  --distribution-profile ./hardmine/hard_distribution_profile.json \
  --profile-strength 1.0
```

See [docs/HARD_EXAMPLE_MINING.md](docs/HARD_EXAMPLE_MINING.md).

## Dataset curation, leakage and real/synthetic comparison

Cross-split leakage candidates:

```bash
scenepaste curate leakage ./dataset
```

Maximally diverse subset selection using the built-in download-free `cv-lite-v1` visual descriptor:

```bash
scenepaste curate diversity ./dataset \
  --select 1000 \
  --export-dataset ./dataset_diverse_1000
```

Compare a real dataset with generated data:

```bash
scenepaste compare ./real_dataset ./generated -o ./comparison
```

The comparison dashboard combines class/count/geometry distribution distances with real-vs-synthetic appearance similarity. The built-in descriptor is deliberately lightweight and local; it is a curation signal, not a replacement for task-model evaluation.

See [docs/DATA_CURATION.md](docs/DATA_CURATION.md) and [docs/EMBEDDINGS.md](docs/EMBEDDINGS.md).

## Multi-million scale / WebDataset sharding

After generation, package a split into deterministic sequential tar shards:

```bash
scenepaste shard ./generated \
  --split train \
  --max-samples 10000 \
  -o ./shards
```

Each shard keeps image + available Detect/Seg/OBB/Semantic/COCO sample payloads under one basename. The manifest records sample counts, bytes and SHA-256 for every tar. See [docs/SHARDING.md](docs/SHARDING.md).

During long generation runs ScenePaste also writes `.scenepaste/status/<run_id>.json` with completed/failed counts, throughput, ETA and free disk space; the Qt large-generation dialog polls this status live.

## Leakage-aware train/val split

```bash
scenepaste split --input ./generated --val-ratio 0.2 --run-id latest
```

Samples are grouped by source object so the same cutout is not blindly randomized across both splits. For the strictest evaluation, keep real validation/test data independent from synthetic training data.

## Merge datasets

```bash
scenepaste merge run_a run_b run_c --output merged
```

The merger supports images, YOLO labels, extra segmentation labels, semantic masks, previews, run logs and COCO annotations with collision-safe prefixes.

## Python API

```python
from pathlib import Path
from scenepaste import GenerationConfig, generate_dataset

cfg = GenerationConfig(
    objects_dir=Path("samples/objects"),
    backgrounds_dir=Path("samples/backgrounds"),
    output_dir=Path("generated"),
    class_map={"person": 0, "truck": 1, "motorcycle": 2},
    count=100,
    output_format="detect",
)
summary = generate_dataset(cfg)
print(summary)
```

## Project layout

```text
ScenePaste/
├── scenepaste/
│   ├── core/                    # generation, sampling, geometry, validation
│   ├── formats/                 # YOLO / COCO / semantic writers
│   ├── tools/                   # QA / curation / compare / shard / split / merge
│   ├── explorer.py              # headless dataset indexing/overlay renderer
│   └── cli.py                   # unified `scenepaste` command
├── compose_app/                 # shared GUI models/rendering helpers
├── compose_app_qt/              # PySide6 editor + Dataset Explorer window
├── samples/
│   ├── objects/
│   ├── backgrounds/
│   └── templates/
├── docs/
├── tests/
├── scripts/
└── pyproject.toml
```

Top-level `analyze_datasets.py`, `split_dataset.py` and `merge_datasets.py` are thin source-tree compatibility wrappers; installed users should use `scenepaste ...`.

## Verification

```bash
pytest -q
```

For the v1.0.0 release, the headless suite verifies the core generator, project manifests, Detect/Seg/OBB hard mining, overlap-aware profiles, generation visibility diagnostics, leakage checks, diversity selection, real-vs-synthetic comparison, WebDataset sharding, live run telemetry and the established annotation/template/data-tool paths. Qt-dependent GUI modules are additionally exercised by CI when `PySide6` is installed. The exact local test count is reported by `pytest`; Qt-dependent modules run in the cross-platform CI matrix when PySide6 is installed. CI also installs the built wheel outside the repository and runs an end-to-end sample smoke test.

Build a clean release archive:

```bash
python scripts/build_release.py
```

The release builder verifies that `.git`, Python caches, test caches, coverage artifacts, egg-info, generated datasets and user asset folders are not included.

The CI release-smoke job builds a wheel, installs it **non-editably** in a clean runner, changes to a directory outside the repository, and verifies bundled samples → offscreen GUI sample load → template save → small `all` generation → Explorer render → QA → WebDataset sharding.

A reproducible 1/4/8-worker benchmark helper and measured reference table are documented in [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Current limitations

ScenePaste intentionally documents its current limits:

- multiprocessing duplicates the decoded cutout library per worker, so very large object libraries require conservative worker counts;
- WebDataset sharding is a post-generation packaging step; direct-to-shard generation is not yet implemented;
- `cv-lite-v1` is a lightweight appearance descriptor, not a semantic foundation-model embedding;
- Hard Mining supports YOLO-style Detect / Seg / OBB TXT; task-specific exporters may require adaptation before use;
- SAM is not yet a completed backend;
- Copy-Paste cannot create source-object diversity that does not exist in the source library;
- a learned 2D-image distribution improves statistical realism but does not guarantee physical 3D plausibility;
- real held-out validation/test sets are strongly recommended.

See [docs/ROADMAP.md](docs/ROADMAP.md).

## Documentation

- [中文使用说明](README_zh.md)
- [Chinese user guide](docs/USER_GUIDE_zh.md)
- [Project Manifest](docs/PROJECTS.md)
- [Data Loop Center](docs/DATA_LOOP_CENTER.md)
- [Embedding backends](docs/EMBEDDINGS.md)
- [Scene templates](docs/SCENE_TEMPLATES.md)
- [Placement constraints](docs/PLACEMENT_CONSTRAINTS.md)
- [Augmentation recipes](docs/AUGMENTATION_RECIPES.md)
- [Real-data distribution profiles](docs/DISTRIBUTION_PROFILES.md)
- [Resumable large runs](docs/LARGE_RUNS.md)
- [QA Dashboard](docs/QA_DASHBOARD.md)
- [Hard-example mining](docs/HARD_EXAMPLE_MINING.md)
- [Dataset curation and real/synthetic comparison](docs/DATA_CURATION.md)
- [WebDataset sharding](docs/SHARDING.md)
- [Output formats](docs/FORMATS.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Performance](docs/PERFORMANCE.md)
- [Automatic cutout](docs/AUTO_MASK.md)
- [Segmentation notes](docs/SEGMENTATION.md)
- [Design inspiration / related projects](docs/INSPIRATION.md)
- [Roadmap](docs/ROADMAP.md)
- [Release checklist](docs/RELEASE_CHECKLIST.md)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)

## License

MIT. See [LICENSE](LICENSE).
