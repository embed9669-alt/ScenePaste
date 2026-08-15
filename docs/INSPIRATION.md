# Design references and inspiration

ScenePaste is an independent, MIT-licensed project. It is not a fork of the projects below. This document records **design and workflow influences**, while `THIRD_PARTY_NOTICES.md` records runtime dependencies, optional integrations, interoperability boundaries, and third-party licensing notes.

## Current product direction

- **Human-reviewed Asset Studio + controllable Copy-Paste** — production default for label-stable synthetic data
- **NVIDIA Physical AI Data Factory** — synthetic-data generation, augmentation, labeling, quality control, and data-centric Physical AI workflow design: https://github.com/NVIDIA/physical-ai-data-factory
- **Meta SAM 2** — promptable segmentation and mask-assisted annotation: https://github.com/facebookresearch/sam2
- **Grounded SAM 2** — open-vocabulary grounding + segmentation and automatic object extraction: https://github.com/IDEA-Research/Grounded-SAM-2
- **LabelMe** — annotation JSON interoperability and polygon-annotation conventions: https://github.com/wkentaro/labelme
- **X-AnyLabeling** — AI-assisted annotation desktop UX reference: https://github.com/CVHub520/X-AnyLabeling
- **rembg** — optional foreground extraction used by ScenePaste automatic cutout: https://github.com/danielgatis/rembg
- **WebDataset** — tar-sharded, sequential large-dataset access patterns: https://github.com/webdataset/webdataset

> LabelMe and X-AnyLabeling are GPL-licensed upstream projects. They are listed here as interoperability/design references. ScenePaste does not copy or bundle their source code.

## Earlier ScenePaste engineering references

The original Copy-Paste / scene-composition line also drew useful engineering ideas from:

- **Albumentations / CopyAndPaste** — annotation-aware Copy-Paste, visibility filtering, hard/gaussian blending, and separation between spatial and image-only transforms: https://albumentations.ai/explore/transform/CopyAndPaste/docs/
- **BlenderProc** — constrained object pose sampling, collision-aware placement, and surface-oriented sampling: https://dlr-rm.github.io/BlenderProc/examples/advanced/object_pose_sampling/README.html
- **NVIDIA Omniverse Replicator** — domain randomization and constrained/scattered placement: https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/randomizer_details.html
- **FiftyOne Brain** — exact/near-duplicate discovery, embedding-based curation, and dataset QA workflows: https://docs.voxel51.com/brain/index.html
- **Ultralytics YOLO** — prediction TXT conventions used as one practical input for hard-example mining: https://docs.ultralytics.com/modes/predict/

ScenePaste translates selected ideas into a local-first workflow around real backgrounds, label-first planning, controlled synthetic generation, automatic labels, QA, hard-case mining, and model-driven data iteration.
