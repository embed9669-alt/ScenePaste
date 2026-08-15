# Third-party notices

> Note: ScenePaste Copy-Paste only builds no longer depend on Diffusers / Qwen image-edit runtimes. Notices below may still document optional cutout/embedding stacks and historical references.

Party Notices

This document clarifies third-party relationships for ScenePaste.

ScenePaste itself is released under the MIT License (see `LICENSE`). Third-party packages, model checkpoints, datasets, formats, and external projects remain subject to their own licenses and terms. This file is informational and is not legal advice.

## 1. Relationship categories

ScenePaste uses four distinct kinds of third-party relationships:

1. **Runtime dependencies** — Python packages installed by `pip` and imported by ScenePaste.
2. **Optional integrations / model backends** — packages or model assets that are only used when a feature is explicitly enabled.
3. **Interoperability targets** — file formats or workflows ScenePaste can read/write without bundling the upstream project's source code.
4. **Design references / inspiration** — external projects whose public ideas or workflows influenced ScenePaste. Acknowledgement does **not** mean ScenePaste is a fork or derivative distribution of those repositories.

## 2. Core and optional runtime dependencies

The authoritative dependency list is `pyproject.toml`. Important examples include:

| Project / package | Relationship | Upstream |
|---|---|---|
| NumPy | Core runtime dependency | https://github.com/numpy/numpy |
| OpenCV / `opencv-python` | Core runtime dependency | https://github.com/opencv/opencv |
| Pillow | Core runtime dependency | https://github.com/python-pillow/Pillow |
| PySide6 / Qt for Python | Optional desktop GUI dependency | https://doc.qt.io/qtforpython-6/ |
| rembg | Optional automatic-cutout backend | https://github.com/danielgatis/rembg |
| ONNX Runtime | Optional rembg/runtime dependency | https://github.com/microsoft/onnxruntime |
| PyTorch | Optional grounding / embedding dependency | https://github.com/pytorch/pytorch |
| Hugging Face Transformers | Optional grounding / embedding dependency | https://github.com/huggingface/transformers |
| Hugging Face Hub | Optional model download/discovery dependency | https://github.com/huggingface/huggingface_hub |
| Accelerate / safetensors / timm | Optional model-runtime dependencies | See the corresponding upstream repositories |

These packages are **not relicensed by ScenePaste**. Users and redistributors should comply with the license shipped by each installed package.

## 3. Model integrations and model assets

### Stable Diffusion v1.5 Inpainting

- Upstream model: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting
- Relationship: default optional lightweight local inpainting checkpoint used by the `light-inpaint` preset.
- The upstream model card lists the **CreativeML OpenRAIL-M** license. Model weights are downloaded separately and are not bundled in ScenePaste release archives.
- ScenePaste downloads only the FP16 Diffusers components needed for local inference; this selection does not change or relicense the upstream model.
- Users should review the exact upstream model card/license before redistribution or deployment.

### Qwen-Image / Qwen-Image-Edit

- Upstream repository: https://github.com/QwenLM/Qwen-Image
- Relationship: optional model integration and major design reference for generative image editing.
- Upstream repository code is published under **Apache-2.0**.
- ScenePaste does not bundle Qwen model weights in its release archive. The `scenepaste qwen download` workflow downloads model assets separately to a user-selected directory.
- **Model checkpoints may have model-card-specific terms. Always review the license/terms shown on the exact model repository before downloading, redistributing, or deploying a checkpoint.**

### Meta SAM 2

- Upstream repository: https://github.com/facebookresearch/sam2
- Relationship: segmentation/model-workflow reference; optional grounding/segmentation ecosystem integration.
- The upstream repository states that SAM 2 model checkpoints and core demo/training code are licensed under **Apache-2.0**; some demo font assets use separate font licenses.
- ScenePaste does not relicense SAM 2 checkpoints.

### Grounded SAM 2

- Upstream repository: https://github.com/IDEA-Research/Grounded-SAM-2
- Relationship: design/workflow reference for open-vocabulary grounding + segmentation and automatic labeling.
- The repository contains multiple upstream components and license files (including Apache-2.0 and BSD-3-Clause notices). Consult the upstream repository for component-specific licensing.

## 4. Annotation interoperability and UX references

### LabelMe

- Upstream repository: https://github.com/wkentaro/labelme
- Upstream license: **GPL-3.0**.
- Relationship: annotation-format interoperability and UX/design reference.
- ScenePaste's MIT-licensed source tree does not copy, vendor, or bundle LabelMe source code. Supporting compatible JSON concepts/formats does not imply bundling LabelMe itself.

### X-AnyLabeling

- Upstream repository: https://github.com/CVHub520/X-AnyLabeling
- Upstream license: **GPL-3.0**.
- Relationship: AI-assisted annotation UX and workflow reference.
- ScenePaste's MIT-licensed source tree does not copy, vendor, or bundle X-AnyLabeling source code.

## 5. Data-factory and dataset-tooling references

### NVIDIA Physical AI Data Factory

- Upstream repository: https://github.com/NVIDIA/physical-ai-data-factory
- Relationship: design reference for synthetic-data generation, augmentation, labeling, quality control, and Physical AI data-factory workflows.
- The upstream repository states that documentation/skills are licensed under **CC-BY-4.0** and source code under **Apache-2.0**.
- ScenePaste is not a fork of NVIDIA Physical AI Data Factory and does not bundle its repository contents.

### WebDataset

- Upstream repository: https://github.com/webdataset/webdataset
- Upstream license: **BSD-3-Clause**.
- Relationship: design reference for large-scale tar-sharded sequential dataset access.
- ScenePaste implements its own post-generation sharding workflow; WebDataset is not a required core runtime dependency.

### rembg

- Upstream repository: https://github.com/danielgatis/rembg
- Upstream source-code license: **MIT**.
- Relationship: optional runtime dependency for automatic foreground extraction.
- Model assets downloaded/used by rembg may have provenance or license terms distinct from the rembg source code; users should review the relevant model source when redistribution matters.

## 6. Formats and ecosystem compatibility

ScenePaste reads/writes or interoperates with commonly used formats and ecosystems such as:

- LabelMe-style JSON annotations
- YOLO Detect / Segment / OBB text formats
- COCO Instances JSON
- semantic-segmentation masks
- Hugging Face / Diffusers model directories and checkpoints
- Qt/PySide6 desktop workflows

Format compatibility does not transfer ownership, trademarks, or licensing from the upstream project to ScenePaste.

## 7. No endorsement

Names such as NVIDIA, Qwen, Meta, LabelMe, X-AnyLabeling, Hugging Face, YOLO, COCO, OpenCV, Qt, and others are used only to identify compatible software, formats, APIs, or design references. ScenePaste is not endorsed by or affiliated with those organizations unless explicitly stated otherwise.

## 8. Keeping notices current

Licenses and model terms can change. Before a public release, maintainers should review:

- `pyproject.toml`
- this `THIRD_PARTY_NOTICES.md`
- model download defaults
- any newly vendored assets
- upstream license/model-card changes

If an upstream project is only a design reference, keep it in the **Acknowledgements / Inspiration** section rather than representing it as a runtime dependency.
