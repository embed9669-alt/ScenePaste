# Visual embedding backends

ScenePaste uses visual embeddings for diversity selection, real-vs-synthetic comparison, and cross-split similarity leakage checks.

## `cv-lite-v1` (default)

`cv-lite-v1` is deterministic, local, CPU-friendly, and requires no model download. It combines spatial Lab statistics, low-frequency DCT content, and gradient orientation. It is designed for dataset curation rather than semantic understanding.

```bash
scenepaste curate diversity ./dataset --embedding-backend cv-lite-v1
```

## Optional CLIP / DINOv2

Install the optional foundation-model dependencies:

```bash
python -m pip install -e ".[embeddings]"
```

Then select a backend:

```bash
scenepaste curate diversity ./dataset --embedding-backend clip
scenepaste curate leakage ./dataset --embedding-backend dinov2
scenepaste compare ./real ./synthetic --embedding-backend dinov2
scenepaste qa ./synthetic --embedding-backend clip
```

The current optional model identifiers are `openai/clip-vit-base-patch32` and `facebook/dinov2-small`. The model/processor is cached in-process after first load. ScenePaste batches foundation-model inference and automatically uses CUDA when PyTorch reports an available GPU. The Transformers backend may download weights on first use; use `cv-lite-v1` when fully offline behavior is required. Missing optional dependencies or unavailable model weights are reported as explicit errors rather than being silently converted to an empty result.

Embedding similarity is a QA/curation signal, not proof that two images are equivalent. Always inspect representative flagged pairs.
