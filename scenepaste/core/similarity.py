"""Lightweight visual similarity, diversity, and cross-split leakage helpers.

The built-in ``cv-lite-v1`` descriptor intentionally has no model download or
GPU dependency.  It combines spatial Lab color statistics, low-frequency DCT
content, and gradient orientation.  It is not a semantic foundation-model
embedding, but it is deterministic and useful for duplicate/leakage checks,
dataset diversity ranking, and real-vs-synthetic appearance comparison.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .models import IMAGE_SUFFIXES

EMBEDDING_BACKEND = "cv-lite-v1"
_MODEL_CACHE = {}


def available_embedding_backends() -> list[str]:
    """Return embedding backends that can be requested.

    ``clip`` and ``dinov2`` are optional Transformer backends. They are loaded
    lazily and may download model weights on first use. ``cv-lite-v1`` remains
    the dependency-free default.
    """
    return ["cv-lite-v1", "clip", "dinov2"]


def _load_model_backend(backend: str):
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
    except Exception as exc:
        raise RuntimeError(
            f"Embedding backend '{backend}' requires optional dependencies. "
            "Install with: python -m pip install 'scenepaste[embeddings]'"
        ) from exc

    if backend == "clip":
        model_id = "openai/clip-vit-base-patch32"
        factory = lambda: (CLIPProcessor.from_pretrained(model_id), CLIPModel.from_pretrained(model_id).eval())
    elif backend == "dinov2":
        model_id = "facebook/dinov2-small"
        factory = lambda: (AutoImageProcessor.from_pretrained(model_id), AutoModel.from_pretrained(model_id).eval())
    else:
        raise ValueError(f"unknown embedding backend: {backend}")

    cache_key = (backend, model_id)
    if cache_key not in _MODEL_CACHE:
        processor, model = factory()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        _MODEL_CACHE[cache_key] = (processor, model, device)
    return _MODEL_CACHE[cache_key]


def _model_embeddings(images: Sequence[np.ndarray], backend: str) -> np.ndarray:
    """Compute a batch of optional foundation-model embeddings.

    The model is loaded once, cached, and moved to CUDA automatically when
    available. This keeps CLIP/DINOv2 optional while avoiding one forward pass
    per image during dataset curation.
    """
    if not images:
        return np.empty((0, 0), dtype=np.float32)
    try:
        import torch
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(
            f"Embedding backend '{backend}' requires optional dependencies. "
            "Install with: python -m pip install 'scenepaste[embeddings]'"
        ) from exc
    processor, model, device = _load_model_backend(backend)
    pil_images = [Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) for image in images]
    inputs = processor(images=pil_images, return_tensors="pt")
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.no_grad():
        if backend == "clip":
            vec = model.get_image_features(**inputs)
        else:
            out = model(**inputs)
            vec = out.last_hidden_state[:, 0]
    arr = vec.detach().cpu().numpy().astype(np.float32, copy=False)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
    return arr


def _model_embedding(image: np.ndarray, backend: str) -> np.ndarray:
    return _model_embeddings([image], backend)[0]

def embedding_for_image(image: np.ndarray, backend: str = EMBEDDING_BACKEND) -> np.ndarray:
    if backend in {"cv-lite", "cv-lite-v1", "default"}:
        return visual_embedding(image)
    return _model_embedding(image, backend)


def iter_dataset_images(root: Path, splits: Sequence[str] = ("train", "val", "test")) -> List[Tuple[str, Path]]:
    """Return ``(split, path)`` pairs from a YOLO/ScenePaste dataset or image folder."""
    root = Path(root)
    rows: List[Tuple[str, Path]] = []
    image_root = root / "images"
    if image_root.exists():
        for split in splits:
            d = image_root / split
            if d.exists():
                rows.extend((split, p) for p in sorted(d.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if rows:
            return rows
    # LabelMe / plain-image folder fallback.  Keep a synthetic "all" split.
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            rows.append(("all", p))
    return rows


def read_image(path: Path) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def phash64(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8].reshape(-1)
    median = float(np.median(dct[1:])) if len(dct) > 1 else float(dct[0])
    bits = dct > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return int(value)


def hamming64(a: int, b: int) -> int:
    return int(int(a) ^ int(b)).bit_count() if hasattr(int, "bit_count") else bin(int(a) ^ int(b)).count("1")


def visual_embedding(image: np.ndarray) -> np.ndarray:
    """Compute a deterministic, L2-normalized appearance descriptor."""
    if image is None or image.size == 0:
        raise ValueError("empty image")
    img = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    feats: List[float] = []
    # 4x4 spatial color moments: 16 * (mean RGB-like Lab + std) = 96 dims.
    for gy in range(4):
        for gx in range(4):
            cell = lab[gy * 16:(gy + 1) * 16, gx * 16:(gx + 1) * 16]
            feats.extend(cell.reshape(-1, 3).mean(axis=0).tolist())
            feats.extend(cell.reshape(-1, 3).std(axis=0).tolist())
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    dct = cv2.dct(gray)[:8, :8].reshape(-1)[1:]  # 63 low-frequency coefficients, exclude DC.
    dct = dct / (float(np.linalg.norm(dct)) + 1e-8)
    feats.extend(dct.tolist())
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=False)
    bins = np.floor((ang % (2 * np.pi)) / (2 * np.pi) * 8).astype(np.int32)
    hist = np.zeros(8, dtype=np.float32)
    for idx in range(8):
        hist[idx] = float(mag[bins == idx].sum())
    hist /= float(hist.sum()) + 1e-8
    feats.extend(hist.tolist())
    vec = np.asarray(feats, dtype=np.float32)
    vec /= float(np.linalg.norm(vec)) + 1e-8
    return vec


def embed_paths(
    paths: Sequence[Path],
    limit: int = 1000,
    backend: str = EMBEDDING_BACKEND,
    batch_size: int = 16,
) -> Tuple[List[Path], np.ndarray]:
    selected = list(paths[: max(0, int(limit))]) if limit > 0 else list(paths)
    good: List[Path] = []
    images: List[np.ndarray] = []
    for path in selected:
        image = read_image(path)
        if image is not None:
            good.append(path)
            images.append(image)
    if not images:
        return [], np.empty((0, 167), dtype=np.float32)

    if backend in {"cv-lite", "cv-lite-v1", "default"}:
        return good, np.stack([visual_embedding(image) for image in images], axis=0)

    # Optional model backends should fail loudly when dependencies/weights are
    # unavailable. Silently returning zero samples makes QA look successful
    # while no requested embedding was actually computed.
    size = max(1, int(batch_size))
    chunks: List[np.ndarray] = []
    for start in range(0, len(images), size):
        try:
            chunks.append(_model_embeddings(images[start:start + size], backend))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to compute '{backend}' embeddings for {good[start]}: {exc}"
            ) from exc
    return good, np.concatenate(chunks, axis=0)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.empty((len(a), len(b)), dtype=np.float32)
    # embeddings are normalized, but normalize defensively for external callers.
    aa = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    bb = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.clip(aa @ bb.T, -1.0, 1.0)


def _diversity_from_embeddings(emb: np.ndarray, backend: str) -> dict:
    n = len(emb)
    if n == 0:
        return {"backend": backend, "samples": 0, "mean_uniqueness": None}
    if n == 1:
        return {
            "backend": backend,
            "samples": 1,
            "mean_uniqueness": 1.0,
            "median_uniqueness": 1.0,
            "p10_uniqueness": 1.0,
            "p90_uniqueness": 1.0,
            "mean_nearest_similarity": 0.0,
        }
    sim = cosine_similarity_matrix(emb, emb)
    np.fill_diagonal(sim, -1.0)
    nearest = sim.max(axis=1)
    uniqueness = np.clip(1.0 - nearest, 0.0, 1.0)
    return {
        "backend": backend,
        "samples": n,
        "mean_uniqueness": float(np.mean(uniqueness)),
        "median_uniqueness": float(np.median(uniqueness)),
        "p10_uniqueness": float(np.percentile(uniqueness, 10)),
        "p90_uniqueness": float(np.percentile(uniqueness, 90)),
        "mean_nearest_similarity": float(np.mean(nearest)),
    }


def diversity_summary(paths: Sequence[Path], limit: int = 1000, backend: str = EMBEDDING_BACKEND) -> dict:
    _used, emb = embed_paths(paths, limit=limit, backend=backend)
    return _diversity_from_embeddings(emb, backend)


def select_diverse(paths: Sequence[Path], count: int, limit: int = 5000, backend: str = EMBEDDING_BACKEND) -> List[Tuple[Path, float]]:
    """Greedy farthest-point selection in the requested embedding space."""
    used, emb = embed_paths(list(paths), limit=limit, backend=backend)
    if not used or count <= 0:
        return []
    count = min(int(count), len(used))
    centroid = emb.mean(axis=0)
    centroid /= float(np.linalg.norm(centroid)) + 1e-8
    first = int(np.argmin(emb @ centroid))
    chosen = [first]
    min_dist = 1.0 - (emb @ emb[first])
    min_dist[first] = -1.0
    scores = [float(1.0 - emb[first] @ centroid)]
    while len(chosen) < count:
        idx = int(np.argmax(min_dist))
        scores.append(float(max(0.0, min_dist[idx])))
        chosen.append(idx)
        dist = 1.0 - (emb @ emb[idx])
        min_dist = np.minimum(min_dist, dist)
        min_dist[chosen] = -1.0
    return [(used[idx], score) for idx, score in zip(chosen, scores)]


def compare_embedding_domains(paths_a: Sequence[Path], paths_b: Sequence[Path], limit: int = 1000, backend: str = EMBEDDING_BACKEND) -> dict:
    a_paths, a = embed_paths(list(paths_a), limit=limit, backend=backend)
    b_paths, b = embed_paths(list(paths_b), limit=limit, backend=backend)
    if len(a_paths) == 0 or len(b_paths) == 0:
        return {"backend": backend, "real_samples": len(a_paths), "synthetic_samples": len(b_paths),
                "centroid_cosine_similarity": None, "synthetic_to_real_mean_nn_similarity": None}
    ca = a.mean(axis=0)
    ca /= float(np.linalg.norm(ca)) + 1e-8
    cb = b.mean(axis=0)
    cb /= float(np.linalg.norm(cb)) + 1e-8
    # Chunking avoids a giant full matrix on larger comparison limits.
    nn = []
    for start in range(0, len(b), 256):
        sims = cosine_similarity_matrix(b[start:start + 256], a)
        nn.extend(sims.max(axis=1).tolist())
    return {
        "backend": backend,
        "real_samples": len(a_paths),
        "synthetic_samples": len(b_paths),
        "centroid_cosine_similarity": float(np.clip(ca @ cb, -1.0, 1.0)),
        "synthetic_to_real_mean_nn_similarity": float(np.mean(nn)) if nn else None,
        "synthetic_to_real_p10_nn_similarity": float(np.percentile(nn, 10)) if nn else None,
        "synthetic_to_real_p90_nn_similarity": float(np.percentile(nn, 90)) if nn else None,
        # Reuse already computed embeddings; foundation-model backends should
        # not perform a second full inference pass just to calculate diversity.
        "real_diversity": _diversity_from_embeddings(a, backend),
        "synthetic_diversity": _diversity_from_embeddings(b, backend),
    }
