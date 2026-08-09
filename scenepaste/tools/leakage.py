"""Cross-split exact/near/visual-similarity leakage detection."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.similarity import (
    cosine_similarity_matrix,
    embed_paths,
    hamming64,
    iter_dataset_images,
    phash64,
    read_image,
    sha1_file,
)


def detect_split_leakage(
    root: Path,
    phash_threshold: int = 6,
    embedding_threshold: float = 0.995,
    embedding_limit_per_split: int = 1000,
    max_examples: int = 50,
    embedding_backend: str = "cv-lite-v1",
) -> dict:
    root = Path(root)
    rows = [(split, path) for split, path in iter_dataset_images(root) if split in {"train", "val", "test"}]
    by_split: Dict[str, List[Path]] = defaultdict(list)
    for split, path in rows:
        by_split[split].append(path)
    split_names = [s for s in ("train", "val", "test") if by_split.get(s)]

    exact_pairs = []
    exact_count = 0
    exact_pair_keys = set()
    digest_owner: Dict[str, Tuple[str, Path]] = {}
    for split in split_names:
        for p in by_split[split]:
            try:
                digest = sha1_file(p)
            except Exception:
                continue
            prev = digest_owner.get(digest)
            if prev and prev[0] != split:
                exact_count += 1
                exact_pair_keys.add(frozenset((str(prev[1]), str(p))))
                if len(exact_pairs) < max_examples:
                    exact_pairs.append({"a_split": prev[0], "a": str(prev[1]), "b_split": split, "b": str(p)})
            else:
                digest_owner[digest] = (split, p)

    # pHash pairs: bounded per split for predictable QA runtime.
    threshold = max(0, min(7, int(phash_threshold)))
    ph_rows: List[Tuple[str, Path, int]] = []
    for split in split_names:
        for p in by_split[split][: max(0, int(embedding_limit_per_split))]:
            img = read_image(p)
            if img is not None:
                ph_rows.append((split, p, phash64(img)))
    near_pairs = []
    near_count = 0
    buckets: List[dict] = [defaultdict(list) for _ in range(8)]
    processed: List[Tuple[str, Path, int]] = []
    for split, p, ph in ph_rows:
        candidates = set()
        for band in range(8):
            key = (ph >> (band * 8)) & 0xFF
            candidates.update(buckets[band].get(key, []))
        matched = None
        for idx in candidates:
            osplit, op, oph = processed[idx]
            if osplit == split:
                continue
            d = hamming64(ph, oph)
            if d <= threshold and frozenset((str(op), str(p))) not in exact_pair_keys:
                matched = (osplit, op, d)
                break
        if matched:
            near_count += 1
            if len(near_pairs) < max_examples:
                near_pairs.append({"a_split": matched[0], "a": str(matched[1]), "b_split": split, "b": str(p), "hamming": matched[2]})
        idx = len(processed)
        processed.append((split, p, ph))
        for band in range(8):
            key = (ph >> (band * 8)) & 0xFF
            buckets[band][key].append(idx)

    embedding_pairs = []
    embedding_count = 0
    if embedding_threshold > 0:
        embedded = {}
        for split in split_names:
            paths, emb = embed_paths(by_split[split], limit=embedding_limit_per_split, backend=embedding_backend)
            embedded[split] = (paths, emb)
        for i, a_split in enumerate(split_names):
            a_paths, a = embedded[a_split]
            if len(a_paths) == 0:
                continue
            for b_split in split_names[i + 1:]:
                b_paths, b = embedded[b_split]
                if len(b_paths) == 0:
                    continue
                for start in range(0, len(b), 256):
                    sims = cosine_similarity_matrix(b[start:start + 256], a)
                    if sims.size == 0:
                        continue
                    best_idx = np.argmax(sims, axis=1)
                    best_sim = sims[np.arange(len(sims)), best_idx]
                    for j, sim in enumerate(best_sim):
                        if float(sim) >= float(embedding_threshold):
                            pa, pb = a_paths[int(best_idx[j])], b_paths[start + j]
                            if frozenset((str(pa), str(pb))) in exact_pair_keys:
                                continue
                            embedding_count += 1
                            if len(embedding_pairs) < max_examples:
                                embedding_pairs.append({
                                    "a_split": a_split, "a": str(pa),
                                    "b_split": b_split, "b": str(pb),
                                    "similarity": float(sim),
                                })

    return {
        "schema": "scenepaste/leakage-report",
        "version": 1,
        "dataset": str(root.resolve()),
        "splits": {k: len(v) for k, v in by_split.items()},
        "exact_cross_split": exact_count,
        "near_cross_split": near_count,
        "embedding_cross_split": embedding_count,
        "phash_threshold": threshold,
        "embedding_threshold": float(embedding_threshold),
        "embedding_backend": embedding_backend,
        "exact_examples": exact_pairs,
        "near_examples": near_pairs,
        "embedding_examples": embedding_pairs,
        "health": "ok" if not (exact_count or near_count or embedding_count) else "warning",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Detect train/val/test exact and visual leakage")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--phash-threshold", type=int, default=6)
    parser.add_argument("--embedding-threshold", type=float, default=0.995)
    parser.add_argument("--embedding-limit", type=int, default=1000)
    parser.add_argument("--embedding-backend", choices=("cv-lite-v1","clip","dinov2"), default="cv-lite-v1")
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args(argv)
    report = detect_split_leakage(args.dataset, args.phash_threshold, args.embedding_threshold, args.embedding_limit, embedding_backend=args.embedding_backend)
    output = args.output or (args.dataset / "leakage_report.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Leakage {report['health']}: exact={report['exact_cross_split']} near={report['near_cross_split']} embedding={report['embedding_cross_split']}")
    print(output)
    return 0 if report["health"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
