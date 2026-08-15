"""Asset and background sampling + decoded-background LRU cache."""

from __future__ import annotations

import random
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .io import imread_with_exif
from .models import ObjectAsset


class BackgroundCache:
    """LRU cache of decoded background images.

    Thread-unsafe by design — single-thread use within one
    ``generate_dataset`` run, or callers serialize access.
    """

    def __init__(self, capacity: int = 16):
        self.capacity = max(1, capacity)
        self._store: "OrderedDict[Path, np.ndarray]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, path: Path) -> Optional[np.ndarray]:
        """Return a decoded background, caching it for next time."""
        img = self._store.get(path)
        if img is not None:
            self.hits += 1
            self._store.move_to_end(path)
            return img
        self.misses += 1
        decoded = imread_with_exif(path)
        if decoded is None:
            return None
        self._store[path] = decoded
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)
        return decoded

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses,
                "size": len(self._store), "capacity": self.capacity}


class BackgroundSampler:
    """Background sampler. ``balanced`` round-robin-shuffles per cycle so long-run
    usage counts differ by at most 1; ``random`` is uniform per draw.
    """

    def __init__(self, paths: Sequence[Path], rng: random.Random, mode: str = "balanced"):
        if not paths:
            raise ValueError("背景列表不能为空")
        self.paths = list(paths)
        self.rng = rng
        self.mode = mode
        self._order: List[Path] = []
        self._index = 0
        self._reshuffle()

    def _reshuffle(self) -> None:
        self._order = list(self.paths)
        self.rng.shuffle(self._order)
        self._index = 0

    def next(self) -> Path:
        if self.mode == "random":
            return self.rng.choice(self.paths)
        if self._index >= len(self._order):
            self._reshuffle()
        path = self._order[self._index]
        self._index += 1
        return path


def build_asset_groups(assets: Sequence[ObjectAsset]) -> Dict[int, List[ObjectAsset]]:
    """Group assets by ``class_id`` for class-balanced sampling."""
    groups: Dict[int, List[ObjectAsset]] = {}
    for asset in assets:
        groups.setdefault(asset.class_id, []).append(asset)
    return groups


def sample_asset(
    rng: random.Random,
    assets: Sequence[ObjectAsset],
    groups: Dict[int, List[ObjectAsset]],
    mode: str = "balanced",
) -> ObjectAsset:
    """Sample one asset. ``balanced`` picks a class uniformly first, then an
    asset inside it; ``random`` picks uniformly across all assets.
    """
    if mode == "random" or len(groups) <= 1:
        return rng.choice(list(assets))
    class_id = rng.choice(sorted(groups))
    return rng.choice(groups[class_id])


def sample_bottom_point(
    rng: random.Random,
    width: int,
    height: int,
    y_min: float,
    y_max: float,
    zone_mask: Optional[np.ndarray],
) -> Tuple[int, int]:
    """Sample a placement point. When a paste-zone mask exists, the point is
    drawn from inside the zone; otherwise uniformly over the ground band.
    """
    if zone_mask is not None:
        ys, xs = np.where(zone_mask > 0)
        if len(xs) > 0:
            index = rng.randrange(len(xs))
            return int(xs[index]), int(ys[index])
    x = int(rng.uniform(0.08, 0.92) * width)
    y = int(rng.uniform(y_min, y_max) * height)
    return x, y
