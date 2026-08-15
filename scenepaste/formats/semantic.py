"""Semantic-segmentation class-mapping writer.

Convention: pixel value 0 is always background; a real class with id ``c`` is
encoded as ``c + 1`` so it never collides with background.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def write_semantic_classes(class_map: Dict[str, int], output_dir: Path) -> Path:
    """Write ``semantic_classes.json`` mapping ``"pixel_value" -> class_name``."""
    mapping = {"0": "background"}
    for name, cid in sorted(class_map.items(), key=lambda kv: kv[1]):
        mapping[str(int(cid) + 1)] = name
    path = output_dir / "semantic_classes.json"
    path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
