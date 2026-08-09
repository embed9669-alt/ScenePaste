"""Format writers for ScenePaste output.

Submodules:
- ``yolo``     – YOLO detect / seg / OBB lines, classes.txt, data.yaml
- ``coco``     – COCO instances writer (checkpointable)
- ``semantic`` – semantic-segmentation class mapping
"""

from . import coco, semantic, yolo
from .coco import CocoWriter, append_coco
from .semantic import write_semantic_classes
from .yolo import (
    obb_lines_for_annotations,
    seg_lines_for_annotations,
    ultralytics_obb_line,
    write_classes_file,
    write_data_yaml,
    yolo_line,
    yolo_seg_line,
)

__all__ = [
    "coco",
    "semantic",
    "yolo",
    "CocoWriter",
    "append_coco",
    "write_semantic_classes",
    "obb_lines_for_annotations",
    "seg_lines_for_annotations",
    "ultralytics_obb_line",
    "write_classes_file",
    "write_data_yaml",
    "yolo_line",
    "yolo_seg_line",
]
