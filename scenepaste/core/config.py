"""Generation config, class-map parsing, and config validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from ..errors import t


@dataclass
class GenerationConfig:
    """All parameters for one ``generate_dataset`` run.

    Attribute defaults are tuned for general street-scene copy-paste. Every
    attribute is keyword-settable so callers can override just what they need.
    """

    objects_dir: Path
    backgrounds_dir: Path
    output_dir: Path
    class_map: Dict[str, int]
    count: int = 100
    min_objects: int = 1
    max_objects: int = 3
    y_min: float = 0.35
    y_max: float = 0.95
    far_height: float = 0.08
    near_height: float = 0.32
    max_iou: float = 0.15
    flip_prob: float = 0.50
    blur_prob: float = 0.25
    color_match_strength: float = 0.25
    feather_sigma: float = 0.8
    seed: int = 42
    save_previews: bool = True
    preview_ratio: float = 1.0       # 0~1; low values (0.01~0.05) for large runs
    background_cache_size: int = 16  # LRU capacity for decoded backgrounds; 0 disables
    coco_checkpoint_interval: int = 1000  # COCO atomic checkpoint every N images; 0 = final only
    asset_sampling: str = "balanced"      # balanced: class-balanced sampling; random: uniform
    background_sampling: str = "balanced"  # balanced: round-robin shuffle; random: uniform
    output_format: str = "detect"   # detect | seg | both | coco | semantic | obb | all
    auto_cutout: bool = False       # rembg-based cutout when no JSON is available
    min_visible_ratio: float = 0.10  # discard instances below this visible fraction after clipping
    run_id: Optional[str] = None     # unique run id; None generates a timestamp
    workers: int = 1                  # 1 = single process; 0 = auto; >1 multiprocessing
    resume: bool = False              # resume an incomplete run using SQLite state
    queue_depth: int = 0              # max in-flight tasks; 0 = workers*2
    distribution_profile: Optional[Path] = None
    profile_strength: float = 1.0     # 0..1, probability of profile-driven planning
    scene_template: Optional[Path] = None
    empty_scene_probability: float = 0.0  # background-only negatives
    augmentation_recipe: Optional[str] = None  # built-in name or JSON path
    object_appearance_recipe: Optional[str] = None  # off/legacy/mild/... or JSON
    blend_mode: str = "alpha"          # alpha | hard | gaussian
    blend_sigma: float = 1.5


def parse_class_map(text: str) -> Dict[str, int]:
    """Parse a ``"person=0,vehicle=1"`` string into ``{label: id}``.

    Commas (both ASCII and full-width) separate entries. Raises ``ValueError``
    on malformed entries, empty labels, duplicate or non-contiguous ids.
    """
    result: Dict[str, int] = {}
    for raw_item in text.replace("，", ",").split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(t("class_map.bad_format", item=item))
        label, class_id = item.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(t("class_map.empty_label"))
        result[label] = int(class_id.strip())
    if not result:
        raise ValueError(t("class_map.need_one"))
    ids = list(result.values())
    if min(ids) < 0 or len(set(ids)) != len(ids):
        raise ValueError(t("class_map.dup_or_negative"))
    if sorted(ids) != list(range(len(ids))):
        raise ValueError(t("class_map.not_contiguous"))
    return result


def validate_config(config: GenerationConfig) -> None:
    """Raise ``ValueError`` if the config is internally inconsistent."""
    if not config.objects_dir.is_dir():
        raise ValueError(t("config.objects_dir_missing", path=config.objects_dir))
    if not config.backgrounds_dir.is_dir():
        raise ValueError(t("config.backgrounds_dir_missing", path=config.backgrounds_dir))
    if config.count <= 0:
        raise ValueError(t("config.count_positive"))
    if config.min_objects < 0 or config.max_objects < config.min_objects:
        raise ValueError(t("config.object_range"))
    if not (0 <= config.y_min < config.y_max <= 1):
        raise ValueError(t("config.y_range"))
    if not (0 < config.far_height <= config.near_height < 1):
        raise ValueError(t("config.height_range"))
    if not (0 <= config.max_iou <= 1):
        raise ValueError(t("config.iou_range"))
    if not (0.0 <= config.preview_ratio <= 1.0):
        raise ValueError(t("config.preview_ratio_range"))
    if config.background_cache_size < 0:
        raise ValueError(t("config.cache_negative"))
    if config.coco_checkpoint_interval < 0:
        raise ValueError(t("config.checkpoint_negative"))
    if config.asset_sampling not in {"balanced", "random"}:
        raise ValueError(t("config.asset_sampling_invalid"))
    if config.background_sampling not in {"balanced", "random"}:
        raise ValueError(t("config.background_sampling_invalid"))
    if config.output_format not in {"detect", "seg", "both", "coco", "semantic", "obb", "all"}:
        raise ValueError(f"不支持的 output_format：{config.output_format}")
    if config.workers < 0:
        raise ValueError("workers 不能小于 0")
    if config.queue_depth < 0:
        raise ValueError("queue_depth 不能小于 0")
    if not (0.0 <= config.profile_strength <= 1.0):
        raise ValueError("profile_strength 必须在 0~1")
    if not (0.0 <= config.empty_scene_probability <= 1.0):
        raise ValueError("empty_scene_probability 必须在 0~1")
    if config.blend_mode not in {"alpha", "hard", "gaussian"}:
        raise ValueError("blend_mode 必须是 alpha / hard / gaussian")
    if config.blend_sigma < 0:
        raise ValueError("blend_sigma 不能小于 0")
    if config.augmentation_recipe:
        from .recipes import load_augmentation_recipe
        load_augmentation_recipe(config.augmentation_recipe)
    if config.object_appearance_recipe:
        from .object_appearance import load_object_appearance_recipe
        load_object_appearance_recipe(config.object_appearance_recipe)
    if config.distribution_profile is not None and not Path(config.distribution_profile).is_file():
        raise ValueError(f"distribution profile 不存在：{config.distribution_profile}")
    if config.scene_template is not None and not Path(config.scene_template).is_file():
        raise ValueError(f"scene template 不存在：{config.scene_template}")
