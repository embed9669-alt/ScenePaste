"""Core engine: asset loading, geometry, sampling, synthesis, pipeline.

This package also re-exports the format writers from :mod:`scenepaste.formats`
so that internal callers can use a single ``core.X`` namespace for everything
the historical ``copy_paste_dataset_tool`` module exposed. The public API
surface is the top-level :mod:`scenepaste` package.
"""

from ..formats import (  # noqa: F401  (re-exported for core.X callers)
    CocoWriter,
    append_coco,
    obb_lines_for_annotations,
    seg_lines_for_annotations,
    ultralytics_obb_line,
    write_classes_file,
    write_data_yaml,
    write_semantic_classes,
    yolo_line,
    yolo_seg_line,
)
from .augmentation import augment_foreground, resize_asset
from .auto_cutout import alpha_to_polygon, load_object_assets_auto, rembg_mask_and_polygon
from .config import GenerationConfig, parse_class_map, validate_config
from .geometry import (
    annotation_canvas_mask,
    box_iou,
    canvas_polygon,
    clip_bbox,
    mask_bbox,
    mask_to_polygons,
    visible_instance_masks,
    visible_ratio,
)
from .io import (
    imread_unicode,
    imread_with_exif,
    imwrite_unicode,
    list_backgrounds,
    load_json,
    log_default,
    save_cutouts,
)
from .labelme import (
    load_object_assets,
    load_paste_zone_mask,
    load_paste_zone_masks,
    polygon_shape,
    resolve_labelme_image,
    shape_to_mask,
    write_labelme_json,
    save_cutout_as_labelme,
)
from .models import IMAGE_SUFFIXES, PASTE_ZONE_LABELS, ObjectAsset, PlacementSpec, is_paste_zone_label
from .pipeline import generate_dataset
from .distribution import DistributionProfile, learn_distribution_profile, compare_profiles, mix_distribution_profiles
from .templates import TemplatePlacement, load_template_data, sample_template, parameterize_payload, template_constraints_satisfied
from .sampling import (
    BackgroundCache,
    BackgroundSampler,
    build_asset_groups,
    sample_asset,
    sample_bottom_point,
)
from .synthesis import paste_one
from .validation import is_valid_yolo_box
from .recipes import BUILTIN_RECIPES, apply_scene_recipe, load_augmentation_recipe, save_augmentation_recipe
from .planning import BUILTIN_HARDCASE_RECIPES, load_hardcase_recipe, placement_to_dict, plan_label_first
from .scene_understanding import resolve_placement_regions
from .object_appearance import (
    BUILTIN_OBJECT_RECIPES,
    apply_object_appearance,
    load_object_appearance_recipe,
    save_object_appearance_recipe,
    validate_object_appearance_recipe,
)
from .asset_studio import (
    AssetStudioExport,
    binary_mask,
    clean_mask,
    export_asset_bundle,
    fill_mask_holes,
    make_clean_background,
    make_foreground_rgba,
    morph_mask,
)

# Underscore-prefixed aliases retained for call sites that still use the
# historical private names. Prefer the public non-underscore spellings.
_append_coco = append_coco
_build_asset_groups = build_asset_groups
_canvas_polygon = canvas_polygon

__all__ = [
    "augment_foreground",
    "resize_asset",
    "alpha_to_polygon",
    "load_object_assets_auto",
    "rembg_mask_and_polygon",
    "GenerationConfig",
    "parse_class_map",
    "validate_config",
    "annotation_canvas_mask",
    "box_iou",
    "canvas_polygon",
    "clip_bbox",
    "mask_bbox",
    "mask_to_polygons",
    "visible_instance_masks",
    "visible_ratio",
    "imread_unicode",
    "imread_with_exif",
    "imwrite_unicode",
    "list_backgrounds",
    "load_json",
    "log_default",
    "save_cutouts",
    "load_object_assets",
    "load_paste_zone_mask",
    "load_paste_zone_masks",
    "polygon_shape",
    "resolve_labelme_image",
    "shape_to_mask",
    "write_labelme_json",
    "save_cutout_as_labelme",
    "IMAGE_SUFFIXES",
    "PASTE_ZONE_LABELS",
    "is_paste_zone_label",
    "ObjectAsset",
    "PlacementSpec",
    "DistributionProfile",
    "learn_distribution_profile",
    "compare_profiles",
    "mix_distribution_profiles",
    "TemplatePlacement",
    "load_template_data",
    "sample_template",
    "parameterize_payload",
    "template_constraints_satisfied",
    "generate_dataset",
    "BackgroundCache",
    "BackgroundSampler",
    "build_asset_groups",
    "sample_asset",
    "sample_bottom_point",
    "paste_one",
    "is_valid_yolo_box",
    "BUILTIN_RECIPES",
    "apply_scene_recipe",
    "load_augmentation_recipe",
    "save_augmentation_recipe",
    "BUILTIN_HARDCASE_RECIPES",
    "load_hardcase_recipe",
    "placement_to_dict",
    "plan_label_first",
    "resolve_placement_regions",
    "BUILTIN_OBJECT_RECIPES",
    "apply_object_appearance",
    "load_object_appearance_recipe",
    "save_object_appearance_recipe",
    "validate_object_appearance_recipe",
    "CocoWriter",
    "append_coco",
    "obb_lines_for_annotations",
    "seg_lines_for_annotations",
    "ultralytics_obb_line",
    "write_classes_file",
    "write_data_yaml",
    "write_semantic_classes",
    "yolo_line",
    "yolo_seg_line",
    "AssetStudioExport",
    "binary_mask",
    "clean_mask",
    "export_asset_bundle",
    "fill_mask_holes",
    "make_clean_background",
    "make_foreground_rgba",
    "morph_mask",
]
