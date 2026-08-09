"""Public Python API for ScenePaste.

Example::

    from scenepaste import GenerationConfig, generate_dataset

    summary = generate_dataset(GenerationConfig(...))

Core generation lives in :mod:`scenepaste.core`; annotation writers live in
:mod:`scenepaste.formats`. The public imports below are kept intentionally
small and stable for downstream users.
"""

from .core import (
    BUILTIN_RECIPES,
    BackgroundCache,
    BackgroundSampler,
    DistributionProfile,
    GenerationConfig,
    ObjectAsset,
    PlacementSpec,
    apply_scene_recipe,
    compare_profiles,
    generate_dataset,
    learn_distribution_profile,
    load_augmentation_recipe,
    mix_distribution_profiles,
    parse_class_map,
)
from .formats import CocoWriter
from .project import ScenePasteProject, init_project

__version__ = "1.0.0"

__all__ = [
    "BUILTIN_RECIPES",
    "BackgroundCache",
    "BackgroundSampler",
    "CocoWriter",
    "DistributionProfile",
    "GenerationConfig",
    "ObjectAsset",
    "PlacementSpec",
    "apply_scene_recipe",
    "compare_profiles",
    "generate_dataset",
    "learn_distribution_profile",
    "load_augmentation_recipe",
    "mix_distribution_profiles",
    "parse_class_map",
    "ScenePasteProject",
    "init_project",
    "__version__",
]
