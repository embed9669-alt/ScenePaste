"""Document state: cutouts, backgrounds, instances, selection.

Qt-agnostic. The MainWindow owns a :class:`Document` and connects it to
views; the same model could back a non-GUI test harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from compose_app.models import Cutout, Instance


@dataclass
class Document:
    """Mutable document state shared by all views/controllers.

    Attributes:
        cutouts: Loaded object assets (cutouts).
        background_paths: List of background image paths.
        background_index: Index into ``background_paths`` currently shown.
        bg_size: ``(width, height)`` of the current background in pixels.
        instances: Instances placed on the current background.
        selected_uid: UID of the selected instance, or ``None``.
    """

    cutouts: List[Cutout] = field(default_factory=list)
    background_paths: List[Path] = field(default_factory=list)
    background_index: int = -1
    bg_size: tuple = (0, 0)
    instances: List[Instance] = field(default_factory=list)
    selected_uid: Optional[int] = None
    output_dir: Optional[Path] = None
    class_map_text: str = "person=0,vehicle=1"
    # Project settings (mutable via ProjectSettingsDialog / GenerationDefaultsPanel).
    output_format: str = "detect"      # detect | seg | both | coco | semantic | obb | all
    do_shadow: bool = True             # apply foot shadow on composite
    do_color_match: bool = True        # color-match foreground to background
    auto_save: bool = False            # auto-save when switching background
    keep_position: bool = True         # keep relative position across backgrounds
    # Batch / large-generation defaults (also shown on the main window).
    scene_recipe: str = ""             # post-render image recipe name/path
    object_appearance_recipe: str = "" # default cutout appearance for batch/large gen
    blend_mode: str = "alpha"          # alpha | hard | gaussian
    empty_scene_prob: float = 0.0
    _next_uid_counter: int = 1
    _listeners: List[Callable[[], None]] = field(default_factory=list)

    def generation_defaults(self) -> dict:
        """Defaults consumed by Large Generate / project save."""
        return {
            "output_format": self.output_format,
            "augmentation_recipe": self.scene_recipe,
            "object_appearance_recipe": self.object_appearance_recipe,
            "blend_mode": self.blend_mode,
            "empty_scene_prob": float(self.empty_scene_prob),
        }

    # ------------------------------------------------------------------ uid
    def next_uid(self) -> int:
        uid = self._next_uid_counter
        self._next_uid_counter += 1
        return uid

    # ------------------------------------------------------------- listeners
    def subscribe(self, listener: Callable[[], None]) -> None:
        """Register a callback fired on every mutating operation."""
        self._listeners.append(listener)

    def _emit(self) -> None:
        for listener in list(self._listeners):
            listener()

    # ------------------------------------------------------------- queries
    def selected(self) -> Optional[Instance]:
        if self.selected_uid is None:
            return None
        for inst in self.instances:
            if inst.uid == self.selected_uid:
                return inst
        return None

    def index_of(self, inst: Instance) -> int:
        for i, candidate in enumerate(self.instances):
            if candidate.uid == inst.uid:
                return i
        return -1

    def current_background(self) -> Optional[Path]:
        if 0 <= self.background_index < len(self.background_paths):
            return self.background_paths[self.background_index]
        return None

    # ------------------------------------------------------------ mutations
    def set_cutouts(self, cutouts: List[Cutout]) -> None:
        self.cutouts = list(cutouts)
        self._emit()

    def set_backgrounds(self, paths: List[Path], index: int = 0) -> None:
        self.background_paths = list(paths)
        self.background_index = index if paths else -1
        self.instances = []
        self.selected_uid = None
        self._emit()

    def set_bg_size(self, width: int, height: int) -> None:
        self.bg_size = (int(width), int(height))
        self._emit()

    def add_instance(self, inst: Instance) -> None:
        if inst.uid == 0:
            inst.uid = self.next_uid()
        self.instances.append(inst)
        self.selected_uid = inst.uid
        self._emit()

    def remove_instance(self, uid: int) -> None:
        before = len(self.instances)
        self.instances = [i for i in self.instances if i.uid != uid]
        if self.selected_uid == uid:
            self.selected_uid = self.instances[-1].uid if self.instances else None
        if len(self.instances) != before:
            self._emit()

    def replace_instances(self, instances: List[Instance]) -> None:
        self.instances = list(instances)
        if self.selected_uid and not any(i.uid == self.selected_uid for i in self.instances):
            self.selected_uid = None
        self._emit()

    def select(self, uid: Optional[int]) -> None:
        self.selected_uid = uid
        self._emit()

    def snapshot(self) -> List[Instance]:
        """Return a deep copy of the instances list (for undo snapshots)."""
        return [i.clone() for i in self.instances]
