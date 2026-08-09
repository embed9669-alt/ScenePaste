"""ScenePaste project manifest.

A project file captures paths and default generation settings so a team can
reopen the same data workflow without reconstructing long CLI commands.
Paths are stored relative to the manifest when possible, making projects
portable across machines and repositories.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional

SCHEMA = "scenepaste/project"
VERSION = 1
DEFAULT_FILENAME = "scenepaste.project.json"


def _rel(base: Path, value: Optional[Path]) -> Optional[str]:
    if value is None:
        return None
    p = Path(value).expanduser()
    try:
        return str(p.resolve().relative_to(base.resolve()))
    except Exception:
        return str(p.resolve())


def _resolve(base: Path, value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    p = Path(value).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


@dataclass
class ScenePasteProject:
    path: Path
    name: str = "ScenePaste Project"
    objects_dir: Optional[Path] = None
    backgrounds_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    class_map: Dict[str, int] = field(default_factory=lambda: {"person": 0, "vehicle": 1})
    real_dataset: Optional[Path] = None
    validation_dataset: Optional[Path] = None
    predictions_dir: Optional[Path] = None
    distribution_profile: Optional[Path] = None
    scene_template: Optional[Path] = None
    defaults: Dict[str, object] = field(default_factory=dict)

    @property
    def base_dir(self) -> Path:
        return Path(self.path).resolve().parent

    @classmethod
    def load(cls, path: Path) -> "ScenePasteProject":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != SCHEMA:
            raise ValueError(f"不是 ScenePaste project manifest: {path}")
        manifest_version = int(data.get("version", 1))
        if manifest_version > VERSION:
            raise ValueError(
                f"project manifest version {manifest_version} is newer than supported version {VERSION}"
            )
        base = path.resolve().parent
        paths = data.get("paths", {}) or {}
        return cls(
            path=path,
            name=str(data.get("name") or path.stem),
            objects_dir=_resolve(base, paths.get("objects")),
            backgrounds_dir=_resolve(base, paths.get("backgrounds")),
            output_dir=_resolve(base, paths.get("output")),
            class_map={str(k): int(v) for k, v in (data.get("class_map") or {}).items()},
            real_dataset=_resolve(base, paths.get("real_dataset")),
            validation_dataset=_resolve(base, paths.get("validation_dataset")),
            predictions_dir=_resolve(base, paths.get("predictions")),
            distribution_profile=_resolve(base, paths.get("distribution_profile")),
            scene_template=_resolve(base, paths.get("scene_template")),
            defaults=dict(data.get("defaults") or {}),
        )

    def to_dict(self) -> dict:
        base = self.base_dir
        return {
            "schema": SCHEMA,
            "version": VERSION,
            "name": self.name,
            "paths": {
                "objects": _rel(base, self.objects_dir),
                "backgrounds": _rel(base, self.backgrounds_dir),
                "output": _rel(base, self.output_dir),
                "real_dataset": _rel(base, self.real_dataset),
                "validation_dataset": _rel(base, self.validation_dataset),
                "predictions": _rel(base, self.predictions_dir),
                "distribution_profile": _rel(base, self.distribution_profile),
                "scene_template": _rel(base, self.scene_template),
            },
            "class_map": dict(sorted(self.class_map.items(), key=lambda kv: kv[1])),
            "defaults": self.defaults,
        }

    def save(self, path: Optional[Path] = None) -> Path:
        if path is not None:
            self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return self.path

    def validate(self, require_generation_paths: bool = False) -> dict:
        checks = {}
        path_specs = {
            "objects": (self.objects_dir, "dir"),
            "backgrounds": (self.backgrounds_dir, "dir"),
            "output": (self.output_dir, "output"),
            "real_dataset": (self.real_dataset, "dir"),
            "validation_dataset": (self.validation_dataset, "dir"),
            "predictions": (self.predictions_dir, "dir"),
            "distribution_profile": (self.distribution_profile, "file"),
            "scene_template": (self.scene_template, "file"),
        }
        errors = []
        for key, (value, expected) in path_specs.items():
            if value is None:
                checks[key] = {"configured": False, "exists": None, "kind_ok": None, "path": None}
                continue
            exists = value.exists()
            if expected == "dir":
                kind_ok = value.is_dir() if exists else False
            elif expected == "file":
                kind_ok = value.is_file() if exists else False
            else:
                # Output may not exist yet, but an existing output must be a directory.
                kind_ok = (not exists) or value.is_dir()
            checks[key] = {
                "configured": True, "exists": exists, "kind_ok": kind_ok, "path": str(value)
            }

        if require_generation_paths:
            for key in ("objects", "backgrounds", "output"):
                if not checks[key]["configured"]:
                    errors.append(f"missing path: {key}")

        for key in ("objects", "backgrounds", "real_dataset", "validation_dataset", "predictions"):
            row = checks[key]
            if row["configured"] and not row["exists"]:
                errors.append(f"path not found: {key}={row['path']}")
            elif row["configured"] and not row["kind_ok"]:
                errors.append(f"expected directory: {key}={row['path']}")
        for key in ("distribution_profile", "scene_template"):
            row = checks[key]
            if row["configured"] and not row["exists"]:
                errors.append(f"file not found: {key}={row['path']}")
            elif row["configured"] and not row["kind_ok"]:
                errors.append(f"expected file: {key}={row['path']}")
        output = checks["output"]
        if output["configured"] and not output["kind_ok"]:
            errors.append(f"output path is not a directory: {output['path']}")

        ids = sorted(self.class_map.values())
        if ids and ids != list(range(len(ids))):
            errors.append("class_map ids must be contiguous from 0")

        defaults = self.defaults
        if "workers" in defaults and (not isinstance(defaults["workers"], int) or defaults["workers"] < 0):
            errors.append("defaults.workers must be an integer >= 0")
        if "output_format" in defaults and defaults["output_format"] not in {
            "detect", "seg", "both", "coco", "semantic", "obb", "all"
        }:
            errors.append("defaults.output_format is invalid")
        for key in ("preview_ratio", "profile_strength", "empty_scene_prob"):
            if key in defaults:
                try:
                    value = float(defaults[key])
                except (TypeError, ValueError):
                    errors.append(f"defaults.{key} must be numeric")
                else:
                    if not 0.0 <= value <= 1.0:
                        errors.append(f"defaults.{key} must be in 0..1")
        if "blend_mode" in defaults and defaults["blend_mode"] not in {"alpha", "hard", "gaussian"}:
            errors.append("defaults.blend_mode is invalid")

        return {"ok": not errors, "errors": errors, "checks": checks, "class_map": self.class_map}

    def update_generation(
        self,
        *,
        objects_dir: Optional[Path] = None,
        backgrounds_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        class_map: Optional[Dict[str, int]] = None,
        distribution_profile: Optional[Path] = None,
        scene_template: Optional[Path] = None,
        defaults: Optional[Mapping[str, object]] = None,
    ) -> "ScenePasteProject":
        """Update generation-facing fields without discarding workflow metadata.

        The desktop editor uses this when saving an already-open project.
        Real/validation datasets and model-prediction paths therefore survive
        normal scene-editing saves instead of being accidentally reset.
        """
        if objects_dir is not None:
            self.objects_dir = Path(objects_dir)
        if backgrounds_dir is not None:
            self.backgrounds_dir = Path(backgrounds_dir)
        if output_dir is not None:
            self.output_dir = Path(output_dir)
        if class_map is not None:
            self.class_map = dict(class_map)
        if distribution_profile is not None:
            self.distribution_profile = Path(distribution_profile)
        if scene_template is not None:
            self.scene_template = Path(scene_template)
        if defaults:
            self.defaults.update(dict(defaults))
        return self

    def generation_defaults(self) -> Mapping[str, object]:
        return self.defaults


def init_project(path: Path, *, name: Optional[str] = None, objects: Optional[Path] = None,
                 backgrounds: Optional[Path] = None, output: Optional[Path] = None,
                 class_map: Optional[Dict[str, int]] = None) -> ScenePasteProject:
    path = Path(path)
    if path.is_dir() or path.suffix == "":
        path = path / DEFAULT_FILENAME
    project = ScenePasteProject(
        path=path,
        name=name or path.parent.name or "ScenePaste Project",
        objects_dir=objects,
        backgrounds_dir=backgrounds,
        output_dir=output,
        class_map=class_map or {"person": 0, "vehicle": 1},
        defaults={"workers": 0, "output_format": "all", "preview_ratio": 0.01},
    )
    project.save()
    return project
