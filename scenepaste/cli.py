"""Unified ScenePaste command-line interface.

Subcommands:
    gui       – Launch the interactive scene editor
    generate  – Generate a synthetic dataset
    analyze   – Run dataset QA and statistics
    split     – Split a dataset with source-aware leakage protection
    merge     – Merge ScenePaste datasets
    explore   – Browse a dataset with annotation overlays
    profile   – Learn, inspect, or mix real-data distribution profiles
    qa        – Generate JSON + HTML dataset QA reports
    recipe    – Inspect or export scene / object appearance recipes
    curate    – Hard mining, leakage checks, and diversity selection
    compare   – Compare real and synthetic datasets
    shard     – Build WebDataset-compatible tar shards
    project   – Manage a portable ScenePaste project manifest
    loop      – Launch the unified data-loop center
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import __version__

_COMMANDS = {
    "gui": "Launch the interactive scene editor",
    "generate": "Generate a synthetic dataset (copy-paste)",
    "analyze": "Run dataset QA and statistics",
    "split": "Split a dataset with source-aware leakage protection",
    "merge": "Merge ScenePaste datasets",
    "explore": "Browse a dataset with annotation overlays",
    "profile": "Learn/show a real-data distribution profile",
    "qa": "Generate a full QA JSON + HTML dashboard",
    "recipe": "List/show/export scene or object appearance recipes",
    "curate": "Hard mining, leakage checks, and diversity selection",
    "compare": "Compare real and synthetic datasets",
    "shard": "Build WebDataset-compatible tar shards",
    "project": "Create/show/validate a ScenePaste project manifest",
    "loop": "Launch the unified data-loop center",
    "factory": "Emit deterministic label-first placement plans",
}


def _print_help() -> None:
    print(
        "ScenePaste — scene-first synthetic dataset generation\n\n"
        "Usage:\n"
        "  scenepaste <command> [options]\n\n"
        "Commands:\n"
        + "\n".join(f"  {name:<10} {desc}" for name, desc in _COMMANDS.items())
        + "\n\n"
        "Examples:\n"
        "  scenepaste gui --objects ./objects --backgrounds ./backgrounds --output ./generated\n"
        "  scenepaste generate --objects ./objects --backgrounds ./backgrounds --output ./generated --count 1000\n"
        "  scenepaste analyze ./generated\n"
        "  scenepaste explore ./generated\n"
        "  scenepaste profile learn ./real_dataset -o distribution_profile.json\n"
        "  scenepaste qa ./generated\n"
        "  scenepaste curate hardmine ./dataset --predictions ./runs/predict/labels\n"
        "  scenepaste compare ./real ./generated\n"
        "  scenepaste shard ./generated -o ./shards\n"
        "  scenepaste project init . --objects ./objects --backgrounds ./backgrounds --output ./generated\n"
        "  scenepaste loop --project ./scenepaste.project.json\n"
        "  scenepaste factory plan --count 20 -o plans.json\n\n"
        "Use `scenepaste <command> --help` for command-specific options."
    )


def build_generate_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the ``generate`` subcommand."""
    parser = argparse.ArgumentParser(
        prog="scenepaste generate",
        description="ScenePaste - controllable synthetic vision dataset generator",
    )
    parser.add_argument("--project", type=Path, default=None, help="ScenePaste project manifest；用于填充路径/类别/默认参数")
    parser.add_argument("--objects", type=Path, help="目标 JSON 和原图所在目录")
    parser.add_argument("--backgrounds", type=Path, help="真实背景图目录")
    parser.add_argument("--output", type=Path, help="输出目录")
    parser.add_argument("--count", type=int, default=100, help="生成图片数量")
    parser.add_argument("--min-objects", type=int, default=1, help="每张最少目标数")
    parser.add_argument("--max-objects", type=int, default=3, help="每张最多目标数")
    parser.add_argument("--class-map", default="person=0,vehicle=1", help="类别映射")
    parser.add_argument("--y-min", type=float, default=0.35, help="默认落地点最小纵向比例")
    parser.add_argument("--y-max", type=float, default=0.95, help="默认落地点最大纵向比例")
    parser.add_argument("--far-height", type=float, default=0.08, help="远处目标高度/背景高度")
    parser.add_argument("--near-height", type=float, default=0.32, help="近处目标高度/背景高度")
    parser.add_argument("--max-iou", type=float, default=0.15, help="合成目标间最大 IoU")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--no-preview", action="store_true", help="不保存带框预览图")
    parser.add_argument("--preview-ratio", type=float, default=1.0,
                        help="预览抽样比例 0~1；海量生成建议 0.01（--no-preview 优先）")
    parser.add_argument("--background-cache-size", type=int, default=16,
                        help="背景解码 LRU 缓存容量，默认 16；0 关闭")
    parser.add_argument("--coco-checkpoint-interval", type=int, default=1000,
                        help="兼容参数：当前使用逐样本 COCO fragment + 最终流式汇总，此参数不再控制主写入路径")
    parser.add_argument("--asset-sampling", choices=("balanced", "random"), default="balanced",
                        help="目标采样：balanced 先均衡类别再抽素材；random 所有素材等概率")
    parser.add_argument("--background-sampling", choices=("balanced", "random"), default="balanced",
                        help="背景采样：balanced 轮转洗牌减少复用偏斜；random 完全随机")
    parser.add_argument("--output-format", default="detect",
                        choices=("detect", "seg", "both", "coco", "semantic", "obb", "all"),
                        help="标注输出格式；all 一次导出 Detect/Seg/OBB/Semantic/COCO")
    parser.add_argument("--auto-cutout", action="store_true",
                        help="自动抠图：直接读 objects/ 下原图用 rembg 抠出（需要 pip install 'scenepaste[auto]'）")
    parser.add_argument("--auto-cutout-label", default=None,
                        help="自动抠图默认类别名（默认 auto）；可与 class-map 对齐，如 person")
    parser.add_argument("--auto-cutout-label-from-subdir", action="store_true",
                        help="按子目录名作为类别（objects/person/*.jpg → person）")
    parser.add_argument("--rectangle-mask-mode", choices=("grabcut", "reject", "legacy"), default="grabcut",
                        help="矩形标注处理：grabcut 自动细化前景（推荐）；reject 拒绝 bbox；legacy 旧版整块矩形")
    parser.add_argument("--run-id", default=None,
                        help="唯一 Run ID（不填则按时间戳自动生成：run_YYYYMMDD_HHMMSS_microseconds），防止高并发启动时重名")
    parser.add_argument("--workers", type=int, default=1,
                        help="并行进程数；1=兼容单进程，0=自动使用 CPU-1")
    parser.add_argument("--resume", action="store_true",
                        help="恢复未完成 run；可结合 --run-id，省略 run-id 时恢复最新未完成 run")
    parser.add_argument("--queue-depth", type=int, default=0,
                        help="多进程最大在途任务数；0=workers*2，百万级任务不会一次性提交")
    parser.add_argument("--distribution-profile", type=Path, default=None,
                        help="真实数据分布 profile JSON；用 `scenepaste profile learn` 生成")
    parser.add_argument("--profile-strength", type=float, default=1.0,
                        help="0~1：每张图采用真实分布规划的概率")
    parser.add_argument("--scene-template", type=Path, default=None,
                        help="参数化 Scene Template；支持关系约束，模板优先于 distribution profile")
    parser.add_argument("--empty-scene-prob", type=float, default=0.0,
                        help="0~1：生成纯背景负样本的概率；负样本写空检测/分割标签")
    parser.add_argument("--augmentation-recipe", default=None,
                        help="图像级增强 Recipe：clean/camera-mild/surveillance/low-light 或自定义 JSON")
    parser.add_argument("--object-appearance-recipe", default=None,
                        help="目标级外观 Recipe：off/legacy/mild/surveillance-object 或自定义 JSON；默认保持 v1.0 轻量 HSV")
    parser.add_argument("--blend-mode", choices=("alpha", "hard", "gaussian"), default="alpha",
                        help="目标边缘混合方式；gaussian 可减轻硬边界")
    parser.add_argument("--blend-sigma", type=float, default=1.5,
                        help="blend-mode=gaussian 时的 alpha 高斯 sigma")

    parser.add_argument("--scene-region-mode", choices=("auto", "explicit", "ground-prior", "none"), default="auto",
                        help="可放置区域来源：显式 LabelMe 优先/地面先验/关闭")
    parser.add_argument("--hardcase-recipe", default=None,
                        help="主动难例 Recipe：small-object/far-occluded/crowded 或自定义 JSON")
    return parser


def config_from_args(args: argparse.Namespace):
    """Convert parsed argparse args into a :class:`GenerationConfig`."""
    from .core.config import GenerationConfig, parse_class_map
    from .errors import t

    project = None
    if getattr(args, "project", None):
        from .project import ScenePasteProject

        project = ScenePasteProject.load(args.project)
        provided = set(getattr(args, "_provided_options", set()))
        if args.objects is None:
            args.objects = project.objects_dir
        if args.backgrounds is None:
            args.backgrounds = project.backgrounds_dir
        if args.output is None:
            args.output = project.output_dir
        if "--class-map" not in provided and project.class_map:
            args.class_map = ",".join(
                f"{k}={v}" for k, v in sorted(project.class_map.items(), key=lambda kv: kv[1])
            )
        if "--distribution-profile" not in provided and project.distribution_profile is not None:
            args.distribution_profile = project.distribution_profile
        if "--scene-template" not in provided and project.scene_template is not None:
            args.scene_template = project.scene_template

        # Manifest defaults are project-level defaults, while explicitly supplied
        # CLI options always win — even when their value equals argparse's default.
        option_by_key = {
            "count": "--count",
            "min_objects": "--min-objects",
            "max_objects": "--max-objects",
            "y_min": "--y-min",
            "y_max": "--y-max",
            "far_height": "--far-height",
            "near_height": "--near-height",
            "max_iou": "--max-iou",
            "seed": "--seed",
            "preview_ratio": "--preview-ratio",
            "background_cache_size": "--background-cache-size",
            "asset_sampling": "--asset-sampling",
            "background_sampling": "--background-sampling",
            "output_format": "--output-format",
            "rectangle_mask_mode": "--rectangle-mask-mode",
            "workers": "--workers",
            "queue_depth": "--queue-depth",
            "profile_strength": "--profile-strength",
            "empty_scene_prob": "--empty-scene-prob",
            "augmentation_recipe": "--augmentation-recipe",
            "object_appearance_recipe": "--object-appearance-recipe",
            "blend_mode": "--blend-mode",
            "blend_sigma": "--blend-sigma",
            "scene_region_mode": "--scene-region-mode",
            "hardcase_recipe": "--hardcase-recipe",
        }
        for key, option in option_by_key.items():
            if key in project.defaults and option not in provided:
                value = project.defaults[key]
                # Custom recipe paths in a portable project are interpreted
                # relative to the manifest, while built-in recipe names remain
                # untouched. This lets the same project run from any CWD.
                if key in {"augmentation_recipe", "object_appearance_recipe", "hardcase_recipe"} and isinstance(value, str):
                    candidate = Path(value).expanduser()
                    if not candidate.is_absolute():
                        manifest_candidate = (project.base_dir / candidate).resolve()
                        if manifest_candidate.is_file():
                            value = str(manifest_candidate)
                setattr(args, key, value)
    missing = [name for name in ["objects", "backgrounds", "output"] if getattr(args, name) is None]
    if missing:
        raise ValueError(t("cli.missing_args", names=", ".join("--" + name for name in missing)))
    return GenerationConfig(
        objects_dir=args.objects,
        backgrounds_dir=args.backgrounds,
        output_dir=args.output,
        class_map=parse_class_map(args.class_map),
        count=args.count,
        min_objects=args.min_objects,
        max_objects=args.max_objects,
        y_min=args.y_min,
        y_max=args.y_max,
        far_height=args.far_height,
        near_height=args.near_height,
        max_iou=args.max_iou,
        seed=args.seed,
        save_previews=not args.no_preview,
        preview_ratio=args.preview_ratio,
        background_cache_size=args.background_cache_size,
        coco_checkpoint_interval=args.coco_checkpoint_interval,
        asset_sampling=args.asset_sampling,
        background_sampling=args.background_sampling,
        output_format=args.output_format,
        auto_cutout=args.auto_cutout,
        auto_cutout_label=getattr(args, "auto_cutout_label", None),
        auto_cutout_label_from_subdir=bool(getattr(args, "auto_cutout_label_from_subdir", False)),
        rectangle_mask_mode=getattr(args, "rectangle_mask_mode", "grabcut"),
        run_id=args.run_id,
        workers=args.workers,
        resume=args.resume,
        queue_depth=args.queue_depth,
        distribution_profile=args.distribution_profile,
        profile_strength=args.profile_strength,
        scene_template=args.scene_template,
        empty_scene_probability=args.empty_scene_prob,
        augmentation_recipe=args.augmentation_recipe,
        object_appearance_recipe=getattr(args, "object_appearance_recipe", None),
        blend_mode=args.blend_mode,
        blend_sigma=args.blend_sigma,
        scene_region_mode=args.scene_region_mode,
        hardcase_recipe=args.hardcase_recipe,
    )


def generate_main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for ``scenepaste generate``."""
    from .core.pipeline import generate_dataset

    parser = build_generate_parser()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_args)
    args._provided_options = {token.split("=", 1)[0] for token in raw_args if token.startswith("--")}
    # Backwards-compatible behaviour: a truly empty generation request opens
    # the GUI. A project manifest is itself a complete generation request and
    # must not accidentally trigger GUI fallback.
    if args.project is None and args.objects is None and args.backgrounds is None and args.output is None:
        try:
            from compose_app_qt.main_entry import main_entry
            return int(main_entry([]) or 0)
        except ImportError as exc:  # pragma: no cover - GUI optional
            print(
                f"GUI 启动失败：{exc}\n"
                "请安装 Qt 依赖：python -m pip install 'scenepaste[gui-qt]'",
                file=sys.stderr,
            )
            return 1
        except Exception as exc:  # pragma: no cover - GUI optional
            print(f"GUI 启动失败：{exc}", file=sys.stderr)
            return 1
    previous_sigterm = None

    def _sigterm_as_interrupt(_signum, _frame):
        # On POSIX this makes QProcess.terminate()/SIGTERM take the same
        # crash-safe finalization path as Ctrl+C. Windows may terminate the
        # process directly; per-sample fragments + SQLite still make resume safe.
        raise KeyboardInterrupt

    try:
        if hasattr(signal, "SIGTERM"):
            previous_sigterm = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, _sigterm_as_interrupt)
        config = config_from_args(args)
        summary = generate_dataset(config)
        return 130 if summary.get("status") == "interrupted" else 0
    except Exception as exc:
        from .errors import t
        print(t("cli.error", msg=str(exc)), file=sys.stderr)
        return 1
    finally:
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)



def explore_main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for ``scenepaste explore [dataset]``."""
    parser = argparse.ArgumentParser(
        prog="scenepaste explore",
        description="Browse a ScenePaste dataset with annotation overlays",
    )
    parser.add_argument("dataset", nargs="?", type=Path, help="dataset root containing images/train|val|test")
    args = parser.parse_args(argv)
    try:
        from compose_app_qt.explorer import launch_dataset_explorer
    except ImportError as exc:
        print(
            f"Dataset Explorer 启动失败：{exc}\n"
            "请安装 Qt 依赖：python -m pip install 'scenepaste[gui-qt]'",
            file=sys.stderr,
        )
        return 1
    return int(launch_dataset_explorer(args.dataset) or 0)

def profile_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="scenepaste profile", description="Learn or inspect real-data distributions")
    sub = parser.add_subparsers(dest="action", required=True)
    learn = sub.add_parser("learn", help="learn a compact profile from LabelMe/YOLO/COCO")
    learn.add_argument("dataset", type=Path)
    learn.add_argument("-o", "--output", type=Path, default=Path("distribution_profile.json"))
    learn.add_argument("--bins", type=int, default=20)
    learn.add_argument("--geometry-source", choices=("auto","detect","seg","obb"), default="auto", help="prefer geometry source when multiple YOLO modalities exist")
    show = sub.add_parser("show", help="print a profile summary")
    show.add_argument("profile", type=Path)
    mix = sub.add_parser("mix", help="mix multiple domain profiles with explicit weights")
    mix.add_argument("profiles", nargs="+", type=Path)
    mix.add_argument("-w", "--weights", nargs="+", type=float, default=None)
    mix.add_argument("-o", "--output", type=Path, default=Path("mixed_distribution_profile.json"))
    args = parser.parse_args(argv)
    from .core.distribution import DistributionProfile, learn_distribution_profile, mix_distribution_profiles
    if args.action == "learn":
        profile = learn_distribution_profile(args.dataset, bins=args.bins, geometry_source=args.geometry_source)
        profile.save(args.output)
        print(f"Profile saved: {args.output} · images={profile.data['image_count']} · objects={profile.data['object_count_total']}")
        for name, row in profile.classes.items():
            print(f"  {name}: {row.get('count',0)}")
        return 0
    if args.action == "mix":
        loaded = [DistributionProfile.load(p) for p in args.profiles]
        mixed = mix_distribution_profiles(loaded, args.weights)
        mixed.save(args.output)
        print(f"Mixed profile saved: {args.output} · domains={len(loaded)}")
        return 0
    profile = DistributionProfile.load(args.profile)
    print(json.dumps(profile.data, ensure_ascii=False, indent=2))
    return 0



def recipe_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scenepaste recipe",
        description="Inspect ScenePaste scene or object appearance recipes",
    )
    parser.add_argument(
        "--kind",
        choices=("scene", "object"),
        default="scene",
        help="scene = post-render image recipe; object = per-cutout appearance recipe",
    )
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list", help="list built-in recipes")
    show = sub.add_parser("show", help="print one built-in or JSON recipe")
    show.add_argument("recipe")
    export = sub.add_parser("export", help="write a built-in recipe to JSON for editing")
    export.add_argument("recipe")
    export.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.kind == "object":
        from .core.object_appearance import (
            BUILTIN_OBJECT_RECIPES,
            load_object_appearance_recipe,
            save_object_appearance_recipe,
        )
        if args.action == "list":
            for name in sorted(BUILTIN_OBJECT_RECIPES):
                print(name)
            return 0
        recipe = load_object_appearance_recipe(args.recipe)
        if args.action == "show":
            print(json.dumps(recipe, ensure_ascii=False, indent=2))
            return 0
        save_object_appearance_recipe(recipe or BUILTIN_OBJECT_RECIPES["off"], args.output)
        print(f"Object appearance recipe saved: {args.output}")
        return 0

    from .core.recipes import BUILTIN_RECIPES, load_augmentation_recipe, save_augmentation_recipe
    if args.action == "list":
        for name in sorted(BUILTIN_RECIPES):
            print(name)
        return 0
    recipe = load_augmentation_recipe(args.recipe)
    if args.action == "show":
        print(json.dumps(recipe, ensure_ascii=False, indent=2))
        return 0
    save_augmentation_recipe(recipe or BUILTIN_RECIPES["clean"], args.output)
    print(f"Recipe saved: {args.output}")
    return 0


def qa_main(argv: Optional[Sequence[str]] = None) -> int:
    from .tools.qa import main as _main
    return int(_main(argv) or 0)



def curate_main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(argv or [])
    if not args or args[0] in {"-h", "--help", "help"}:
        print(
            "ScenePaste data curation\n\n"
            "Usage:\n"
            "  scenepaste curate hardmine DATASET --predictions PRED_DIR [options]\n"
            "  scenepaste curate leakage DATASET [options]\n"
            "  scenepaste curate diversity DATASET [options]\n"
        )
        return 0
    action, rest = args[0], args[1:]
    if action == "hardmine":
        from .tools.hardmine import main as _main
        return int(_main(rest) or 0)
    if action == "leakage":
        from .tools.leakage import main as _main
        return int(_main(rest) or 0)
    if action == "diversity":
        from .tools.diversity import main as _main
        return int(_main(rest) or 0)
    print(f"Unknown curate action: {action}", file=sys.stderr)
    return 2


def compare_main(argv: Optional[Sequence[str]] = None) -> int:
    from .tools.compare import main as _main
    return int(_main(argv) or 0)


def shard_main(argv: Optional[Sequence[str]] = None) -> int:
    from .tools.shard import main as _main
    return int(_main(argv) or 0)


def project_main(argv: Optional[Sequence[str]] = None) -> int:
    from .project import ScenePasteProject, init_project
    from .core.config import parse_class_map
    parser = argparse.ArgumentParser(prog="scenepaste project", description="Manage a portable ScenePaste project manifest")
    sub = parser.add_subparsers(dest="action", required=True)
    init = sub.add_parser("init", help="create scenepaste.project.json")
    init.add_argument("path", nargs="?", type=Path, default=Path("."))
    init.add_argument("--name", default=None)
    init.add_argument("--objects", type=Path, default=None)
    init.add_argument("--backgrounds", type=Path, default=None)
    init.add_argument("--output", type=Path, default=None)
    init.add_argument("--class-map", default="person=0,vehicle=1")
    show = sub.add_parser("show", help="print resolved project manifest")
    show.add_argument("project", type=Path)
    val = sub.add_parser("validate", help="validate project paths and class map")
    val.add_argument("project", type=Path)
    val.add_argument("--generation", action="store_true", help="require objects/backgrounds/output")
    setp = sub.add_parser("set", help="update project paths/settings")
    setp.add_argument("project", type=Path)
    setp.add_argument("--name")
    setp.add_argument("--objects", type=Path)
    setp.add_argument("--backgrounds", type=Path)
    setp.add_argument("--output", type=Path)
    setp.add_argument("--real-dataset", type=Path)
    setp.add_argument("--validation-dataset", type=Path)
    setp.add_argument("--predictions", type=Path)
    setp.add_argument("--distribution-profile", type=Path)
    setp.add_argument("--scene-template", type=Path)
    setp.add_argument("--class-map")
    setp.add_argument("--workers", type=int)
    setp.add_argument("--output-format", choices=("detect", "seg", "both", "coco", "semantic", "obb", "all"))
    setp.add_argument("--preview-ratio", type=float)
    setp.add_argument("--profile-strength", type=float)
    setp.add_argument("--empty-scene-prob", type=float)
    setp.add_argument("--augmentation-recipe")
    setp.add_argument("--object-appearance-recipe")
    setp.add_argument("--blend-mode", choices=("alpha", "hard", "gaussian"))
    args = parser.parse_args(argv)
    if args.action == "init":
        project = init_project(args.path, name=args.name, objects=args.objects, backgrounds=args.backgrounds,
                               output=args.output, class_map=parse_class_map(args.class_map))
        print(project.path)
        return 0
    project = ScenePasteProject.load(args.project)
    if args.action == "set":
        mapping = {
            "objects": "objects_dir", "backgrounds": "backgrounds_dir", "output": "output_dir",
            "real_dataset": "real_dataset", "validation_dataset": "validation_dataset", "predictions": "predictions_dir",
            "distribution_profile": "distribution_profile", "scene_template": "scene_template",
        }
        for arg_name, attr in mapping.items():
            value = getattr(args, arg_name, None)
            if value is not None:
                setattr(project, attr, value.resolve())
        if args.name:
            project.name = args.name
        if args.class_map:
            project.class_map = parse_class_map(args.class_map)
        default_args = {
            "workers": args.workers,
            "output_format": args.output_format,
            "preview_ratio": args.preview_ratio,
            "profile_strength": args.profile_strength,
            "empty_scene_prob": args.empty_scene_prob,
            "augmentation_recipe": args.augmentation_recipe,
            "object_appearance_recipe": args.object_appearance_recipe,
            "blend_mode": args.blend_mode,
        }
        for key, value in default_args.items():
            if value is not None:
                project.defaults[key] = value
        project.save()
        print(project.path)
        return 0
    if args.action == "show":
        payload = project.to_dict()
        payload["resolved"] = {
            "objects": str(project.objects_dir) if project.objects_dir else None,
            "backgrounds": str(project.backgrounds_dir) if project.backgrounds_dir else None,
            "output": str(project.output_dir) if project.output_dir else None,
            "real_dataset": str(project.real_dataset) if project.real_dataset else None,
            "validation_dataset": str(project.validation_dataset) if project.validation_dataset else None,
            "predictions": str(project.predictions_dir) if project.predictions_dir else None,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    report = project.validate(require_generation_paths=args.generation)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


def loop_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="scenepaste loop", description="Launch the ScenePaste data-loop center")
    parser.add_argument("dataset", nargs="?", type=Path, default=None)
    parser.add_argument("--project", type=Path, default=None)
    args = parser.parse_args(argv)
    dataset = args.dataset
    if args.project:
        from .project import ScenePasteProject
        project = ScenePasteProject.load(args.project)
        dataset = dataset or project.output_dir or project.validation_dataset
    try:
        from compose_app_qt.data_loop import launch_data_loop_center
    except ImportError as exc:
        print(f"Data Loop Center 启动失败：{exc}\n请安装：python -m pip install 'scenepaste[gui-qt]'", file=sys.stderr)
        return 1
    return int(launch_data_loop_center(dataset_root=dataset, output_root=(Path(dataset)/"shards") if dataset else None) or 0)



def factory_main(argv: Optional[Sequence[str]] = None) -> int:
    """VisionDataForge V9 utilities that do not require the Qt desktop app."""
    parser = argparse.ArgumentParser(prog="scenepaste factory", description="Label-first placement planning tools")
    sub = parser.add_subparsers(dest="action", required=True)
    plans = sub.add_parser("plan", help="emit deterministic label-first plans without rendering images")
    plans.add_argument("--project", type=Path, default=None)
    plans.add_argument("--class-map", default="person=0,vehicle=1")
    plans.add_argument("--count", type=int, default=10)
    plans.add_argument("--min-objects", type=int, default=1)
    plans.add_argument("--max-objects", type=int, default=3)
    plans.add_argument("--seed", type=int, default=42)
    plans.add_argument("--y-min", type=float, default=0.35)
    plans.add_argument("--y-max", type=float, default=0.95)
    plans.add_argument("--far-height", type=float, default=0.08)
    plans.add_argument("--near-height", type=float, default=0.32)
    plans.add_argument("--hardcase-recipe", default=None)
    plans.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args(argv)
    from types import SimpleNamespace
    from .core.config import parse_class_map
    from .core.planning import load_hardcase_recipe, placement_to_dict, plan_label_first
    class_map = parse_class_map(args.class_map)
    if args.project:
        from .project import ScenePasteProject
        project = ScenePasteProject.load(args.project)
        if project.class_map:
            class_map = project.class_map
    cfg = SimpleNamespace(
        class_map=class_map, min_objects=args.min_objects, max_objects=args.max_objects,
        y_min=args.y_min, y_max=args.y_max, far_height=args.far_height, near_height=args.near_height,
        flip_prob=0.5, empty_scene_probability=0.0, profile_strength=0.0,
    )
    import random
    hard = load_hardcase_recipe(args.hardcase_recipe)
    rows = []
    for idx in range(args.count):
        rng = random.Random((args.seed * 1000003 + idx * 9176 + 0x5EED123) & 0xFFFFFFFF)
        placements = plan_label_first(cfg, rng, hardcase_recipe=hard)
        rows.append({"index": idx, "objects": [placement_to_dict(x) for x in placements]})
    payload = {"schema": "scenepaste/label-first-plan", "version": 9, "seed": args.seed, "samples": rows}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(args.output)
    else:
        print(text)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Top-level ``scenepaste`` dispatcher."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        _print_help()
        return 0
    if args[0] in {"-V", "--version", "version"}:
        print(f"ScenePaste {__version__}")
        return 0

    command, rest = args[0], args[1:]
    if command == "gui":
        # `scenepaste gui` launches the PySide6 UI. Historical `--qt` is a
        # no-op and may appear anywhere among the gui args.
        rest = [a for a in rest if a != "--qt"]
        try:
            from compose_app_qt.main_entry import main_entry as qt_entry
        except ImportError as exc:
            print(
                f"GUI 启动失败：{exc}\n"
                "请安装 Qt 依赖：python -m pip install 'scenepaste[gui-qt]'",
                file=sys.stderr,
            )
            return 1
        return int(qt_entry(rest) or 0)
    if command == "generate":
        return generate_main(rest)
    if command == "analyze":
        from .tools.analyze import main as analyze_main
        return int(analyze_main(rest) or 0)
    if command == "split":
        from .tools.split import main as split_main
        return int(split_main(rest) or 0)
    if command == "merge":
        from .tools.merge import main as merge_main
        return int(merge_main(rest) or 0)
    if command == "explore":
        return explore_main(rest)
    if command == "profile":
        return profile_main(rest)
    if command == "qa":
        return qa_main(rest)
    if command == "recipe":
        return recipe_main(rest)
    if command == "curate":
        return curate_main(rest)
    if command == "compare":
        return compare_main(rest)
    if command == "shard":
        return shard_main(rest)
    if command == "project":
        return project_main(rest)
    if command == "loop":
        return loop_main(rest)
    if command == "factory":
        return factory_main(rest)

    print(f"Unknown ScenePaste command: {command}\n", file=sys.stderr)
    _print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
