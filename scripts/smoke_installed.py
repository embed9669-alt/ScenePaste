#!/usr/bin/env python3
"""Smoke-test an *installed* ScenePaste wheel from outside the source tree.

This intentionally exercises the bundled package samples, a tiny all-format
run, Explorer rendering, QA, WebDataset sharding and (when PySide6 is
available) the offscreen GUI sample/template/Explorer path.
"""
from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path


def _core_smoke(work: Path) -> Path:
    import scenepaste
    from scenepaste import GenerationConfig, generate_dataset
    from scenepaste.explorer import index_dataset, render_dataset_image
    from scenepaste.sample_data import bundled_samples_root
    from scenepaste.tools.qa import write_qa_dashboard
    from scenepaste.tools.shard import build_webdataset_shards

    assert scenepaste.__version__ == "10.0.0", scenepaste.__version__
    samples = bundled_samples_root()
    assert (samples / "objects").is_dir()
    assert (samples / "backgrounds").is_dir()
    assert (samples / "templates").is_dir()

    output = work / "generated"
    summary = generate_dataset(GenerationConfig(
        objects_dir=samples / "objects",
        backgrounds_dir=samples / "backgrounds",
        output_dir=output,
        class_map={"person": 0, "motorcycle": 1, "truck": 2},
        count=4,
        min_objects=1,
        max_objects=2,
        output_format="all",
        workers=1,
        preview_ratio=0.0,
        save_previews=False,
        seed=1001,
        run_id="wheel-smoke",
    ))
    assert int(summary.get("generated_images", summary.get("saved_images", summary.get("saved", 0)))) == 4, summary

    items = index_dataset(output)
    assert len(items) == 4, len(items)
    overlay = render_dataset_image(output, items[0])
    assert overlay.image.width > 0 and overlay.image.height > 0

    qa = write_qa_dashboard(output, html_path=work / "qa.html", json_path=work / "qa.json",
                            duplicate_limit=100, embedding_limit=0)
    assert qa.get("health") in {"ok", "warning"}, qa.get("health")
    shards = build_webdataset_shards(output, work / "shards", split="train", max_samples=2)
    assert shards.get("samples") == 4, shards
    assert len(shards.get("shards", [])) == 2, shards
    return output


def _qt_smoke(work: Path, output: Path, require_qt: bool) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except Exception:
        if require_qt:
            raise
        print("Qt smoke skipped: PySide6 is not installed")
        return

    from compose_app.models import Instance
    from compose_app_qt.app import MainWindow
    from compose_app_qt.explorer import DatasetExplorerWindow
    from compose_app_qt.templates import save_template

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.load_sample_dataset()
    deadline = time.monotonic() + 15.0
    while window._worker is not None and window._worker.isRunning() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert window.doc.cutouts, "GUI sample cutouts were not loaded"
    assert window.doc.background_paths, "GUI sample backgrounds were not loaded"
    assert window.doc.bg_size[0] > 0 and window.doc.bg_size[1] > 0, window.doc.bg_size

    bw, bh = window.doc.bg_size
    window.doc.add_instance(Instance(
        cutout_index=0, cx=bw * 0.5, cy=bh * 0.65, h_ratio=0.22,
        flip=False, angle=0.0, uid=window.doc.next_uid(),
    ))
    app.processEvents()
    assert window.doc.instances, "GUI smoke instance was not created"
    template_path = work / "smoke_template.json"
    assert save_template(window.doc, template_path) >= 1
    assert template_path.is_file()

    explorer = DatasetExplorerWindow(output)
    assert explorer.items, "Dataset Explorer did not index generated samples"
    explorer.close()
    window.close()
    app.processEvents()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-qt", action="store_true", help="fail when PySide6/Qt smoke cannot run")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="scenepaste-installed-smoke-") as tmp:
        work = Path(tmp)
        output = _core_smoke(work)
        _qt_smoke(work, output, args.require_qt)
    print("ScenePaste installed-wheel smoke: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
