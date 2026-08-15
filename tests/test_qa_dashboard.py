from pathlib import Path

from scenepaste import GenerationConfig, generate_dataset, parse_class_map
from scenepaste.tools.qa import build_qa_report, write_qa_dashboard

ROOT = Path(__file__).resolve().parents[1]


def test_qa_dashboard_is_standalone_and_reports_integrity(tmp_path):
    out = tmp_path / "ds"
    generate_dataset(GenerationConfig(
        objects_dir=ROOT/"samples"/"objects", backgrounds_dir=ROOT/"samples"/"backgrounds",
        output_dir=out, class_map=parse_class_map("person=0"), count=3, run_id="qa",
        save_previews=False,
    ))
    report = build_qa_report(out, duplicate_limit=100)
    assert report["summary"]["images"] == 3
    assert report["integrity"]["unreadable_images"] == 0
    final = write_qa_dashboard(out)
    html = Path(final["html_path"]).read_text(encoding="utf-8")
    assert "ScenePaste QA Dashboard" in html
    assert (out/"qa_report.json").is_file()


def test_qa_dashboard_renders_crowding_and_visibility_columns(tmp_path):
    out = tmp_path / "ds"
    generate_dataset(GenerationConfig(
        objects_dir=ROOT / "samples" / "objects",
        backgrounds_dir=ROOT / "samples" / "backgrounds",
        output_dir=out,
        class_map=parse_class_map("person=0"),
        count=2,
        run_id="qa_columns",
        output_format="all",
        save_previews=False,
    ))
    final = write_qa_dashboard(out, embedding_limit=2)
    html = Path(final["html_path"]).read_text(encoding="utf-8")
    assert "Overlap IoU" in html
    assert "Visible shape" in html
    assert "cv-lite-v1 nearest-neighbor distance" in html
    assert "Generation diagnostics" in html
