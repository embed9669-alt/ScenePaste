from pathlib import Path
import sqlite3

from scenepaste import GenerationConfig, generate_dataset, parse_class_map

ROOT = Path(__file__).resolve().parents[1]


def _cfg(out, workers=1, resume=False):
    return GenerationConfig(
        objects_dir=ROOT/"samples"/"objects", backgrounds_dir=ROOT/"samples"/"backgrounds",
        output_dir=out, class_map=parse_class_map("person=0"), count=6, seed=42,
        workers=workers, resume=resume, run_id="resume_case", save_previews=False,
    )


def test_resume_only_regenerates_missing_indices(tmp_path):
    out = tmp_path / "ds"
    first = generate_dataset(_cfg(out))
    assert first["generated_images"] == 6
    db = out / ".scenepaste" / "runs" / "resume_case.sqlite3"
    con = sqlite3.connect(str(db))
    for idx in (4,5):
        con.execute("DELETE FROM tasks WHERE idx=?", (idx,))
        (out/"images"/"train"/f"resume_case_42_{idx:06d}.jpg").unlink()
        (out/"labels"/"train"/f"resume_case_42_{idx:06d}.txt").unlink()
        (out/".scenepaste"/"fragments"/"resume_case"/"meta"/f"{idx:09d}.json").unlink()
    con.execute("UPDATE metadata SET value='\"interrupted\"' WHERE key='status'")
    con.commit(); con.close()
    resumed = generate_dataset(_cfg(out, resume=True))
    assert resumed["generated_images"] == 6
    assert resumed["resumed_images"] == 4
    assert len(list((out/"images"/"train").glob("*.jpg"))) == 6
