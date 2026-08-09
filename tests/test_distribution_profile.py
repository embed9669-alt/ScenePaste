from pathlib import Path
import json

from scenepaste import GenerationConfig, generate_dataset, parse_class_map
from scenepaste.core.distribution import DistributionProfile, learn_distribution_profile

ROOT = Path(__file__).resolve().parents[1]


def test_learn_labelme_distribution_profile(tmp_path):
    profile = learn_distribution_profile(ROOT / "samples" / "objects", bins=10)
    assert profile.data["schema"] == "scenepaste/distribution-profile"
    assert profile.data["image_count"] == 3
    assert set(profile.classes) >= {"person", "truck", "motorcycle"}
    out = tmp_path / "profile.json"
    profile.save(out)
    loaded = DistributionProfile.load(out)
    assert loaded.data["object_count_total"] == 3
    assert sum(loaded.classes["person"]["height"]["counts"]) == 1


def test_profile_driven_generation_persists_target_profile(tmp_path):
    profile_path = tmp_path / "source_profile.json"
    learn_distribution_profile(ROOT / "samples" / "objects").save(profile_path)
    out = tmp_path / "generated"
    cfg = GenerationConfig(
        objects_dir=ROOT / "samples" / "objects",
        backgrounds_dir=ROOT / "samples" / "backgrounds",
        output_dir=out,
        class_map=parse_class_map("person=0"), count=4, seed=7,
        distribution_profile=profile_path, profile_strength=1.0,
        workers=1, run_id="profile_run", save_previews=False,
    )
    summary = generate_dataset(cfg)
    assert summary["generated_images"] == 4
    target = json.loads((out / "target_distribution_profile.json").read_text(encoding="utf-8"))
    assert set(target["classes"]) == {"person"}
