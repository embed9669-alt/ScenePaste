import random
from pathlib import Path

from scenepaste.core.templates import (
    load_template_data, parameterize_payload, sample_template,
)

ROOT = Path(__file__).resolve().parents[1]


def test_v1_template_upgrades_and_samples_ranges():
    data = load_template_data(ROOT / "samples" / "templates" / "mixed_traffic_scene.json")
    assert data["version"] == 2
    varied = parameterize_payload(data, position_jitter_x=.05, position_jitter_y=.04,
                                  scale_jitter=.2, angle_jitter=10, same_class_random=True)
    a = sample_template(varied, random.Random(10), {"person":0,"truck":1,"motorcycle":2})
    b = sample_template(varied, random.Random(11), {"person":0,"truck":1,"motorcycle":2})
    assert len(a) == 3 and len(b) == 3
    assert any(abs(x.cx_ratio-y.cx_ratio) > 1e-6 for x,y in zip(a,b))
    assert all(x.same_class_random for x in a)


def test_instance_probability_can_disable_all():
    data = load_template_data(ROOT / "samples" / "templates" / "distant_person.json")
    varied = parameterize_payload(data, instance_probability=0.0)
    assert sample_template(varied, random.Random(1), {"person":0}) == []
