"""models 层测试。"""

from __future__ import annotations

from PIL import Image

from compose_app.models import Cutout, Instance


def _make_rgba(w=40, h=60):
    return Image.new("RGBA", (w, h), (255, 0, 0, 255))


def test_instance_clone_preserves_uid():
    """clone() preserves uid (so undo snapshots are stable); callers that
    need a fresh uid (duplicate) assign it explicitly."""
    inst = Instance(cutout_index=2, cx=100.0, cy=200.0, h_ratio=0.5, uid=42)
    clone = inst.clone()
    assert clone.cx == 100.0
    assert clone.cy == 200.0
    assert clone.cutout_index == 2
    assert clone.uid == 42  # clone 保留 uid；新场景由调用方赋值


def test_instance_render_key_depends_on_size_and_transform():
    inst = Instance(cutout_index=0, cx=0, cy=0, h_ratio=0.5, flip=False, angle=0.0)
    k1 = inst.render_key(100)
    k2 = inst.render_key(100)
    assert k1 == k2
    inst.flip = True
    assert inst.render_key(100) != k1
    inst.flip = False
    inst.angle = 5.0
    assert inst.render_key(100) != k1


def test_instance_get_rendered_uses_cache():
    rgba = _make_rgba()
    inst = Instance(cutout_index=0, cx=0, cy=0, h_ratio=0.5, flip=False, angle=0.0)
    out1 = inst.get_rendered(rgba, 60)
    # 同样的 key 再取，应该复用同一对象
    out2 = inst.get_rendered(rgba, 60)
    assert out1 is out2


def test_instance_invalidate_cache_forces_rerender():
    rgba = _make_rgba()
    inst = Instance(cutout_index=0, cx=0, cy=0, h_ratio=0.5, flip=False, angle=0.0)
    out1 = inst.get_rendered(rgba, 60)
    inst.invalidate_cache()
    out2 = inst.get_rendered(rgba, 60)
    assert out1 is not out2


def test_cutout_dataclass_fields():
    rgba = _make_rgba()
    c = Cutout(label="truck", class_id=2, source="x.json#1", rgba=rgba, thumb=None)
    assert c.label == "truck"
    assert c.class_id == 2
    assert c.source == "x.json#1"
