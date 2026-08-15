"""Tests for selectable cutout models (no network / heavy weights)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scenepaste.core import cutout_models as cm


@pytest.fixture(autouse=True)
def _reset_models():
    cm.reset_loaded_models()
    yield
    cm.reset_loaded_models()


def test_list_cutout_models_includes_rembg_and_grounding():
    ids = [m.id for m in cm.list_cutout_models()]
    assert "rembg:u2net" in ids
    assert "sam2_click" in ids
    assert "grounding_sam2" in ids
    g = cm.get_model_spec("grounding_sam2")
    assert g.needs_text_prompt
    assert g.family == "grounding_sam2"
    s = cm.get_model_spec("sam2_click")
    assert s.family == "sam2_point"


def test_get_model_spec_unknown_raises():
    with pytest.raises(KeyError):
        cm.get_model_spec("nope")


def test_ensure_rembg_missing_raises_clear_error():
    with patch.dict("sys.modules", {"rembg": None}):
        with patch("builtins.__import__", side_effect=ImportError("no rembg")):
            # Direct path: call _ensure_rembg after forcing import fail inside ensure
            pass
    with patch.object(cm, "_ensure_rembg", side_effect=ImportError("需要 rembg")):
        with pytest.raises(ImportError, match="rembg"):
            cm.ensure_model("rembg:u2net")


def test_ensure_rembg_loads_session():
    fake_session = object()

    def _fake_new_session(name):
        assert name == "u2net"
        return fake_session

    fake_rembg = MagicMock()
    fake_rembg.new_session = _fake_new_session
    with patch.dict("sys.modules", {"rembg": fake_rembg}):
        msgs = []
        cm.ensure_model("rembg:u2net", progress=msgs.append)
        assert cm.is_model_ready("rembg:u2net")
        assert any("u2net" in m or "就绪" in m for m in msgs)


def test_tqdm_progress_callback_reports_percent():
    pytest.importorskip("tqdm")
    events = []

    def cb(msg, pct=None):
        events.append((msg, pct))

    T = cm._tqdm_with_callback(cb, "rembg isnet")
    with open(os.devnull, "w") as sink:
        bar = T(total=1_000_000, file=sink)
        bar.update(250_000)
        bar.update(250_000)
        bar.close()
    assert events
    assert any(p is not None and p >= 25 for _, p in events)
    assert any("MB" in m or "KB" in m or "/" in m for m, _ in events)


def test_patch_tqdm_progress_swaps_pooch_reference():
    pytest.importorskip("tqdm")
    pytest.importorskip("pooch")
    events = []

    def cb(msg, pct=None):
        events.append((msg, pct))

    import pooch.downloaders as pooch_dl

    original = pooch_dl.tqdm
    with cm._patch_tqdm_progress(cb, "test"):
        assert pooch_dl.tqdm is not original
        with open(os.devnull, "w") as sink:
            bar = pooch_dl.tqdm(total=100, file=sink)
            bar.update(40)
            bar.close()
    assert pooch_dl.tqdm is original
    assert any(p is not None and abs(p - 40) < 1 for _, p in events)


def test_predict_rembg_uses_loaded_session():
    rgb_out = np.zeros((20, 30, 4), dtype=np.uint8)
    rgb_out[5:15, 5:25, 3] = 255
    rgb_out[5:15, 5:25, :3] = 100

    fake_session = object()

    def _remove(rgb, session=None):
        assert session is fake_session
        return rgb_out

    fake_rembg = MagicMock()
    fake_rembg.new_session = MagicMock(return_value=fake_session)
    fake_rembg.remove = _remove
    with patch.dict("sys.modules", {"rembg": fake_rembg}):
        cm.ensure_model("rembg:u2net")
        bgr = np.zeros((20, 30, 3), dtype=np.uint8)
        poly, alpha = cm.predict_cutout(bgr, "rembg:u2net")
        assert alpha is not None
        assert alpha.shape == (20, 30)
        assert poly is not None
        assert len(poly) >= 3


def test_normalize_prompt():
    assert cm._normalize_prompt("Person") == "person."
    assert cm._normalize_prompt("car.") == "car."
    assert cm._normalize_prompt("") == "object."


def test_normalize_socks_proxy_and_mirror(monkeypatch):
    monkeypatch.setenv("ALL_PROXY", "socks://127.0.0.1:7897/")
    monkeypatch.setenv("HTTPS_PROXY", "socks://127.0.0.1:1080")
    cm.configure_hf_download(use_mirror=True)
    # Mirror downloads bypass socks proxies that break urllib3
    assert "socks" not in (os.environ.get("ALL_PROXY") or "").lower()
    assert "socks" not in (os.environ.get("HTTPS_PROXY") or "").lower()
    assert os.environ.get("HF_ENDPOINT", "").startswith("https://hf-mirror.com")


def test_ensure_grounding_missing_transformers():
    real_import = __import__

    def _import(name, *args, **kwargs):
        if name == "transformers" or name.startswith("transformers."):
            raise ImportError("no transformers")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_import):
        with pytest.raises(ImportError, match="transformers|GroundingSAM2|grounding"):
            cm.ensure_model("grounding_sam2")
