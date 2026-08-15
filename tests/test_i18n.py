"""i18n behaviour: locale switching and message formatting."""

from __future__ import annotations

import importlib

import pytest


def _reload_errors():
    import scenepaste.errors as errors
    return importlib.reload(errors)


def test_default_locale_is_chinese(monkeypatch):
    monkeypatch.delenv("SCENEPASTE_LANG", raising=False)
    errors = _reload_errors()
    assert errors._detect_locale() == "zh"


def test_english_locale(monkeypatch):
    monkeypatch.setenv("SCENEPASTE_LANG", "en")
    errors = _reload_errors()
    assert errors._detect_locale() == "en"


def test_t_returns_formatted_message(monkeypatch):
    monkeypatch.setenv("SCENEPASTE_LANG", "en")
    errors = _reload_errors()
    assert errors.t("class_map.bad_format", item="foo") == \
        "Invalid class-map entry: foo (expected 'name=id')"


def test_t_falls_back_to_zh_for_unknown_locale(monkeypatch):
    monkeypatch.setenv("SCENEPASTE_LANG", "fr")
    errors = _reload_errors()
    # Unknown locale falls back to default (zh).
    assert "类别" in errors.t("class_map.need_one")


def test_t_returns_key_for_unknown_message(monkeypatch):
    monkeypatch.delenv("SCENEPASTE_LANG", raising=False)
    errors = _reload_errors()
    assert errors.t("no.such.key") == "no.such.key"


def test_config_error_uses_catalog(monkeypatch):
    """parse_class_map error respects the active locale."""
    monkeypatch.setenv("SCENEPASTE_LANG", "en")
    # Re-import config so it picks up the reloaded errors module.
    import scenepaste.core.config as config
    config = importlib.reload(config)
    with pytest.raises(ValueError) as exc_info:
        config.parse_class_map("no_equals_sign")
    assert "Invalid class-map entry" in str(exc_info.value)
