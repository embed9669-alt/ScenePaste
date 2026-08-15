from pathlib import Path


def test_acknowledgements_and_third_party_notices_are_present():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    readme_zh = (root / "README_zh.md").read_text(encoding="utf-8")
    notices = (root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    inspiration = (root / "docs" / "INSPIRATION.md").read_text(encoding="utf-8")

    assert "Acknowledgements & inspiration" in readme
    assert "致谢与设计参考" in readme_zh
    for name in (
        "NVIDIA Physical AI Data Factory",
        "SAM 2",
        "LabelMe",
        "X-AnyLabeling",
        "rembg",
        "WebDataset",
    ):
        assert name in readme
        assert name in notices
    # Generative stacks were removed in V10; keep historical license notes only.
    assert "Qwen-Image" in notices
    assert "Stable Diffusion" in notices
    assert "removed" in notices.lower() or "no longer" in notices.lower()
    assert "not a fork" in readme.lower()
    assert "does not copy" in notices
    assert "GPL-3.0" in notices
    assert "THIRD_PARTY_NOTICES.md" in inspiration
