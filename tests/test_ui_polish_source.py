from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_main_window_promotes_asset_studio_mainline():
    text = (ROOT / "compose_app_qt" / "app.py").read_text(encoding="utf-8")
    assert "Controllable Vision Data Factory" in text
    assert 'QPushButton("素材工作室")' in text
    assert 'QPushButton("加载目标素材…")' in text
    assert "手工场景编辑" not in text


def test_factory_ui_keeps_copy_paste_dashboard():
    text = (ROOT / "compose_app_qt" / "large_generate.py").read_text(encoding="utf-8")
    for phrase in [
        "可控 Copy-Paste",
        "困难样本策略",
        "可放置区域",
        "最近生成结果",
        "详细日志",
    ]:
        assert phrase in text
    assert "MetricCard" in text
    assert "CollapsibleSection" in text
    for phrase in [
        "生成方案",
        "_build_factory_tab",
        "_load_final_diagnostics",
        "AI 轻量修图",
        "Qwen 高质量修图",
        "Hugging Face 官方",
        "qwen_process",
        "qwen_download_process",
        "--generation-mode",
        "--generator-backend",
        "--qwen-",
        "--label-qa-",
        "--label-refine-mode",
        "模型回退",
        "QA 拒绝",
    ]:
        assert phrase not in text


def test_theme_has_polished_cards_for_both_modes():
    text = (ROOT / "compose_app_qt" / "theme.py").read_text(encoding="utf-8")
    assert "DARK_POLISH_QSS" in text
    assert "LIGHT_POLISH_QSS" in text
    assert "metricCard" in text
    assert "previewSurface" in text


def test_v98_home_menu_and_persistence_are_productized():
    app = (ROOT / "compose_app_qt" / "app.py").read_text(encoding="utf-8")
    for phrase in [
        "打开素材工作室", "最近工程", "快速开始",
        "GitHub 项目主页", "报告问题", "恢复默认布局", "QSettings",
        'mb.addMenu("素材")', 'mb.addMenu("场景")', 'mb.addMenu("合成")',
    ]:
        assert phrase in app
    assert 'mb.addMenu("生成")' not in app
    assert "模型下载说明" not in app
    assert 'self._right_tabs.addTab(self.gen_defaults, "批量默认")' in app
    assert "整目录 rembg 加载" in app


def test_v98_factory_has_trial_run_and_copy_paste_controls():
    text = (ROOT / "compose_app_qt" / "large_generate.py").read_text(encoding="utf-8")
    assert 'QPushButton("试生成 10 张")' in text
    assert 'QPushButton("开始正式生成")' in text
    assert 'tabs.addTab(adv_scroll, "运行与性能")' in text
    assert '"_trials"' in text
    assert "hardcase_recipe" in text
    assert "scene_region_mode" in text
    assert "生成式编辑" not in text


def test_pipeline_is_copy_paste_only():
    text = (ROOT / "scenepaste" / "core" / "advanced_pipeline.py").read_text(encoding="utf-8")
    assert "create_generative_backend" not in text
    assert "generation_mode" not in text
    assert "generator_fallback_count" not in text
