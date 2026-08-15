# Contributing

Contributions are welcome. ScenePaste prioritizes **annotation correctness, reproducibility, practical UI workflows and dataset quality** over adding formats for their own sake.

## Development setup

```bash
git clone https://github.com/embed9669-alt/ScenePaste.git
cd ScenePaste
python -m pip install -e ".[dev,gui-qt]"
export QT_QPA_PLATFORM=offscreen   # for headless GUI tests
pytest -q
ruff check scenepaste compose_app compose_app_qt tests
```

Optional extras:

- `.[auto]` — rembg auto-cutout
- `.[gui-qt]` — PySide6 desktop editor (required for `scenepaste gui`)

If an older editable install named `copy-paste-tool` is present, uninstall it first so it cannot shadow `compose_app`:

```bash
python -m pip uninstall copy-paste-tool
```

All tests must pass. Exact counts change as features land.

## Good contributions

- bug fixes with regression tests
- GUI ergonomics that keep canvas pose identical to exported labels
- performance improvements backed by measurements
- dataset QA / validation improvements
- scene-template and controlled-generation workflows
- output-format fixes that match the target framework specification
- documentation and examples (update **both** `README.md` and `README_zh.md` for user-facing changes)
- optional auto-cutout backends that stay out of core dependencies

## Architecture rules

- keep `compose_app.models` / `rendering` / `segmentation` / `io_utils` testable without launching Qt
- put window wiring in `compose_app_qt/app.py`; document state in `compose_app_qt/state.py`; panels stay presentation-only
- bake flip/rotation into rendered RGBA once — do not also apply Qt item transforms
- derive segmentation / COCO / semantic / OBB annotations from the **final visible mask** whenever possible
- do not silently change class IDs or dataset split semantics
- keep heavy ML dependencies optional (`[auto]`, future SAM extras)

## Pull requests

Keep each PR focused. Add tests for changed behavior, then run:

```bash
pytest -q
python -m compileall -q scenepaste compose_app compose_app_qt .
ruff check scenepaste compose_app compose_app_qt tests
```

## UI screenshots

Refresh README images after meaningful GUI changes:

```bash
DISPLAY=:1 python scripts/capture_ui_screenshots.py
```

Writes `docs/images/ui_overview.png`, `ui_overview_light.png`, and `ui_settings.png`.

## Community

Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Security reports go through [SECURITY.md](SECURITY.md), not public issues.

## Reporting issues

Use the GitHub issue templates. Include ScenePaste version (`scenepaste --version`), OS, Python version, and a minimal command or sample that reproduces the problem.
