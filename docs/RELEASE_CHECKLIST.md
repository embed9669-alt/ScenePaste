# ScenePaste release checklist

Use this checklist for v1.x releases. A feature-complete source tree is not enough: the installed wheel must work outside the repository.

## Metadata and documentation

- [ ] `pyproject.toml`, `scenepaste.__version__`, `compose_app.__version__`, README and CHANGELOG use the same version.
- [ ] Development Status classifier matches the release stage.
- [ ] README claims correspond to visible UI behavior.
- [ ] Declared Python versions are present in CI.
- [ ] Markdown internal links resolve.

## Automated verification

- [ ] `python -m pytest -q`
- [ ] `ruff check scenepaste compose_app compose_app_qt tests scripts`
- [ ] engine mypy job passes
- [ ] Windows / Linux / macOS Qt-offscreen matrix is green

## Installed-wheel smoke

Build a wheel and install it **non-editably**. Run from a directory outside the source repository:

1. `scenepaste --version`
2. resolve package-bundled samples
3. launch the Qt editor offscreen
4. click/call **★ Load Sample**
5. save a scene template
6. generate a tiny `--output-format all` dataset
7. render one Dataset Explorer overlay
8. build QA JSON/HTML
9. build small WebDataset shards

CI performs this flow through `scripts/smoke_installed.py`.

## Release artifacts

- [ ] `python scripts/build_release.py`
- [ ] source ZIP has no `.git`, caches, egg-info, runtime state or user data
- [ ] wheel contains `scenepaste/resources/samples/`
- [ ] wheel installs and imports from site-packages, not the source checkout
- [ ] release SHA-256 recorded

## Performance sanity

Run:

```bash
python scripts/benchmark_generation.py --count 1000 --workers 1 4 8
```

Record the environment and keep the result honest even when more workers are slower. See [PERFORMANCE.md](PERFORMANCE.md).
