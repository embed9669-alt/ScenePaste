# Augmentation Recipes

ScenePaste can apply deterministic **image-only** domain augmentation after the scene geometry has been finalized. This is intentionally different from spatial augmentation: boxes, polygons, OBBs and semantic masks do not move, so all exported annotations remain aligned.

## Built-in recipes

```bash
scenepaste recipe list
scenepaste recipe show camera-mild
```

Built-ins:

- `clean` — no post-render effect;
- `camera-mild` — light brightness/contrast, gamma, noise, compression and occasional resolution loss;
- `surveillance` — stronger compression/noise/downscale/motion blur for CCTV-like imagery;
- `low-light` — darker exposure, gamma, noise and vignetting.

Use one during generation:

```bash
scenepaste generate ... \
  --augmentation-recipe surveillance \
  --blend-mode gaussian \
  --blend-sigma 1.5
```

## Custom recipe

Export a built-in recipe and edit it:

```bash
scenepaste recipe export camera-mild -o my_camera.json
scenepaste generate ... --augmentation-recipe my_camera.json
```

Example schema:

```json
{
  "schema": "scenepaste/augmentation-recipe",
  "version": 1,
  "name": "industrial-camera",
  "effects": {
    "brightness_contrast": {"p": 0.4, "brightness": [-0.08, 0.05], "contrast": [0.9, 1.1]},
    "gamma": {"p": 0.2, "range": [0.9, 1.12]},
    "gaussian_noise": {"p": 0.3, "sigma": [1.5, 6.0]},
    "motion_blur": {"p": 0.15, "ksize": [3, 7]},
    "downscale": {"p": 0.15, "scale": [0.75, 0.95]},
    "jpeg": {"p": 0.3, "quality": [75, 95]},
    "vignette": {"p": 0.1, "strength": [0.05, 0.2]}
  }
}
```

The exact effects applied to each generated sample are persisted in the crash-safe per-sample metadata fragments, so a run remains auditable.

## Blend modes

Foreground/background edge blending is controlled independently:

- `alpha` — preserve the asset alpha (default);
- `hard` — threshold the alpha for paper-style hard Copy-Paste;
- `gaussian` — blur alpha near boundaries to reduce cutout seams.

Do not assume a stronger recipe is always better. The recipe should model plausible camera/domain variation for the deployment environment.
