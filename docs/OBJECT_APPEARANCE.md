# Object Appearance Recipes

Object Appearance Recipes diversify **each pasted cutout before color matching and blending**. They are separate from scene recipes (`AUGMENTATION_RECIPES.md`), which run on the finished image after geometry is frozen.

```text
Cutout
  → resize / rotate / flip
  → Object Appearance Recipe
  → alpha-weighted Color Match (optional)
  → Blend
  → Scene Recipe
```

The object recipe changes RGB appearance only. Placement, alpha geometry and annotation coordinates remain unchanged, so Detect / Seg / OBB / Semantic / COCO stay aligned.

## Built-ins

```bash
scenepaste recipe --kind object list
scenepaste recipe --kind object show mild
```

| Name | Role |
|---|---|
| `off` | No object photometric change. Color-match may still run. |
| `legacy` | Explicit v1.0-compatible HSV jitter + `blur_prob`-gated blur. |
| `mild` | Recommended starting point for safe object diversity. |
| `surveillance-object` | Stronger CCTV-like degradation: blur, motion blur, noise, JPEG and low-resolution loss. |

CLI and GUI default to **unset** for backwards compatibility with v1.0 deterministic behavior. Opt into the new system explicitly:

```bash
scenepaste generate ... --object-appearance-recipe mild
```

## Effects

| Effect | Main fields | Notes |
|---|---|---|
| `brightness_contrast` | `brightness`, `contrast` | Brightness is a fraction of 255; contrast is multiplicative. |
| `saturation` | `range` | Multiplicative HSV saturation. |
| `hue` | `range` | **Degrees**, e.g. `[-5, 5]`. |
| `gamma` | `range` | Positive gamma values. |
| `color_temperature` | `strength` | Perceptual warm/cool strength, not a literal Kelvin delta. |
| `gaussian_blur` | `sigma` | Alpha-aware; does not drag mask-exterior RGB into target edges. |
| `motion_blur` | `ksize` | Alpha-aware directional blur. |
| `gaussian_noise` | `sigma` | RGB sensor-like noise. |
| `resolution_degrade` | `scale` | Downsample + restore RGB detail while preserving alpha geometry. |
| `jpeg` | `quality` | Object-local JPEG artifacts using edge-safe exterior fill. |
| `sharpness` | `amount` | `1.0` is neutral; above/below adjusts local sharpness. |
| `hsv_jitter` | `hue`, `saturation`, `value` | Compatibility effect used by `legacy`; `hue` here remains historical OpenCV hue units. |

Every probability `p` is `0..1`. ScenePaste v1.1 validates effect names, field names and safe numeric ranges before a generation run. Typos such as `brightnes_contrast` fail early instead of silently disabling an intended augmentation.

## Custom recipe + class overrides

Export a built-in recipe and edit JSON:

```bash
scenepaste recipe --kind object export mild -o my_object.json
```

```json
{
  "schema": "scenepaste/object-appearance-recipe",
  "version": 1,
  "name": "industrial-objects",
  "effects": {
    "brightness_contrast": {"p": 0.8, "brightness": [-0.12, 0.12], "contrast": [0.88, 1.12]},
    "saturation": {"p": 0.6, "range": [0.8, 1.2]},
    "hue": {"p": 0.4, "range": [-5, 5]},
    "gamma": {"p": 0.3, "range": [0.88, 1.18]},
    "color_temperature": {"p": 0.3, "strength": [-12, 12]},
    "gaussian_blur": {"p": 0.25, "sigma": [0.2, 1.0]},
    "gaussian_noise": {"p": 0.2, "sigma": [1.0, 5.0]},
    "resolution_degrade": {"p": 0.4, "scale": [0.35, 0.9]},
    "sharpness": {"p": 0.15, "amount": [0.85, 1.15]}
  },
  "by_class": {
    "truck": {
      "effects": {
        "saturation": {"p": 0.85, "range": [0.7, 1.3]},
        "hue": {"p": 0.55, "range": [-8, 8]}
      }
    },
    "person": {
      "effects": {
        "hue": {"p": 0.20, "range": [-3, 3]},
        "resolution_degrade": {"p": 0.5, "scale": [0.3, 0.85]}
      }
    }
  }
}
```

`by_class` overrides individual effect keys on top of the base `effects`. Use conservative hue ranges for classes where color is semantically or physically constrained.

## Why alpha-aware filtering matters

A cutout crop can still contain original-background RGB outside its alpha mask. A naive Gaussian blur or downsample can pull those pixels back into the object boundary and create dark/colored halos. ScenePaste v1.1 performs neighborhood-based object effects in alpha-aware form so exterior crop pixels do not contaminate the pasted object.

Color matching is also computed only over the **alpha-weighted object footprint**, rather than averaging the entire bounding rectangle.

## Metadata and QA

Each generated sample fragment records per-instance `object_effects`. Run diagnostics aggregate `object_effect_counts` alongside `scene_effect_counts`.

The QA Dashboard renders both groups as coverage tables so you can answer questions such as:

- what percentage of generated objects received low-resolution degradation?
- how often was motion blur applied?
- did a class-specific recipe actually activate?
- how much full-scene camera degradation was used?

## Project manifests

A project can store a built-in recipe name or a custom JSON path in `defaults.object_appearance_recipe`. Relative custom JSON paths are resolved from the directory containing `scenepaste.project.json`, so portable projects work from a different current working directory.

## Recommended usage

Start with `mild`, inspect the QA Dashboard and a held-out real validation set, then adjust. Strong appearance augmentation can reduce domain gap, but unrealistic colors or excessive degradation can also hurt training.

## Out of scope

- region-aware recolor (vehicle paint / clothing submasks);
- generative AI editing or inpainting;
- changing object pose or geometry.

Those are deliberately kept out of the deterministic lightweight core for now.
