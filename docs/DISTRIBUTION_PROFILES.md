# Real-data distribution profiles

ScenePaste can learn a compact 2D distribution profile from an existing annotated dataset and use that profile to plan synthetic scenes.

## Supported sources

`scenepaste profile learn` accepts:

- LabelMe / X-AnyLabeling JSON + images;
- YOLO Detect;
- YOLO Segment;
- Ultralytics YOLO OBB;
- COCO instances JSON.

The profile stores class frequency, objects-per-image statistics and normalized histograms for object center X/Y, bbox bottom-Y, width, height, area and aspect ratio.

## Learn a profile

```bash
scenepaste profile learn ./real_dataset -o distribution_profile.json
scenepaste profile show distribution_profile.json
```

Use `--bins` to change histogram resolution when learning.

## Generate from a profile

```bash
scenepaste generate \
  --objects ./objects \
  --backgrounds ./backgrounds \
  --output ./generated \
  --class-map "person=0,truck=1" \
  --count 100000 \
  --distribution-profile distribution_profile.json \
  --profile-strength 1.0 \
  --workers 0
```

`--profile-strength` is in `[0, 1]`:

- `1.0`: every sample is planned from the learned distribution;
- `0.5`: roughly half profile-driven, half normal ScenePaste random planning;
- `0.0`: profile is loaded but does not drive placement.

When a profile is active, ScenePaste copies the relevant target profile into the generated dataset as `target_distribution_profile.json`. `scenepaste qa` can then compare the generated distribution against that target.

## What this does not learn

The current profile is image-space and 2D. It does not infer camera intrinsics, true metric depth, physical collision constraints, lighting, object pose semantics or causal scene relationships. Treat it as statistical guidance, not a replacement for physically grounded simulation.

For scenes where relationships matter, combine a profile with a human-authored parameterized Scene Template rather than relying on global statistics alone.


## Mix multiple domains

Learn each real domain separately, then combine normalized distributions with explicit weights:

```bash
scenepaste profile mix factory.json tunnel.json -w 0.7 0.3 -o mixed_profile.json
```

Each source profile is normalized before weighting, so a large source dataset does not dominate merely because it contains more images. The mixed JSON is a normal ScenePaste DistributionProfile and can be passed to `--distribution-profile`.

## v1.0 crowding and visibility-aware statistics

Profiles now include two extra geometry signals per class:

- `overlap_iou`: the maximum bbox overlap IoU with another object in the same image. This is available for Detect, Seg, OBB, COCO and LabelMe and acts as an observable **crowding / occlusion proxy**.
- `visible_shape_fraction`: polygon/mask area divided by its axis-aligned bbox area. This is only meaningful when polygon or mask geometry exists (Seg/OBB/COCO/LabelMe polygon).

When a dataset contains multiple YOLO modalities, `profile learn` defaults to the richer geometry source. Override it when needed:

```bash
scenepaste profile learn ./dataset --geometry-source seg -o profile.json
scenepaste profile learn ./dataset --geometry-source obb -o profile.json
scenepaste profile learn ./dataset --geometry-source detect -o profile.json
```

A distribution profile does **not** claim to recover the invisible full shape of a real occluded object from Detect labels. Detect-only overlap is explicitly treated as a proxy. During generation, profile-sampled crowded objects are allowed to overlap more often; actual rendered visibility is measured separately in each run's generation diagnostics.
