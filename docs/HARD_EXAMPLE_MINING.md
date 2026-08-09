# Hard Example Mining

ScenePaste v1.0 closes the loop between **model evaluation** and **the next synthetic-data generation run** for YOLO-style Detect, Segmentation, and OBB tasks.

## Supported prediction TXT

Coordinates are normalized.

### Detect

```text
class xc yc w h confidence
```

### OBB

```text
class x1 y1 x2 y2 x3 y3 x4 y4 confidence
```

### Segmentation

```text
class x1 y1 x2 y2 ... xn yn confidence
```

The confidence field is optional. For Seg/OBB, geometry IoU is computed by rasterizing normalized polygons into a common mask grid, so rotated boxes and non-axis-aligned polygons are scored using the same matching path.

## Run mining

```bash
# Detect
scenepaste curate hardmine ./real_val \
  --predictions ./runs/detect/predict/labels \
  --task detect --split val -o ./hardmine-detect

# Segmentation
scenepaste curate hardmine ./real_val \
  --predictions ./runs/segment/predict/labels \
  --task seg --split val -o ./hardmine-seg

# OBB
scenepaste curate hardmine ./real_val \
  --predictions ./runs/obb/predict/labels \
  --task obb --split val -o ./hardmine-obb
```

ScenePaste scores:

- false negatives (largest penalty);
- false positives;
- low-confidence matched objects;
- matched objects whose geometry IoU is below the localization threshold.

## Outputs

```text
hard_examples.json
hard_examples.csv
hard_examples.txt
hardmine_dashboard.html
hard_negative_backgrounds.txt
hard_distribution_profile.json
```

The hard profile is learned from the hard ground-truth subset and can immediately drive another generation run:

```bash
scenepaste generate \
  --project ./scenepaste.project.json \
  --distribution-profile ./hardmine-seg/hard_distribution_profile.json \
  --profile-strength 0.8 \
  --count 50000
```

For Seg/OBB hard subsets, profile learning uses the richer geometry labels when available. The resulting profile also contains scene crowding/overlap statistics.

## Interpretation

Hard-example mining is a prioritization tool, not a substitute for held-out evaluation. Review high-scoring samples before generating large follow-up runs; annotation errors and distribution shifts can also look like model failures.
