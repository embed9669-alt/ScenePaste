# Output formats

## YOLO detect

`labels/train/<stem>.txt`

```text
class cx cy w h
```

All geometry values are normalized to `[0, 1]`.

## YOLO segmentation

When `--output-format seg` is selected, labels are written to:

```text
labels-seg/train/<stem>.txt
```

Each line:

```text
class x1 y1 x2 y2 ...
```

The polygon is derived from the final visible alpha mask. When occlusion creates multiple disconnected regions, the largest outer contour is used for the YOLO text line.

`both` keeps detection labels in `labels/train` and writes an additional segmentation export to `labels-seg/train`.

## YOLO OBB

Ultralytics-compatible four-corner format:

```text
class x1 y1 x2 y2 x3 y3 x4 y4
```

Labels live in `labels-obb/train`. The exporter writes Ultralytics four-corner OBB rows (`class x1 y1 ... x4 y4`).

## Semantic segmentation

```text
masks/train/<stem>.png
semantic_classes.json
```

Mask value convention:

```text
0 = background
1 = class_id 0
2 = class_id 1
...
```

`uint8` is used while the maximum value fits in 255; the writer switches to `uint16` when required.

## COCO instances

```text
instances_coco.json
```

Visible masks drive `segmentation`, `bbox` and `area`. COCO can preserve multiple disconnected outer polygons for one visible instance.


## One-pass multi-task export

`--output-format all` renders the scene once and exports Detect, Seg, OBB, Semantic and COCO annotations from the same mask-first visible geometry. This is useful when the same synthetic corpus is consumed by several perception experiments.
