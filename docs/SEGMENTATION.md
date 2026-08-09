# Segmentation and mask pipeline

[English](#english) · [中文](#中文)

## English

The project uses a **mask-first visible-instance pipeline** for segmentation-related outputs. Each rendered RGBA instance is placed on a full-canvas mask, then later layers remove pixels they occlude. Detection boxes, YOLO segmentation polygons, COCO instance annotations, semantic masks and OBB labels can therefore describe the same visible object geometry.

### Output formats

| Value | Output |
|---|---|
| `detect` | `labels/train/*.txt` — YOLO `class cx cy w h` |
| `seg` | `labels/train/*.txt` — Ultralytics YOLO segmentation polygons |
| `both` | detection in `labels/train/`, segmentation in `labels-seg/train/` |
| `coco` | detection labels + `instances_coco.json` |
| `semantic` | class mask PNGs in `masks/train/` + `semantic_classes.json` |
| `obb` | Ultralytics four-corner OBB labels in `labels/train/` |

### Visible-mask rule

Instances are rendered back-to-front. If a later instance covers an earlier instance, covered pixels are removed from the earlier instance's visible mask. Very small/empty visible masks are omitted from segmentation-derived outputs.

This is substantially more reliable than transforming only the source polygon because alpha feathering, rotation rasterization and occlusion are all reflected in the final mask. Polygon formats still require contour approximation, so they should not be described as mathematically pixel-identical to the raster mask.

### Semantic IDs

Semantic masks reserve `0` for background. Dataset class `class_id=N` is stored as mask value `N+1`. The mapping is written to `semantic_classes.json`. `uint8` is used when possible and `uint16` is used automatically for large class counts.

### OBB format

OBB files follow the Ultralytics dataset format:

```text
class x1 y1 x2 y2 x3 y3 x4 y4
```

Coordinates are normalized to `[0, 1]` and are derived from `cv2.minAreaRect` on the final visible mask.

## 中文

本项目对分割相关任务采用 **mask-first（最终可见掩码优先）** 的统一标注链路。每个 RGBA 目标先渲染成整幅画布上的实例 mask，再按照图层顺序扣除被前景目标遮挡的区域。因此检测框、YOLO 分割、COCO 实例分割、语义分割和 OBB 可以来自同一份最终可见几何信息。

### 输出格式

| 值 | 输出 |
|---|---|
| `detect` | `labels/train/*.txt`，YOLO 检测 |
| `seg` | `labels/train/*.txt`，Ultralytics YOLO 分割 |
| `both` | `labels/train/` 检测 + `labels-seg/train/` 分割 |
| `coco` | 检测标签 + `instances_coco.json` |
| `semantic` | `masks/train/*.png` + `semantic_classes.json` |
| `obb` | `labels/train/*.txt`，Ultralytics 四角点 OBB |

### 遮挡规则

图层靠前的目标会从后方目标的可见 mask 中扣除真实遮挡区域。这样不会再出现“图片里已经被挡住，但分割标签仍包含被挡区域”的问题。

### Semantic 类别值

`0` 永远表示背景；数据集中的 `class_id=N` 写成 mask 像素值 `N+1`。映射保存在 `semantic_classes.json`。类别较多时自动使用 `uint16`。

### OBB 格式

输出兼容 Ultralytics：

```text
class x1 y1 x2 y2 x3 y3 x4 y4
```

四个角点从最终可见 mask 的最小外接旋转矩形计算，并归一化到 `[0, 1]`。
