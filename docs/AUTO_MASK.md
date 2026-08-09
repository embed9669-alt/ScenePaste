# Auto-cutout

[English](#english) · [中文](#中文)

## English

Auto-cutout lets the tool ingest ordinary images without LabelMe polygons. The currently supported backend is **rembg**.

### Install

```bash
pip install 'scenepaste[auto]'
```

### CLI

```bash
scenepaste generate \
  --objects ./raw_images \
  --backgrounds ./backgrounds \
  --output ./generated \
  --auto-cutout \
  --class-map "auto=0"
```

`--auto-cutout` reads supported images from `--objects`, reuses one rembg model session for batch efficiency, extracts the alpha mask and derives a contour for annotation interoperability.

### GUI

The current Qt desktop editor does **not** yet expose an auto-cutout button.
Use the CLI `--auto-cutout` flow above, then load the resulting LabelMe/cutout
assets into the GUI with **加载目标**.

### Backend status

| Backend | Status |
|---|---|
| rembg | Supported |
| SAM | Interface reserved; not reported as ready until a real prompt/model integration exists |

For best training quality, inspect automatically extracted masks before large-scale generation. Auto-cutout quality is model- and domain-dependent.

## 中文

自动抠图允许直接使用普通图片作为目标素材，不必先逐个画 LabelMe 多边形。当前正式支持的后端是 **rembg**。

### 安装

```bash
pip install 'scenepaste[auto]'
```

### 命令行

```bash
scenepaste generate \
  --objects ./raw_images \
  --backgrounds ./backgrounds \
  --output ./generated \
  --auto-cutout \
  --class-map "auto=0"
```

批量抠图会复用同一个 rembg session，避免每张图片重复加载模型。alpha mask 会同时用于生成透明素材，并提取轮廓供分割/COCO 等格式使用。

### GUI

当前 Qt 桌面编辑器**尚未**提供自动抠图按钮。请先用上面的 CLI `--auto-cutout`
生成素材，再在 GUI 里用 **加载目标** 导入。

### 后端状态

| 后端 | 状态 |
|---|---|
| rembg | 已支持 |
| SAM | 预留接口；在真正完成模型与 prompt 集成之前不会显示“已就绪” |

大规模生成之前建议抽查自动 mask。自动抠图质量与具体场景和模型能力有关。
