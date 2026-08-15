# Auto-cutout

[English](#english) · [中文](#中文)

## English

Auto-cutout lets the tool ingest ordinary images without LabelMe polygons. The currently supported backend is **rembg**.

### Install

```bash
pip install 'scenepaste[auto]'
```

The default U2Net weights (~168 MB) are downloaded to `~/.u2net/u2net.onnx` on first use. If GitHub releases are unreachable, place the file there manually (mirrors such as Hugging Face / hf-mirror work).

### CLI

```bash
scenepaste generate \
  --objects ./raw_images \
  --backgrounds ./backgrounds \
  --output ./generated \
  --auto-cutout \
  --auto-cutout-label person \
  --class-map "person=0"
```

Or classify by folder layout (`raw_images/person/*.jpg` → `person`):

```bash
scenepaste generate ... --auto-cutout --auto-cutout-label-from-subdir \
  --class-map "person=0,truck=1"
```

`--auto-cutout` reads supported images from `--objects`, extracts the alpha mask and derives a contour for annotation interoperability. Use `--auto-cutout-label` for a single class, or `--auto-cutout-label-from-subdir` when each class lives in its own subdirectory.

### GUI

The Qt editor exposes cutout / auto-cutout in three places:

1. **文件 / Toolbar → 抠图工作室…** — LabelMe-style studio:
   - **打开文件夹…** — left pane shows folder path + thumbnail browser; click to preview/annotate
   - **交互模式：点选分割** — click on an object (SAM2) → auto mask → set **类别名** → save LabelMe JSON
   - Or draw polygons manually; or rembg / GroundingSAM2 auto-cutout
   - Choose a **model**, click **下载并加载** (auto-downloads with visible % progress; optional hf-mirror)
   - Optional **batch** folder cutout
2. **Toolbar / 文件 → 自动抠图加载…** — rembg folder load into the canvas without writing JSON.
3. **大规模生成** — check **自动抠图 rembg**, set **抠图类别名**, optionally **按子文件夹名**.

Install extras:

```bash
pip install 'scenepaste[auto]'          # rembg
pip install 'scenepaste[grounding]'     # GroundingSAM2 (torch + transformers)
```

If Hugging Face is slow/unreachable in China, set `HF_ENDPOINT=https://hf-mirror.com` before downloading GroundingSAM2.

### Backend status

| Backend | Status |
|---|---|
| rembg (multiple models) | Supported; auto-download on first load |
| SAM2 click-to-segment | Supported in Cutout Studio（点选分割） |
| GroundingSAM2 | Supported via Hugging Face (Grounding DINO tiny + SAM2 / SAM fallback) |
| Classic SAM click-prompt | Covered by SAM2 click mode (falls back to SAM if needed) |

For best training quality, inspect automatically extracted masks before large-scale generation. Auto-cutout quality is model- and domain-dependent.

## 中文

自动抠图允许直接使用普通图片作为目标素材，不必先逐个画 LabelMe 多边形。当前正式支持的后端是 **rembg**。

### 安装

```bash
pip install 'scenepaste[auto]'
```

首次运行会把 U2Net 权重下载到 `~/.u2net/u2net.onnx`（约 168 MB）。若 GitHub 不可达，可自行放到该路径（可用 Hugging Face / hf-mirror 等镜像）。

### 命令行

```bash
scenepaste generate \
  --objects ./raw_images \
  --backgrounds ./backgrounds \
  --output ./generated \
  --auto-cutout \
  --auto-cutout-label person \
  --class-map "person=0"
```

或按子目录分类（`raw_images/person/*.jpg` → `person`）：

```bash
scenepaste generate ... --auto-cutout --auto-cutout-label-from-subdir \
  --class-map "person=0,truck=1"
```

### GUI

Qt 编辑器可用：

1. **文件 / 工具栏 → 抠图工作室…**
   - **打开文件夹…**：左侧显示目录路径与缩略图列表，点击预览/标注
   - **交互模式 → 点选分割**：在物体上点一下（SAM2）自动出 mask，上方「类别名」即标签，再保存 LabelMe
   - 或手绘多边形；或 rembg / GroundingSAM2 自动抠图
   - **选择模型** → **下载并加载**（可见百分比进度；国内勾选 hf-mirror）
   - 可选批量按类别自动抠图
2. **自动抠图加载…**：rembg 目录直接进画布（不写 JSON）
3. **大规模生成**：勾选自动抠图 rembg

安装：

```bash
pip install 'scenepaste[auto]'          # rembg
pip install 'scenepaste[grounding]'     # GroundingSAM2
```

国内 Hugging Face 较慢时可先设：`export HF_ENDPOINT=https://hf-mirror.com`

### 后端状态

| 后端 | 状态 |
|---|---|
| rembg（多模型） | 已支持；首次加载自动下载 |
| SAM2 点选分割 | 抠图工作室「点选分割」已支持 |
| GroundingSAM2 | 已支持（HF：Grounding DINO tiny + SAM2，必要时回退 SAM） |
| 手动画多边形 | 抠图工作室内可用 |

大规模生成之前建议抽查自动 mask。自动抠图质量与具体场景和模型能力有关。
