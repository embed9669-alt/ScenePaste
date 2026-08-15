# Asset Studio / 素材工作室

ScenePaste V10 把 **人工审核素材** 作为默认数据生产路径。
素材工作室用于修正分割结果，而不是用生成式模型重画目标。

## Why this exists

训练数据生成需要可控标签。ScenePaste 把问题拆成两类可人工编辑的资产：

1. **Foreground Mask** — 权威目标分割。该 Mask 成为透明前景 alpha，也是训练标签。
2. **Background Removal Mask** — 生成干净背景时要抹掉的区域。可比前景更大，以覆盖阴影、光晕或邻接像素。

```text
real annotated image
        |
        v
initial segmentation (LabelMe / auto cutout)
        |
        v
Asset Studio
  |                 |
  |                 +--> edit background-removal mask
  |                           |
  |                           +--> clean background
  |
  +--> edit foreground mask
              |
              +--> transparent foreground + reviewed LabelMe polygon

reviewed foregrounds + real/clean backgrounds
        |
        v
controlled Copy-Paste -> auto labels -> QA -> training dataset
```

## GUI workflow

打开 **素材 → 素材工作室…**，或点击工具栏 / 欢迎页上的 **素材工作室**。

1. 打开一张已标注图片或整个文件夹。
2. 从当前图中选择实例。
3. 选择 **编辑前景 Mask（训练标签）**。
4. 用 **画笔 +** / **橡皮擦 -** 修正分割。
5. 需要时使用填洞 / 膨胀 / 腐蚀 / 去毛刺。
6. 切换到 **编辑背景移除 Mask**。默认由前景外扩得到，可独立再改。
7. 点击 **预览干净背景**。效果不够再改移除 Mask 后重预览。
8. 点击 **保存审核素材：前景 + Mask + 干净背景**。

## What gets exported

审核后的前景直接写入目标素材库：

```text
objects/
  edited_assets/
    person/
      scene_001__i00__person.png
      scene_001__i00__person.json
```

干净背景写入背景库：

```text
backgrounds/
  edited_backgrounds/
    scene_001__clean_i00.jpg
```

溯源包保留自动与人工编辑两套 Mask：

```text
.scenepaste/asset_studio/scene_001__i00__person/
  original.jpg
  mask_auto.png
  mask_edited.png
  mask_background_remove.png
  foreground.png
  background_clean.jpg
  annotation_edited.json
  metadata.json
```

原始 Mask 不会被覆盖，方便核对自动分割质量并一键恢复。

## Background reconstruction

默认使用确定性 OpenCV inpainting（Telea）：只填充用户选定的移除区域，不会凭空生成新的前景语义。

## Recommended project policy

生产训练数据建议：

- 以真实图片与人工审核分割为真值；
- 通过素材工作室扩展前景 / 背景资产库；
- 大规模合成只用 **可控 Copy-Paste**；
- 在独立真实验证集 / 测试集上评估合成数据收益。
