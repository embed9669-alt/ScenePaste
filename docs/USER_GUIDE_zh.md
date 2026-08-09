# 中文详细使用说明

## 1. 准备目标素材

推荐使用 LabelMe 或 X-AnyLabeling 多边形标注：

```text
objects/
├── person_001.jpg
├── person_001.json
├── truck_001.jpg
└── truck_001.json
```

类别名必须与 `--class-map` 对应，例如：

```text
person=0,truck=1,motorcycle=2
```

## 2. 准备背景

```text
backgrounds/
├── bg_001.jpg
├── bg_002.jpg
└── bg_003.jpg
```

背景最好不要包含没有标注的目标类别，否则容易产生错误监督。

## 3. 可选：标注 paste_zone

同名 JSON 中用 polygon 标注地面区域，label 写 `paste_zone`。程序优先从该区域采样目标落地点。

如果不同类别可出现的区域不同，可以直接标：

```text
paste_zone:person
paste_zone:truck
zone:forklift
ground:motorcycle
```

类别专属区域优先级高于通用 `paste_zone`。例如固定监控相机下，可以把行人限制在人行通道、叉车限制在车道。

## 4. GUI 编辑

```bash
scenepaste gui --objects ./objects --backgrounds ./backgrounds --output ./generated
```

也可点工具栏 **★ 加载示例**。正式 wheel 已内置这套示例资源，因此 pip/wheel 安装后也能直接使用；源码仓库同时保留根目录 `samples/` 方便 CLI 示例。

![Qt 界面总览](images/ui_overview.png)

常用操作：

| 操作 | 功能 |
|---|---|
| 双击 / 拖入左侧素材 | 加入画布 |
| 鼠标拖动 | 移动 |
| 滚轮（选中目标后） | 缩放贴图 |
| Shift+拖动 | 旋转贴图 |
| 右侧缩放 / 旋转角度 | 精确数值调整 |
| 水平翻转 / F | 左右翻转 |
| Ctrl+滚轮 | 画布缩放 |
| Ctrl+Z / Ctrl+Y | 撤销 / 重做 |
| Delete | 删除选中 |
| Ctrl+S | 保存当前 |
| ⚡ 批量套用 | 把布局套到后续背景 |
| 📝 / 📂 模板 | 保存 / 加载场景模板 |
| ⚙ 项目设置 | 输出格式、阴影、色调匹配等 |
| 🔎 数据集浏览 | 检查 Detect / Seg / OBB / COCO / Semantic 标签 |

左侧素材库支持按类别名搜索过滤。

## 5. 场景模板

如果一个场景需要反复生成，例如“车辆遮挡行人”“远处小目标”“道路边缘三轮车”，先人工摆好一次，然后用工具栏 **📝 存模板**。Scene Template v2 不仅保存名义布局，还可以保存参数范围：

- 源素材身份与类别；
- 相对位置；
- 高度比例；
- 翻转与旋转；
- X/Y 位置抖动范围；
- 尺度、角度随机范围；
- 单个实例出现概率；
- 翻转概率；
- 是否随机替换为同类别其他素材；
- 是否允许模板内有意遮挡。

通过工具栏 **📝 存模板 / 📂 读模板** 操作。保存时会出现参数化设置对话框。加载会**替换**当前画布布局（可撤销），展示模板的名义场景；真正批量生成时会按 v2 参数逐张采样。换到不同分辨率背景时会按比例恢复；模板优先按源身份匹配，移动工程后还能按 `文件名#shape_index` 匹配，最后才使用唯一类别名兜底。

仓库自带 `samples/templates/parameterized_mixed_traffic.json` 参数化示例，以及 `samples/templates/constrained_person_truck.json` 关系约束示例。模板还支持 `left_of / right_of / above / below / min_distance / max_distance`。详见 [SCENE_TEMPLATES.md](SCENE_TEMPLATES.md) 和 [PLACEMENT_CONSTRAINTS.md](PLACEMENT_CONSTRAINTS.md)。

## 6. 批量套用

先在一张背景上摆好场景，然后点工具栏 **⚡ 批量套用**，在对话框里选择范围：

```text
后续 5 / 10 / 15 / 20 / 全部剩余
```

目标会按背景相对坐标和高度比例映射到新背景。批量任务在后台线程执行，取消按钮会等待当前原子保存结束后再真正退出。

## 7. 从真实数据学习生成分布

先从真实标注数据学习 Profile：

```bash
scenepaste profile learn ./real_dataset -o distribution_profile.json
scenepaste profile show distribution_profile.json
```

支持 LabelMe、YOLO Detect/Seg/OBB 和 COCO。Profile 会学习类别比例、每图目标数、位置、bbox 底部位置、尺度、面积和宽高比分布。然后：

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

`--profile-strength 1.0` 表示全部样本按真实分布规划，`0.5` 表示约一半使用 Profile、一半使用普通随机规划。

如果项目有多个 Domain，例如“厂房 + 隧道”或“白天 + 夜间”，可以先分别学习，再按权重混合：

```bash
scenepaste profile mix factory.json tunnel.json -w 0.7 0.3 -o mixed_profile.json
```

每个 Profile 会先归一化，因此不会仅因为某个真实数据集图片更多就自动支配混合结果。详见 [DISTRIBUTION_PROFILES.md](DISTRIBUTION_PROFILES.md)。

## 8. 增强 Recipe、融合方式与纯背景负样本

ScenePaste 支持在场景几何与标注确定后执行**不改变空间位置**的图像级增强。这样 Detect / Seg / OBB / Semantic / COCO 标签仍然保持对齐。

查看内置 Recipe：

```bash
scenepaste recipe list
scenepaste recipe show surveillance
```

典型批量生成：

```bash
scenepaste generate ... \
  --output-format all \
  --augmentation-recipe surveillance \
  --blend-mode gaussian \
  --blend-sigma 1.5 \
  --empty-scene-prob 0.10
```

- `clean`：不增加场景后处理；
- `camera-mild`：轻量亮度/对比度、Gamma、噪声、压缩和分辨率退化；
- `surveillance`：更偏监控画面的压缩、噪声、下采样、运动模糊；
- `low-light`：弱光、噪声和暗角。

`--blend-mode` 支持 `alpha / hard / gaussian`；`--empty-scene-prob` 用于按比例生成**纯背景负样本**，对减少误检尤其有帮助。可以把内置 Recipe 导出成 JSON 再按自己的真实相机修改：

```bash
scenepaste recipe export camera-mild -o my_camera.json
```

详见 [AUGMENTATION_RECIPES.md](AUGMENTATION_RECIPES.md)。

## 9. 可恢复多进程大规模生成

推荐给十万/百万级任务显式设置 Run ID：

```bash
scenepaste generate ... \
  --count 500000 \
  --workers 8 \
  --queue-depth 16 \
  --run-id production_a \
  --preview-ratio 0.01
```

运行状态持久化在：

```text
<output>/.scenepaste/runs/<run_id>.sqlite3
```

如果中断，使用相同配置恢复：

```bash
scenepaste generate ... --run-id production_a --resume
```

ScenePaste 只补未完成 index，并在恢复前校验配置、Profile 和模板内容哈希，避免错误续跑。`workers=0` 会自动使用约 `CPU-1` 个进程。详见 [LARGE_RUNS.md](LARGE_RUNS.md)。

GUI 中可直接点击 **🚀 大规模生成** 配置这些参数，并可以从真实数据直接学习 Profile。

## 10. CLI 均衡采样

批量随机生成默认使用：

```bash
--asset-sampling balanced
--background-sampling balanced
```

第一项先均衡抽类别，再在类别内抽素材；第二项让背景按洗牌轮次使用，降低重复次数偏斜。如果你明确需要所有 cutout 完全等概率或背景完全随机，可以改成 `random`。

## 11. 输出模式

- `detect`：目标检测
- `seg`：实例分割
- `both`：检测 + 额外分割副本
- `coco`：COCO instance segmentation
- `semantic`：语义分割 mask
- `obb`：旋转框
- `all`：单次渲染同时导出 Detect + Seg + OBB + Semantic + COCO

## 12. Dataset Explorer 与 QA Dashboard

生成完成后可直接：

```bash
scenepaste explore ./generated
```

浏览器会自动叠加当前图片对应的 Detect、Seg、OBB、COCO 和 Semantic 标注。左右方向键切图，适合在训练前快速抽查标签。主界面的 **🔎 数据集浏览** 也会打开同一个浏览器。

完整 QA Dashboard：

```bash
scenepaste qa ./generated
```

会生成 `qa_report.json` 与独立的 `qa_dashboard.html`，包括：

- 损坏图片和非法标注；
- 完全重复、pHash 感知近重复和 embedding 高相似候选；
- train / val / test 跨 split 泄漏；
- 类别、尺度、位置以及重叠/拥挤分布；
- 素材与背景复用；
- 目标 Profile 与实际生成分布偏移；
- mask-aware 输出模式下真实渲染得到的 visible ratio、按类别可见率和增强效果计数。

Detect-only 生成不会额外投影 mask 来计算 visible ratio，以避免纯检测大任务承担不必要开销。详见 [QA_DASHBOARD.md](QA_DASHBOARD.md)。

## 13. Project Manifest 与数据闭环中心

推荐一个项目保存一个可移植 `scenepaste.project.json`：

```bash
scenepaste project init . \
  --objects ./objects \
  --backgrounds ./backgrounds \
  --output ./generated \
  --class-map "person=0,truck=1"

scenepaste project set ./scenepaste.project.json \
  --real-dataset ./real \
  --validation-dataset ./real_val \
  --predictions ./runs/detect/predict/labels \
  --workers 8 \
  --output-format all \
  --preview-ratio 0.01

scenepaste project validate ./scenepaste.project.json --generation
scenepaste generate --project ./scenepaste.project.json --count 100000
```

Manifest 中的路径尽量保存为相对路径，方便工程整体移动；**显式 CLI 参数始终优先于工程默认参数**。

主 GUI 支持 **📁 打开工程 / 💼 保存工程**。点击 **🧠 数据闭环**，或运行：

```bash
scenepaste loop --project ./scenepaste.project.json
```

即可在一个窗口里完成 Profile 学习、Hard Mining、QA/泄漏检查、真实/合成对比、多样性筛选和 WebDataset Sharding。详见 [PROJECTS.md](PROJECTS.md) 和 [DATA_LOOP_CENTER.md](DATA_LOOP_CENTER.md)。

## 14. Detect / Seg / OBB 模型难例闭环

模型在真实 val/test 上推理后，把带置信度的预测 TXT 目录交给 ScenePaste：

```bash
# Detect
scenepaste curate hardmine ./real_dataset \
  --predictions ./runs/detect/predict/labels \
  --task detect --split val -o ./hardmine_detect

# YOLO Segment
scenepaste curate hardmine ./real_dataset \
  --predictions ./runs/segment/predict/labels \
  --task seg --split val -o ./hardmine_seg

# YOLO OBB
scenepaste curate hardmine ./real_dataset \
  --predictions ./runs/obb/predict/labels \
  --task obb --split val -o ./hardmine_obb
```

三种任务统一按几何 IoU 匹配，并统计 FN、FP、低置信 TP 和定位/几何质量不足。输出包括难例排序、HTML Dashboard、hard-negative 背景清单和可直接回灌生成器的 `hard_distribution_profile.json`。详见 [HARD_EXAMPLE_MINING.md](HARD_EXAMPLE_MINING.md)。

## 15. 数据策展、泄漏与可选 Embedding

默认 `cv-lite-v1` 完全离线、无需模型下载：

```bash
scenepaste curate leakage ./dataset
scenepaste curate diversity ./dataset --select 1000 --export-dataset ./diverse_1k
scenepaste compare ./real_dataset ./generated -o ./comparison
```

如果需要更强的语义相似度，可选安装 CLIP / DINOv2：

```bash
python -m pip install -e ".[embeddings]"
scenepaste curate leakage ./dataset --embedding-backend clip
scenepaste compare ./real_dataset ./generated --embedding-backend dinov2 -o ./comparison_dino
```

模型权重仅在明确选择对应 backend 时加载；推理按批处理执行，并在 PyTorch 检测到 CUDA 时自动使用 GPU。依赖或权重加载失败会明确报错，不会静默生成“0 个 embedding”的假成功结果。详见 [EMBEDDINGS.md](EMBEDDINGS.md) 与 [DATA_CURATION.md](DATA_CURATION.md)。

## 16. WebDataset 分片

```bash
scenepaste shard ./generated --split train --max-samples 10000 -o ./shards
```

每个 tar 的 sample 数、字节数和 SHA-256 会写入 manifest。Detect / Seg / OBB / Semantic / COCO 等已有模态会使用相同 basename 放入同一 sample。

## 17. 长任务实时状态

多进程生成会持续更新：

```text
<output>/.scenepaste/status/<run_id>.json
```

字段包括已完成/失败/总量、img/s、ETA 和剩余磁盘。GUI 的“大规模生成”窗口会实时显示这些信息。

## 18. 推荐的数据闭环

```text
建立 scenepaste.project.json
 -> 真实数据 profile learn（优先 auto/Seg/OBB 几何）
 -> GUI + 参数化模板定义关键场景关系
 -> 可选 profile mix（多 Domain）
 -> generate（多进程、可恢复、Recipe、负样本、多格式）
 -> explore + qa（人工抽检 + 自动 QA + 可见率诊断）
 -> 真实 val/test 训练评估
 -> curate hardmine（Detect / Seg / OBB）
 -> hard profile / hard negative 回灌下一轮生成
 -> leakage / diversity / compare 做数据策展
 -> shard 发布到大规模训练流水线
```

合成数据是否真正有效，最终仍应以**独立真实验证集/测试集上的模型表现**为准。
