#!/usr/bin/env python3
"""生成示例数据：3 个合成目标 + 4 张合成背景。

示例数据完全是像素画的（非真实牧场素材），方便任何人 clone 后立刻试跑工具：
    python -m scenepaste gui \
        --objects ./samples/objects \
        --backgrounds ./samples/backgrounds \
        --output ./samples/generated

LabelMe JSON 里的多边形包裹的就是这些像素目标，工具会按标注抠出。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


SAMPLES_DIR = Path(__file__).resolve().parent
OBJECTS_DIR = SAMPLES_DIR / "objects"
BACKGROUNDS_DIR = SAMPLES_DIR / "backgrounds"


# ---------------------------------------------------------------------------
# 目标素材（带 LabelMe JSON）
# ---------------------------------------------------------------------------

def _labelme_json(filename: str, width: int, height: int, shapes: list) -> dict:
    return {
        "version": "4.0.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": filename,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def _shape(label: str, points: list, shape_type: str = "polygon") -> dict:
    return {
        "label": label,
        "score": None,
        "points": points,
        "group_id": None,
        "shape_type": shape_type,
        "description": "",
        "flags": {},
    }


def make_person_sample() -> None:
    """画一个简单的像素人：圆头 + 长方形身体 + 两条腿。"""
    w, h = 120, 240
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    # 头
    d.ellipse([40, 10, 80, 50], fill=(255, 200, 150), outline=(60, 40, 30))
    # 身体（红色上衣）
    d.rectangle([35, 50, 85, 140], fill=(200, 50, 50), outline=(80, 20, 20))
    # 手臂
    d.rectangle([20, 55, 35, 130], fill=(200, 50, 50), outline=(80, 20, 20))
    d.rectangle([85, 55, 100, 130], fill=(200, 50, 50), outline=(80, 20, 20))
    # 腿（深色裤子）
    d.rectangle([40, 140, 60, 220], fill=(40, 50, 100), outline=(20, 25, 60))
    d.rectangle([60, 140, 80, 220], fill=(40, 50, 100), outline=(20, 25, 60))
    # 鞋
    d.rectangle([38, 220, 62, 232], fill=(30, 30, 30))
    d.rectangle([58, 220, 82, 232], fill=(30, 30, 30))
    img.save(OBJECTS_DIR / "sample_person.jpg", quality=95)

    polygon = [
        [40, 10], [80, 10], [80, 50],       # 头顶
        [100, 55], [100, 130],               # 右手臂外
        [85, 130], [85, 140],                # 右身侧
        [80, 140], [80, 220],                # 右腿外
        [82, 232], [58, 232],                # 右鞋底
        [60, 220], [40, 220],                # 裆部
        [40, 140], [35, 140], [35, 130],     # 左身侧
        [20, 130], [20, 55],                 # 左手臂外
        [35, 50], [40, 10],                  # 回到头顶
    ]
    data = _labelme_json("sample_person.jpg", w, h, [_shape("person", polygon)])
    (OBJECTS_DIR / "sample_person.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def make_truck_sample() -> None:
    """画一辆像素卡车：货箱 + 驾驶室 + 车轮。"""
    w, h = 320, 180
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    # 货箱（蓝色）
    d.rectangle([10, 40, 220, 130], fill=(50, 100, 200), outline=(20, 50, 120))
    # 驾驶室（浅蓝）
    d.rectangle([220, 60, 300, 130], fill=(120, 170, 230), outline=(40, 80, 140))
    # 风挡
    d.rectangle([240, 70, 290, 105], fill=(200, 230, 255), outline=(40, 80, 140))
    # 底盘
    d.rectangle([10, 130, 310, 145], fill=(60, 60, 60))
    # 车轮
    d.ellipse([40, 130, 90, 180], fill=(30, 30, 30), outline=(80, 80, 80))
    d.ellipse([50, 140, 80, 170], fill=(120, 120, 120))
    d.ellipse([230, 130, 280, 180], fill=(30, 30, 30), outline=(80, 80, 80))
    d.ellipse([240, 140, 270, 170], fill=(120, 120, 120))
    img.save(OBJECTS_DIR / "sample_truck.jpg", quality=95)

    polygon = [
        [10, 40], [220, 40],                # 货箱顶
        [220, 60], [300, 60],               # 驾驶室顶
        [300, 130], [310, 130],             # 右侧到底盘
        [310, 145],
        [280, 145], [280, 180], [230, 180], # 右轮
        [230, 145],
        [90, 145], [90, 180], [40, 180],    # 左轮
        [40, 145],
        [10, 145], [10, 40],                # 回到起点
    ]
    data = _labelme_json("sample_truck.jpg", w, h, [_shape("truck", polygon)])
    (OBJECTS_DIR / "sample_truck.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def make_motorcycle_sample() -> None:
    """画一辆像素摩托车。"""
    w, h = 240, 140
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    # 车架（黑色）
    d.polygon([(40, 90), (200, 90), (210, 80), (180, 70), (60, 70), (40, 90)],
              fill=(40, 40, 40))
    # 油箱（橙色）
    d.rectangle([80, 50, 160, 80], fill=(230, 130, 30), outline=(120, 60, 10))
    # 座椅
    d.rectangle([150, 45, 200, 65], fill=(20, 20, 20))
    # 把手
    d.rectangle([40, 35, 80, 50], fill=(20, 20, 20))
    d.rectangle([40, 30, 50, 50], fill=(20, 20, 20))
    # 车轮
    d.ellipse([10, 80, 80, 140], fill=(20, 20, 20), outline=(80, 80, 80))
    d.ellipse([25, 95, 65, 130], fill=(100, 100, 100))
    d.ellipse([160, 80, 230, 140], fill=(20, 20, 20), outline=(80, 80, 80))
    d.ellipse([175, 95, 215, 130], fill=(100, 100, 100))
    img.save(OBJECTS_DIR / "sample_motorcycle.jpg", quality=95)

    polygon = [
        [10, 110], [10, 140], [80, 140], [80, 95],    # 左轮
        [80, 70], [60, 70], [40, 90],                  # 车架左
        [40, 50], [80, 50],                            # 把手
        [80, 45], [150, 45], [150, 50],                # 油箱顶
        [160, 50], [160, 80],                          # 油箱右
        [200, 80], [200, 65], [150, 65],               # 座椅
        [200, 95],                                     # 座椅右
        [230, 95], [230, 140], [160, 140], [160, 110], # 右轮
    ]
    data = _labelme_json("sample_motorcycle.jpg", w, h, [_shape("motorcycle", polygon)])
    (OBJECTS_DIR / "sample_motorcycle.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# 背景图
# ---------------------------------------------------------------------------

def _gradient(w: int, h: int, top: tuple, bottom: tuple) -> Image.Image:
    """画一个垂直渐变。"""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        t = y / max(h - 1, 1)
        arr[y, :] = [int(top[i] + (bottom[i] - top[i]) * t) for i in range(3)]
    return Image.fromarray(arr, "RGB")


def make_pasture_bg(name: str, sky_top: tuple, sky_bottom: tuple,
                    grass: tuple, with_road: bool = False) -> None:
    """画一张牧场背景：渐变天空 + 地面 + 远山。"""
    w, h = 800, 600
    img = _gradient(w, h, sky_top, sky_bottom)
    d = ImageDraw.Draw(img)
    # 远山
    mountain_color = (90, 110, 100) if sky_top[0] > 150 else (60, 80, 90)
    d.polygon([(0, 360), (180, 280), (320, 340), (500, 260),
               (680, 320), (800, 290), (800, 400), (0, 400)],
              fill=mountain_color)
    # 草地
    d.rectangle([0, 380, w, h], fill=grass)
    # 加点草地纹理（小色斑）
    rng = np.random.default_rng(hash(name) & 0xFFFF)
    for _ in range(200):
        x = int(rng.integers(0, w))
        y = int(rng.integers(400, h))
        size = int(rng.integers(2, 6))
        delta = int(rng.integers(-20, 20))
        c = tuple(max(0, min(255, v + delta)) for v in grass)
        d.ellipse([x, y, x + size, y + size], fill=c)
    # 可选：一条土路
    if with_road:
        road = (140, 110, 80)
        d.polygon([(0, 600), (200, 480), (260, 460), (320, 480), (320, 600)], fill=road)
    img.save(BACKGROUNDS_DIR / name, quality=92)


def make_backgrounds() -> None:
    # 晴天
    make_pasture_bg("sample_bg_sunny.jpg",
                    sky_top=(140, 200, 240), sky_bottom=(220, 240, 255),
                    grass=(90, 140, 70))
    # 黄昏
    make_pasture_bg("sample_bg_dusk.jpg",
                    sky_top=(240, 170, 110), sky_bottom=(255, 220, 180),
                    grass=(110, 110, 70))
    # 阴天
    make_pasture_bg("sample_bg_cloudy.jpg",
                    sky_top=(160, 165, 175), sky_bottom=(200, 200, 205),
                    grass=(80, 110, 75))
    # 带路
    make_pasture_bg("sample_bg_with_road.jpg",
                    sky_top=(150, 200, 230), sky_bottom=(220, 235, 245),
                    grass=(95, 130, 65), with_road=True)


# ---------------------------------------------------------------------------

def main() -> int:
    OBJECTS_DIR.mkdir(parents=True, exist_ok=True)
    BACKGROUNDS_DIR.mkdir(parents=True, exist_ok=True)
    print("生成示例目标…")
    make_person_sample()
    make_truck_sample()
    make_motorcycle_sample()
    print("生成示例背景…")
    make_backgrounds()
    print(f"完成。目标目录：{OBJECTS_DIR}")
    print(f"     背景目录：{BACKGROUNDS_DIR}")
    print("\n启动示例：")
    print("  python -m scenepaste gui \\")
    print("      --objects ./samples/objects \\")
    print("      --backgrounds ./samples/backgrounds \\")
    print("      --output ./samples/generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
