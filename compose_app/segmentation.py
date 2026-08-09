"""分割多边形变换与序列化（纯函数，无 tkinter 依赖，可单测）。

变换链必须与 ``compose_app.rendering.render_instance`` 在几何上完全一致：
    1. resize：以原图高度为基准等比缩放
    2. flip：水平翻转
    3. rotate(expand=True)：绕图像中心顺时针旋转 angle 度（PIL 用 -angle）
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 几何变换：与 render_instance 一致
# ---------------------------------------------------------------------------

def transform_local_polygon(
    polygon: np.ndarray,
    source_w: int,
    source_h: int,
    target_h: int,
    flip: bool,
    angle: float,
) -> Optional[Tuple[np.ndarray, Tuple[int, int]]]:
    """对一个 cutout 的局部多边形应用 resize+flip+rotate。

    返回 (新多边形, 渲染后图像尺寸 (w, h))；输入不足返回 None。
    """
    if polygon is None or len(polygon) < 3:
        return None
    pts = np.asarray(polygon, dtype=np.float64).copy()
    if source_h <= 0 or source_w <= 0 or target_h < 4:
        return None

    # 1) resize
    scale = float(target_h) / float(source_h)
    target_w = max(2, int(round(source_w * scale)))
    pts[:, 0] *= scale
    pts[:, 1] *= scale

    # 2) flip：水平镜像
    if flip:
        pts[:, 0] = target_w - pts[:, 0]

    # 3) rotate(-angle, expand=True)
    if abs(angle) > 0.05:
        theta = math.radians(-angle)  # 与 PIL 一致
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        cx_pre, cy_pre = target_w / 2.0, target_h / 2.0
        x0 = pts[:, 0] - cx_pre
        y0 = pts[:, 1] - cy_pre
        rx = x0 * cos_t - y0 * sin_t + cx_pre
        ry = x0 * sin_t + y0 * cos_t + cy_pre
        # 计算 expand 后的新画布尺寸（旋转后 AABB）
        corners = np.array([[0, 0], [target_w, 0],
                            [target_w, target_h], [0, target_h]], dtype=np.float64)
        cx0 = corners[:, 0] - cx_pre
        cy0 = corners[:, 1] - cy_pre
        rx_c = cx0 * cos_t - cy0 * sin_t + cx_pre
        ry_c = cx0 * sin_t + cy0 * cos_t + cy_pre
        new_w = max(2, int(math.ceil(rx_c.max() - rx_c.min())))
        new_h = max(2, int(math.ceil(ry_c.max() - ry_c.min())))
        offset_x = -rx_c.min()
        offset_y = -ry_c.min()
        pts = np.column_stack([rx + offset_x, ry + offset_y])
        return pts.astype(np.float32), (new_w, new_h)

    return pts.astype(np.float32), (target_w, target_h)


def place_polygon(local_polygon: np.ndarray, x1: float, y1: float) -> np.ndarray:
    """把渲染后局部多边形平移到背景坐标系。"""
    out = np.asarray(local_polygon, dtype=np.float64).copy()
    out[:, 0] += x1
    out[:, 1] += y1
    return out


def clip_polygon_to_canvas(polygon: np.ndarray, w: int, h: int) -> Optional[np.ndarray]:
    """Sutherland–Hodgman 轴对齐矩形裁剪。"""
    if polygon is None or len(polygon) < 3:
        return None
    poly = np.asarray(polygon, dtype=np.float64)
    # 每条边：(axis, value, keep_le)；axis=0→x，axis=1→y
    edges = [
        (0, 0.0, False),       # x >= 0
        (0, float(w), True),   # x <= w
        (1, 0.0, False),       # y >= 0
        (1, float(h), True),   # y <= h
    ]
    for axis, value, keep_le in edges:
        if len(poly) == 0:
            return None
        result = []
        n = len(poly)
        for i in range(n):
            cur = poly[i]
            nxt = poly[(i + 1) % n]
            cur_in = (cur[axis] <= value) if keep_le else (cur[axis] >= value)
            nxt_in = (nxt[axis] <= value) if keep_le else (nxt[axis] >= value)
            if cur_in:
                result.append(cur)
                if not nxt_in:
                    result.append(_intersect_axis(cur, nxt, axis, value))
            elif nxt_in:
                result.append(_intersect_axis(cur, nxt, axis, value))
        if not result:
            return None
        poly = np.array(result, dtype=np.float64)

    if len(poly) >= 3:
        return poly.astype(np.float32)
    return None


def _intersect_axis(a: np.ndarray, b: np.ndarray, axis: int, value: float) -> np.ndarray:
    """线段 a→b 与 x=value 或 y=value 的交点（精确）。"""
    d = b[axis] - a[axis]
    if abs(d) < 1e-12:
        return a.copy()
    t = (value - a[axis]) / d
    return a + t * (b - a)


# ---------------------------------------------------------------------------
# 序列化
# ---------------------------------------------------------------------------

def yolo_seg_line(class_id: int, polygon: np.ndarray, width: int, height: int) -> str:
    """归一化多边形 → YOLO-seg 行：`class x1 y1 x2 y2 ...`（坐标 0~1）。"""
    if polygon is None or len(polygon) < 3 or width <= 0 or height <= 0:
        return ""
    pts = np.asarray(polygon, dtype=np.float64).copy()
    pts[:, 0] = np.clip(pts[:, 0] / width, 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1] / height, 0.0, 1.0)
    coords = " ".join(f"{v:.6f}" for v in pts.reshape(-1))
    return f"{int(class_id)} {coords}"


def coco_polygon(polygon: np.ndarray) -> List[float]:
    """扁平化为 COCO segmentation 格式 `[x1,y1,x2,y2,...]`。"""
    if polygon is None or len(polygon) < 3:
        return []
    return [float(v) for v in np.asarray(polygon, dtype=np.float64).reshape(-1)]


def polygon_area(polygon: np.ndarray) -> float:
    """Shoelace 面积（绝对值）。"""
    if polygon is None or len(polygon) < 3:
        return 0.0
    pts = np.asarray(polygon, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_bbox(polygon: np.ndarray) -> Tuple[float, float, float, float]:
    """返回 (x1, y1, x2, y2)。"""
    if polygon is None or len(polygon) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    pts = np.asarray(polygon, dtype=np.float64)
    return (float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max()))


# ---------------------------------------------------------------------------
# 顶层封装：从 Instance 直接得到画布坐标的多边形
# ---------------------------------------------------------------------------

def instance_canvas_polygon(
    cutout,
    instance,
    bg_w: int,
    bg_h: int,
) -> Optional[np.ndarray]:
    """计算一个 instance 在背景画布坐标系下的多边形（已 clip 到画布范围）。

    与 ``Instance.get_rendered`` 共用同一个目标高度，保证几何一致。
    """
    if getattr(cutout, "polygon", None) is None:
        return None
    rgba = cutout.rgba
    target_h = max(4, int(round(instance.h_ratio * bg_h)))
    result = transform_local_polygon(
        cutout.polygon, rgba.width, rgba.height, target_h, instance.flip, instance.angle,
    )
    if result is None:
        return None
    local_poly, _size = result
    # instance.cx/cy 是渲染后图像（含 expand）的中心，要真实算一次渲染拿尺寸
    from .rendering import render_instance
    rendered = render_instance(rgba, target_h, instance.flip, instance.angle)
    x1 = int(round(instance.cx - rendered.width / 2.0))
    y1 = int(round(instance.cy - rendered.height / 2.0))
    placed = place_polygon(local_poly, x1, y1)
    return clip_polygon_to_canvas(placed, bg_w, bg_h)

# ---------------------------------------------------------------------------
# Mask-first helpers：从真正渲染后的 alpha 计算可见区域
# ---------------------------------------------------------------------------

def instance_canvas_mask(cutout, instance, bg_w: int, bg_h: int,
                         alpha_threshold: int = 32) -> np.ndarray:
    """返回单个实例投影到背景画布后的 bool mask。

    该路径直接复用 ``Instance.get_rendered``，因此 resize / flip / rotate 与
    GUI 最终渲染完全一致，比仅变换原 polygon 更适合生成训练真值。
    """
    target_h = max(4, int(round(instance.h_ratio * bg_h)))
    rendered = instance.get_rendered(cutout.rgba, target_h)
    alpha = np.asarray(rendered.getchannel("A")) > int(alpha_threshold)
    mask = np.zeros((bg_h, bg_w), dtype=bool)
    x1 = int(round(instance.cx - rendered.width / 2.0))
    y1 = int(round(instance.cy - rendered.height / 2.0))
    x2, y2 = x1 + rendered.width, y1 + rendered.height
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(bg_w, x2), min(bg_h, y2)
    if cx2 <= cx1 or cy2 <= cy1:
        return mask
    lx1, ly1 = cx1 - x1, cy1 - y1
    lx2, ly2 = lx1 + (cx2 - cx1), ly1 + (cy2 - cy1)
    mask[cy1:cy2, cx1:cx2] = alpha[ly1:ly2, lx1:lx2]
    return mask


def visible_instance_masks(cutouts, instances, bg_w: int, bg_h: int,
                           alpha_threshold: int = 32) -> List[np.ndarray]:
    """按画布 z-order 返回每个实例的最终可见 mask。

    ``instances`` 越靠后越在上层，因此会遮挡前面的实例。
    """
    raw: List[np.ndarray] = []
    for inst in instances:
        cut = cutouts[inst.cutout_index]
        raw.append(instance_canvas_mask(cut, inst, bg_w, bg_h, alpha_threshold))
    covered = np.zeros((bg_h, bg_w), dtype=bool)
    visible = [np.zeros_like(covered) for _ in raw]
    for i in range(len(raw) - 1, -1, -1):
        visible[i] = raw[i] & ~covered
        covered |= raw[i]
    return visible
