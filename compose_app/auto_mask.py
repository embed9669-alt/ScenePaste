"""自动抠图后端：rembg / SAM 等（全部可选依赖，运行时按需懒加载）。

不安装 rembg 时 ``backends()["rembg"] == False``，调用 ``cutout_from_image`` 会抛
``ImportError``，提示 ``pip install 'scenepaste[auto]'``。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import threading

import cv2
import numpy as np

_REMBG_SESSION = None
_REMBG_LOCK = threading.Lock()


def _get_rembg_session():
    """进程内复用 rembg session，批量抠图时避免重复加载模型。"""
    global _REMBG_SESSION
    if _REMBG_SESSION is not None:
        return _REMBG_SESSION
    with _REMBG_LOCK:
        if _REMBG_SESSION is None:
            try:
                from rembg import new_session
            except ImportError as exc:
                raise ImportError(
                    "rembg 未安装。请运行：pip install 'scenepaste[auto]'"
                ) from exc
            _REMBG_SESSION = new_session()
    return _REMBG_SESSION


def backends() -> Dict[str, bool]:
    """探测当前环境里可用的自动抠图后端。

    注意：SAM 只 import 成功不能算"就绪"——还需要使用者注入已加载的
    mask generator。所以这里固定返回 sam=False，避免 UI 误导。
    要启用 SAM，请参考 docs/AUTO_MASK.md 自行实现 _sam_remove。
    """
    avail = {"sam": False}
    try:
        import rembg  # noqa: F401
        avail["rembg"] = True
    except Exception:
        avail["rembg"] = False
    return avail


def cutout_from_image(
    image: "np.ndarray | Path | str",
    backend: str = "rembg",
    label: str = "auto",
    class_id: int = 0,
    source: str = "auto",
    polygon_simplify_eps: float = 0.004,
):
    """从一张图直接抠出目标，返回 ``Cutout``。

    image 可以是 BGR ndarray、Path 或路径字符串。
    polygon_simplify_eps：cv2.approxPolyDP 的轮廓简化精度（按周长比例）。
    """
    from .models import Cutout

    arr = _to_bgr(image)
    if arr is None:
        raise ValueError("无法读取图像用于自动抠图")

    if backend == "rembg":
        rgba = _rembg_remove(arr)
    elif backend == "sam":
        rgba = _sam_remove(arr)
    else:
        raise ValueError(f"未知后端：{backend}（支持 rembg / sam）")

    polygon = _polygon_from_alpha(rgba[..., 3], polygon_simplify_eps)
    # 裁到非零区域，减少空白边
    if polygon is not None:
        x, y, w, h = cv2.boundingRect((polygon.astype(np.int32).reshape(-1, 1, 2)))
        if w >= 4 and h >= 4:
            pad = 2
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(rgba.shape[1], x + w + pad)
            y2 = min(rgba.shape[0], y + h + pad)
            rgba = rgba[y1:y2, x1:x2].copy()
            polygon = polygon - np.array([x1, y1], dtype=np.float32)

    from PIL import Image
    pil = Image.fromarray(rgba, mode="RGBA")
    return Cutout(label=label, class_id=class_id, source=source,
                  rgba=pil, polygon=polygon, thumb=None)


# ---------------------------------------------------------------------------
# 后端实现
# ---------------------------------------------------------------------------

def _rembg_remove(bgr: np.ndarray) -> np.ndarray:
    """调 rembg 输出 RGBA（与 OpenCV BGR 同尺寸）。"""
    try:
        import rembg
    except ImportError as exc:
        raise ImportError(
            "rembg 未安装。请运行：pip install 'scenepaste[auto]'"
        ) from exc
    except Exception as exc:
        # rembg is installed but a transitive dependency (e.g. numba +
        # coverage) crashed at import time. Treat as "unusable" so the
        # caller sees a clear ImportError instead of the underlying noise.
        raise ImportError(
            "rembg 当前不可用（依赖加载失败）。请运行：pip install 'scenepaste[auto]' "
            f"或检查环境。原因：{exc}"
        ) from exc
    rgb = bgr[..., ::-1].copy()  # → RGB
    out = rembg.remove(rgb, session=_get_rembg_session())
    if out.shape[-1] != 4:
        raise RuntimeError("rembg 返回结果不是 RGBA")
    # 转回 BGR 顺序的 RGBA，保持与 PIL.Image.fromarray(rgba) 一致
    out_bgr = out.copy()
    out_bgr[..., :3] = out[..., :3][..., ::-1]
    return out_bgr


def _sam_remove(bgr: np.ndarray) -> np.ndarray:
    """SAM 后端占位：要求用户提供点/框才能分割单实例。

    简单起见这里只做整图最大连通区，用作演示；生产场景应继承此类提供 prompt。
    """
    try:
        import segment_anything  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "segment-anything 未安装。请运行："
            "pip install 'git+https://github.com/facebookresearch/segment-anything.git'"
        ) from exc
    # 这里不实例化模型（需要权重路径）；上层应自行提供已加载的 generator
    raise NotImplementedError(
        "SAM 后端需要使用者注入已加载的 mask generator，"
        "请参考 docs/AUTO_MASK.md"
    )


def _polygon_from_alpha(alpha: np.ndarray, simplify_eps: float) -> Optional[np.ndarray]:
    """从 alpha 通道提取最大轮廓并简化为多边形。"""
    if alpha is None or alpha.size == 0:
        return None
    bin_mask = (alpha > 64).astype(np.uint8) * 255
    if bin_mask.sum() == 0:
        return None
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    if len(biggest) < 3:
        return None
    eps = simplify_eps * cv2.arcLength(biggest, True)
    approx = cv2.approxPolyDP(biggest, eps, True).reshape(-1, 2).astype(np.float32)
    if len(approx) < 3:
        return None
    return approx


def _to_bgr(image) -> Optional[np.ndarray]:
    if isinstance(image, np.ndarray):
        return image.copy()
    import scenepaste.core as core
    p = Path(image)
    # 用 EXIF 自动旋转版，避免手机原图横躺
    return core.imread_with_exif(p, cv2.IMREAD_COLOR)
