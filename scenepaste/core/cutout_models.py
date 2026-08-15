"""Selectable cutout models with auto-download and lazy load.

Supported families:
- **rembg** — U2Net / ISNet / BiRefNet etc. (weights via rembg cache ``~/.u2net``)
- **grounding_sam2** — Grounding DINO + SAM2 (Hugging Face ``from_pretrained``)

Use :func:`list_cutout_models` for the GUI combo, :func:`ensure_model` to
download/load, then :func:`predict_cutout` with a text prompt (class label).
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from .auto_cutout import alpha_to_polygon

ProgressCb = Callable[..., None]  # (msg: str, pct: Optional[float] = None)

# Process-wide loaded sessions (model_id -> runtime object)
_LOCK = threading.Lock()
_LOADED: Dict[str, object] = {}
_READY: Dict[str, bool] = {}

HF_MIRROR_DEFAULT = "https://hf-mirror.com"
HF_OFFICIAL = "https://huggingface.co"


@dataclass(frozen=True)
class CutoutModelSpec:
    """One selectable cutout backend."""

    id: str
    title: str
    family: str  # rembg | grounding_sam2
    description: str
    # rembg session name, or HF detector / segmenter ids
    rembg_name: str = ""
    detector_id: str = ""
    segmenter_id: str = ""
    needs_text_prompt: bool = False
    pip_hint: str = ""


def list_cutout_models() -> List[CutoutModelSpec]:
    """Models shown in the Cutout Studio combo (stable order)."""
    return [
        CutoutModelSpec(
            id="rembg:u2net",
            title="rembg · U2Net（通用）",
            family="rembg",
            description="通用抠图，首次自动下载到 ~/.u2net/",
            rembg_name="u2net",
            pip_hint="pip install 'scenepaste[auto]'",
        ),
        CutoutModelSpec(
            id="rembg:u2netp",
            title="rembg · U2NetP（轻量）",
            family="rembg",
            description="更小更快的 U2Net",
            rembg_name="u2netp",
            pip_hint="pip install 'scenepaste[auto]'",
        ),
        CutoutModelSpec(
            id="rembg:u2net_human_seg",
            title="rembg · 人体分割",
            family="rembg",
            description="偏人像/人体",
            rembg_name="u2net_human_seg",
            pip_hint="pip install 'scenepaste[auto]'",
        ),
        CutoutModelSpec(
            id="rembg:isnet-general-use",
            title="rembg · ISNet 通用",
            family="rembg",
            description="ISNet general-use",
            rembg_name="isnet-general-use",
            pip_hint="pip install 'scenepaste[auto]'",
        ),
        CutoutModelSpec(
            id="rembg:birefnet-general",
            title="rembg · BiRefNet 通用",
            family="rembg",
            description="更高质量通用抠图（更大）",
            rembg_name="birefnet-general",
            pip_hint="pip install 'scenepaste[auto]'",
        ),
        CutoutModelSpec(
            id="sam2_click",
            title="SAM2 点选分割（点击物体）",
            family="sam2_point",
            description=(
                "在图上点一下即可分割该物体，再套用上方类别标签。"
                "首次自动下载 SAM2 权重（可用 hf-mirror）。"
            ),
            segmenter_id="facebook/sam2.1-hiera-tiny",
            needs_text_prompt=False,
            pip_hint="pip install 'scenepaste[grounding]'",
        ),
        CutoutModelSpec(
            id="grounding_sam2",
            title="GroundingSAM2（按类别文本抠图）",
            family="grounding_sam2",
            description=(
                "Grounding DINO 检测 + SAM2 分割；用类别名作为文本提示。"
                "首次自动从 Hugging Face 下载权重。"
            ),
            detector_id="IDEA-Research/grounding-dino-tiny",
            segmenter_id="facebook/sam2.1-hiera-tiny",
            needs_text_prompt=True,
            pip_hint="pip install 'scenepaste[grounding]'",
        ),
    ]


def get_model_spec(model_id: str) -> CutoutModelSpec:
    for spec in list_cutout_models():
        if spec.id == model_id:
            return spec
    raise KeyError(f"unknown cutout model: {model_id}")


def is_model_ready(model_id: str) -> bool:
    return bool(_READY.get(model_id))


def model_cache_dir() -> Path:
    """Local cache root for non-HF artifacts (HF still uses its own cache)."""
    root = Path.home() / ".scenepaste" / "models"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _progress(cb: Optional[ProgressCb], msg: str, pct: Optional[float] = None) -> None:
    """Emit progress. ``pct`` is 0..100, or None for indeterminate."""
    if not cb:
        return
    try:
        cb(msg, pct)
    except TypeError:
        cb(msg)


def apply_hf_endpoint(endpoint: str, progress: Optional[ProgressCb] = None) -> str:
    """Point huggingface_hub / transformers downloads at ``endpoint``.

    Must patch live ``huggingface_hub.constants`` because the module may already
    have been imported with the official URL.
    """
    endpoint = (endpoint or HF_OFFICIAL).rstrip("/")
    os.environ["HF_ENDPOINT"] = endpoint
    try:
        import huggingface_hub.constants as const

        const.ENDPOINT = endpoint
        const.HUGGINGFACE_CO_URL_TEMPLATE = (
            f"{endpoint}/{{repo_id}}/resolve/{{revision}}/{{filename}}"
        )
    except Exception:
        pass
    _progress(progress, f"下载源：{endpoint}", None)
    return endpoint


def normalize_proxy_env(progress: Optional[ProgressCb] = None) -> List[str]:
    """Fix common Clash/V2Ray proxy env that breaks huggingface_hub.

    ``socks://host:port`` is rejected by urllib3/httpx (\"Unknown scheme\").
    Convert to ``socks5h://`` when PySocks is installed; otherwise clear the
    broken variable so downloads can use the HF mirror directly.
    """
    keys = (
        "ALL_PROXY", "all_proxy",
        "HTTP_PROXY", "http_proxy",
        "HTTPS_PROXY", "https_proxy",
    )
    notes: List[str] = []
    has_socks = False
    try:
        import socks  # noqa: F401
        has_socks = True
    except Exception:
        has_socks = False

    for key in keys:
        val = os.environ.get(key)
        if not val:
            continue
        low = val.strip().lower()
        if not low.startswith("socks://"):
            continue
        rest = val.split("://", 1)[1]
        if has_socks:
            new_val = "socks5h://" + rest
            os.environ[key] = new_val
            notes.append(f"{key}: socks:// → socks5h://")
        else:
            del os.environ[key]
            notes.append(f"{key}: 已清除无效 socks://（可 pip install PySocks，或直连镜像）")

    if notes:
        _progress(progress, "代理：" + "；".join(notes), None)
    return notes


def configure_hf_download(
    *,
    use_mirror: bool = True,
    progress: Optional[ProgressCb] = None,
    bypass_socks_proxy: bool = True,
) -> str:
    """Configure HF download endpoint + sanitize proxy env.

    When downloading via ``hf-mirror.com``, local Clash ``socks://`` proxies are
    cleared by default — they often break huggingface_hub (unknown scheme /
    dead tunnel) while the mirror is reachable directly.
    """
    normalize_proxy_env(progress)
    if use_mirror and bypass_socks_proxy:
        cleared = []
        for key in (
            "ALL_PROXY", "all_proxy",
            "HTTP_PROXY", "http_proxy",
            "HTTPS_PROXY", "https_proxy",
        ):
            val = os.environ.get(key) or ""
            if "socks" in val.lower():
                del os.environ[key]
                cleared.append(key)
        if cleared:
            _progress(
                progress,
                "已临时忽略 SOCKS 代理以下载镜像：" + ", ".join(cleared),
                None,
            )
    if use_mirror:
        return apply_hf_endpoint(HF_MIRROR_DEFAULT, progress)
    existing = (os.environ.get("HF_ENDPOINT") or "").rstrip("/")
    if existing and existing != HF_MIRROR_DEFAULT:
        return apply_hf_endpoint(existing, progress)
    return apply_hf_endpoint(HF_OFFICIAL, progress)


def _is_network_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    needles = (
        "connection reset", "connection refused", "timed out", "timeout",
        "name or service not known", "temporary failure", "network is unreachable",
        "max retries exceeded", "proxy error", "ssl", "errno 104", "errno 110",
        "failed to establish", "huggingface.co", "unknown scheme", "socks://",
    )
    return any(n in text for n in needles)


def ensure_model(
    model_id: str,
    progress: Optional[ProgressCb] = None,
    *,
    use_hf_mirror: bool = True,
) -> CutoutModelSpec:
    """Download weights if needed and load the model into memory.

    Safe to call repeatedly; subsequent calls reuse the loaded session.
    ``use_hf_mirror`` applies to GroundingSAM2 Hugging Face downloads.
    """
    spec = get_model_spec(model_id)
    with _LOCK:
        if _READY.get(model_id) and model_id in _LOADED:
            _progress(progress, f"已加载：{spec.title}")
            return spec

    if spec.family == "rembg":
        _ensure_rembg(spec, progress)
    elif spec.family == "sam2_point":
        try:
            _ensure_sam2_point(spec, progress, use_hf_mirror=use_hf_mirror)
        except Exception as exc:
            if use_hf_mirror or not _is_network_error(exc):
                msg = str(exc)
                if "unknown scheme" in msg.lower() or "socks://" in msg.lower():
                    raise RuntimeError(
                        "代理地址 socks:// 不被支持。请重试，或 unset ALL_PROXY；"
                        f"或 pip install PySocks。\n原始错误：{exc}"
                    ) from exc
                raise
            _progress(progress, f"官方源失败，改用镜像重试：{exc}", None)
            normalize_proxy_env(progress)
            _ensure_sam2_point(spec, progress, use_hf_mirror=True)
    elif spec.family == "grounding_sam2":
        try:
            _ensure_grounding_sam2(spec, progress, use_hf_mirror=use_hf_mirror)
        except Exception as exc:
            # Auto-fallback: official failed → mirror; or bad socks proxy → clear + mirror
            if use_hf_mirror or not _is_network_error(exc):
                # Enrich socks / proxy errors with a clear hint
                msg = str(exc)
                if "unknown scheme" in msg.lower() or "socks://" in msg.lower():
                    raise RuntimeError(
                        "代理地址 socks:// 不被支持。ScenePaste 已尝试修复；"
                        "请重试，或执行：unset ALL_PROXY all_proxy "
                        "HTTP_PROXY HTTPS_PROXY；或 pip install PySocks。"
                        f"\n原始错误：{exc}"
                    ) from exc
                raise
            _progress(progress, f"官方源失败，改用镜像重试：{exc}", None)
            normalize_proxy_env(progress)
            _ensure_grounding_sam2(spec, progress, use_hf_mirror=True)
    else:
        raise ValueError(f"unsupported family: {spec.family}")

    with _LOCK:
        _READY[model_id] = True
    _progress(progress, f"就绪：{spec.title}")
    return spec


def predict_cutout(
    bgr: np.ndarray,
    model_id: str,
    *,
    text_prompt: str = "",
    simplify_eps: float = 0.004,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Run the selected model; return ``(polygon Nx2|None, alpha HxW|None)``.

    Call :func:`ensure_model` first (or this will raise if not ready).
    For GroundingSAM2, ``text_prompt`` should be the class name (e.g. ``person``).
    For ``sam2_click``, prefer :func:`predict_from_point` with a click coordinate.
    """
    if not is_model_ready(model_id):
        ensure_model(model_id)
    spec = get_model_spec(model_id)
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] < 3:
        raise ValueError("expected BGR image")

    if spec.family == "rembg":
        return _predict_rembg(bgr, model_id, simplify_eps=simplify_eps)
    if spec.family == "grounding_sam2":
        return _predict_grounding_sam2(
            bgr, model_id, text_prompt=text_prompt, simplify_eps=simplify_eps,
        )
    if spec.family == "sam2_point":
        # Center click fallback when no explicit point is given
        h, w = bgr.shape[:2]
        return predict_from_point(
            bgr, w * 0.5, h * 0.5, model_id=model_id, simplify_eps=simplify_eps,
        )
    raise ValueError(f"unsupported family: {spec.family}")


def predict_from_point(
    bgr: np.ndarray,
    x: float,
    y: float,
    *,
    model_id: str = "sam2_click",
    positive: bool = True,
    simplify_eps: float = 0.004,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Segment the object under ``(x, y)`` with SAM2; return polygon + alpha."""
    if not is_model_ready(model_id):
        ensure_model(model_id)
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] < 3:
        raise ValueError("expected BGR image")
    bundle = _LOADED.get(model_id)
    if bundle is None:
        raise RuntimeError(f"model not loaded: {model_id}")
    return _predict_sam_point(
        bundle, bgr, float(x), float(y), positive=positive, simplify_eps=simplify_eps,
    )


# ---------------------------------------------------------------------------
# rembg
# ---------------------------------------------------------------------------

def _format_bytes(n: float) -> str:
    n = float(max(0, n))
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= div:
            return f"{n / div:.1f}{unit}"
    return f"{int(n)}B"


def _tqdm_with_callback(cb: Optional[ProgressCb], label: str):
    """tqdm subclass that reports percent to ``cb`` (HF / pooch / rembg)."""
    from tqdm.auto import tqdm as _tqdm

    class _ProgressTqdm(_tqdm):
        _last_emit = 0.0
        _last_pct_i = -1

        def update(self, n=1):
            out = super().update(n)
            if not cb or not self.total:
                return out
            total = float(self.total)
            cur = float(min(self.n, self.total))
            pct = min(100.0, 100.0 * cur / total) if total else 0.0
            pct_i = int(pct)
            now = time.monotonic()
            # Throttle UI flood; always emit near start / every 1% / done
            if (
                pct_i == self._last_pct_i
                and pct < 100.0
                and (now - self._last_emit) < 0.2
                and self.n > 0
            ):
                return out
            self._last_emit = now
            self._last_pct_i = pct_i
            desc = (getattr(self, "desc", None) or label or "下载").strip()
            _progress(
                cb,
                f"{desc}  {_format_bytes(cur)} / {_format_bytes(total)}",
                pct,
            )
            return out

    return _ProgressTqdm


@contextmanager
def _patch_tqdm_progress(progress: Optional[ProgressCb], label: str) -> Iterator[None]:
    """Route terminal tqdm bars (pooch / rembg) into ``progress`` callbacks.

    rembg uses ``pooch.retrieve(..., progressbar=True)``. pooch holds its own
    ``from tqdm.auto import tqdm`` reference, so we must patch that module too.
    """
    if not progress:
        yield
        return

    try:
        ProgressTqdm = _tqdm_with_callback(progress, label)
    except ImportError:
        # tqdm is optional outside download environments; rembg still works.
        _progress(progress, f"{label}：下载中…", None)
        yield
        return

    restored: List[Tuple[object, str, object]] = []

    def _swap(mod: object, name: str) -> None:
        if mod is None or not hasattr(mod, name):
            return
        old = getattr(mod, name)
        if old is None:
            return
        restored.append((mod, name, old))
        setattr(mod, name, ProgressTqdm)

    try:
        import tqdm as tqdm_mod
        import tqdm.auto as tqdm_auto

        _swap(tqdm_mod, "tqdm")
        _swap(tqdm_auto, "tqdm")
    except Exception:
        pass
    try:
        import pooch.downloaders as pooch_dl

        _swap(pooch_dl, "tqdm")
    except Exception:
        pass

    try:
        yield
    finally:
        for mod, name, old in restored:
            try:
                setattr(mod, name, old)
            except Exception:
                pass


def _hf_prefetch(repo_id: str, progress: Optional[ProgressCb], label: str) -> None:
    """Download a HF repo into cache with progress bars, then local loads are fast."""
    _progress(progress, f"{label}：准备下载 {repo_id} …", None)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        _progress(progress, f"{label}：将由 transformers 拉取 {repo_id}", None)
        return
    try:
        snapshot_download(
            repo_id=repo_id,
            tqdm_class=_tqdm_with_callback(progress, label),
        )
        _progress(progress, f"{label}：缓存就绪", 100.0)
    except Exception as exc:
        # Fall through to from_pretrained which may still succeed / retry
        _progress(progress, f"{label}：预下载跳过（{exc}），改为直接加载…", None)


def _ensure_rembg(spec: CutoutModelSpec, progress: Optional[ProgressCb]) -> None:
    try:
        from rembg import new_session
    except ImportError as exc:
        hint = spec.pip_hint or "pip install 'scenepaste[auto]'"
        raise ImportError(f"需要 rembg。请运行：{hint}") from exc

    cache = Path.home() / ".u2net" / f"{spec.rembg_name}.onnx"
    if cache.is_file() and cache.stat().st_size > 0:
        _progress(progress, f"加载本地 rembg 权重 {spec.rembg_name} …", None)
    else:
        _progress(
            progress,
            f"下载 rembg 模型 {spec.rembg_name} → ~/.u2net/ …",
            0.0,
        )

    # rembg → pooch tqdm only prints to the terminal unless we patch it
    with _patch_tqdm_progress(progress, f"rembg {spec.rembg_name}"):
        session = new_session(spec.rembg_name)
    _progress(progress, f"rembg {spec.rembg_name} 已加载", 100.0)
    with _LOCK:
        _LOADED[spec.id] = session


def _predict_rembg(
    bgr: np.ndarray,
    model_id: str,
    *,
    simplify_eps: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    from rembg import remove as rembg_remove

    session = _LOADED.get(model_id)
    if session is None:
        ensure_model(model_id)
        session = _LOADED[model_id]
    rgb = bgr[:, :, :3][:, :, ::-1].copy()
    out = rembg_remove(rgb, session=session)
    if out is None or out.ndim != 3 or out.shape[2] < 4:
        raise RuntimeError("rembg did not return RGBA")
    alpha = np.asarray(out[:, :, 3], dtype=np.uint8)
    return alpha_to_polygon(alpha, simplify_eps), alpha


# ---------------------------------------------------------------------------
# Grounding DINO + SAM2 / SAM2 point click (Hugging Face)
# ---------------------------------------------------------------------------

@dataclass
class _SamPointBundle:
    segmenter: object
    segmenter_processor: object
    device: str
    use_sam2: bool
    segmenter_id: str = ""


@dataclass
class _GroundingBundle:
    detector: object
    detector_processor: object
    segmenter: object
    segmenter_processor: object
    device: str
    use_sam2: bool
    detector_id: str = ""
    segmenter_id: str = ""
    extras: dict = field(default_factory=dict)


def _pick_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _load_sam_segmenter(seg_id: str, progress: Optional[ProgressCb]):
    """Load SAM2 (preferred) or classic SAM. Returns (model, processor, use_sam2, seg_id)."""
    device = _pick_device()
    try:
        from transformers import Sam2Model, Sam2Processor

        _hf_prefetch(seg_id, progress, "SAM2")
        _progress(progress, f"加载 SAM2 到内存（{device}）…", None)
        try:
            processor = Sam2Processor.from_pretrained(seg_id)
            model = Sam2Model.from_pretrained(seg_id)
        except Exception:
            alt = "facebook/sam2-hiera-tiny"
            _progress(progress, f"改用 {alt} …", None)
            _hf_prefetch(alt, progress, "SAM2")
            processor = Sam2Processor.from_pretrained(alt)
            model = Sam2Model.from_pretrained(alt)
            seg_id = alt
        model.to(device)
        model.eval()
        return model, processor, True, seg_id, device
    except Exception:
        from transformers import SamModel, SamProcessor

        alt = "facebook/sam-vit-base"
        _progress(progress, f"无 SAM2，改用 {alt} …", None)
        _hf_prefetch(alt, progress, "SAM")
        processor = SamProcessor.from_pretrained(alt)
        model = SamModel.from_pretrained(alt)
        model.to(device)
        model.eval()
        return model, processor, False, alt, device


def _ensure_sam2_point(
    spec: CutoutModelSpec,
    progress: Optional[ProgressCb],
    *,
    use_hf_mirror: bool = True,
) -> None:
    configure_hf_download(use_mirror=use_hf_mirror, progress=progress)
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        hint = spec.pip_hint or "pip install 'scenepaste[grounding]'"
        raise ImportError(f"SAM2 点选需要 torch + transformers。请运行：{hint}") from exc

    # Reuse segmenter already loaded by GroundingSAM2 when possible.
    other = _LOADED.get("grounding_sam2")
    if isinstance(other, _GroundingBundle) and other.segmenter is not None:
        _progress(progress, "复用已加载的 SAM2 权重…", 80.0)
        bundle = _SamPointBundle(
            segmenter=other.segmenter,
            segmenter_processor=other.segmenter_processor,
            device=other.device,
            use_sam2=other.use_sam2,
            segmenter_id=other.segmenter_id or spec.segmenter_id,
        )
        with _LOCK:
            _LOADED[spec.id] = bundle
        return

    model, processor, use_sam2, seg_id, device = _load_sam_segmenter(
        spec.segmenter_id, progress,
    )
    bundle = _SamPointBundle(
        segmenter=model,
        segmenter_processor=processor,
        device=device,
        use_sam2=use_sam2,
        segmenter_id=seg_id,
    )
    with _LOCK:
        _LOADED[spec.id] = bundle
    _progress(progress, "SAM2 点选模型已就绪", 100.0)


def _binary_mask_from_outputs(processor, outputs, inputs, height: int, width: int):
    """Pick the best mask from SAM/SAM2 outputs → uint8 HxW {0,1}."""
    post = getattr(processor, "post_process_masks", None)
    scores = getattr(outputs, "iou_scores", None)
    if callable(post):
        try:
            masks = post(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs.get("reshaped_input_sizes", inputs["original_sizes"]).cpu(),
            )
            m = masks[0]
            if hasattr(m, "detach"):
                m = m.detach().float().cpu().numpy()
            m = np.asarray(m)
            # shapes: (n_masks, H, W) or (1, n_masks, H, W) …
            while m.ndim > 3:
                m = m[0]
            if m.ndim == 3:
                if scores is not None:
                    sc = scores.detach().float().cpu().numpy().reshape(-1)
                    idx = int(np.argmax(sc[: m.shape[0]]))
                else:
                    # largest area
                    idx = int(np.argmax(m.reshape(m.shape[0], -1).sum(axis=1)))
                m = m[idx]
            binary = m > 0.5 if m.dtype != np.bool_ else m
            if binary.any():
                if binary.shape[0] != height or binary.shape[1] != width:
                    binary = cv2.resize(
                        binary.astype(np.uint8), (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                return binary.astype(np.uint8)
        except Exception:
            pass

    masks = getattr(outputs, "pred_masks", None)
    if masks is None:
        return None
    m = masks.detach().float().cpu().numpy()
    while m.ndim > 2:
        m = m[0]
    if m.ndim != 2:
        return None
    if m.shape[0] != height or m.shape[1] != width:
        m = cv2.resize(m, (width, height), interpolation=cv2.INTER_LINEAR)
    binary = m > 0.0
    if not binary.any():
        binary = (1.0 / (1.0 + np.exp(-np.clip(m, -50, 50)))) > 0.5
    if not binary.any():
        return None
    return binary.astype(np.uint8)


def _predict_sam_point(
    bundle,
    bgr: np.ndarray,
    x: float,
    y: float,
    *,
    positive: bool = True,
    simplify_eps: float = 0.004,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    import torch
    from PIL import Image

    rgb = cv2.cvtColor(bgr[:, :, :3], cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    h, w = rgb.shape[:2]
    x = float(np.clip(x, 0, w - 1))
    y = float(np.clip(y, 0, h - 1))
    label = 1 if positive else 0

    processor = bundle.segmenter_processor
    model = bundle.segmenter
    if bundle.use_sam2:
        input_points = [[[[x, y]]]]
        input_labels = [[[label]]]
    else:
        input_points = [[[x, y]]]
        input_labels = [[label]]

    try:
        inputs = processor(
            images=pil,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        )
    except Exception:
        inputs = processor(
            pil,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        )

    inputs = {
        k: v.to(bundle.device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }
    with torch.no_grad():
        outputs = model(**inputs)

    mask = _binary_mask_from_outputs(processor, outputs, inputs, h, w)
    if mask is None:
        return None, None
    alpha = (mask.astype(np.uint8) * 255)
    return alpha_to_polygon(alpha, simplify_eps), alpha


def _ensure_grounding_sam2(
    spec: CutoutModelSpec,
    progress: Optional[ProgressCb],
    *,
    use_hf_mirror: bool = True,
) -> None:
    configure_hf_download(use_mirror=use_hf_mirror, progress=progress)
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError as exc:
        hint = spec.pip_hint or "pip install 'scenepaste[grounding]'"
        raise ImportError(
            f"GroundingSAM2 需要 torch + transformers。请运行：{hint}"
        ) from exc

    device = _pick_device()
    det_id = spec.detector_id
    seg_id = spec.segmenter_id

    _hf_prefetch(det_id, progress, "检测模型")
    _progress(progress, f"加载检测模型到内存（{device}）…", None)
    detector_processor = AutoProcessor.from_pretrained(det_id)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(det_id)
    detector.to(device)
    detector.eval()
    _progress(progress, "检测模型已加载", 50.0)

    use_sam2 = True
    segmenter = segmenter_processor = None
    # Prefer SAM2; fall back to classic SAM if transformers build lacks Sam2.
    try:
        from transformers import Sam2Model, Sam2Processor

        _hf_prefetch(seg_id, progress, "分割模型")
        _progress(progress, f"加载分割模型到内存（{device}）…", None)
        try:
            segmenter_processor = Sam2Processor.from_pretrained(seg_id)
            segmenter = Sam2Model.from_pretrained(seg_id)
        except Exception:
            alt = "facebook/sam2-hiera-tiny"
            _progress(progress, f"改用 {alt} …", None)
            _hf_prefetch(alt, progress, "分割模型")
            segmenter_processor = Sam2Processor.from_pretrained(alt)
            segmenter = Sam2Model.from_pretrained(alt)
            seg_id = alt
    except Exception:
        use_sam2 = False
        from transformers import SamModel, SamProcessor

        alt = "facebook/sam-vit-base"
        _progress(progress, f"当前 transformers 无 SAM2，改用 {alt} …", None)
        _hf_prefetch(alt, progress, "分割模型")
        segmenter_processor = SamProcessor.from_pretrained(alt)
        segmenter = SamModel.from_pretrained(alt)
        seg_id = alt

    segmenter.to(device)
    segmenter.eval()
    _progress(progress, "分割模型已加载", 95.0)

    bundle = _GroundingBundle(
        detector=detector,
        detector_processor=detector_processor,
        segmenter=segmenter,
        segmenter_processor=segmenter_processor,
        device=device,
        use_sam2=use_sam2,
        detector_id=det_id,
        segmenter_id=seg_id,
    )
    with _LOCK:
        _LOADED[spec.id] = bundle


def _normalize_prompt(text: str) -> str:
    """Grounding DINO works best with lowercase phrases ending in a period."""
    t = (text or "").strip().lower()
    if not t:
        t = "object"
    if not t.endswith("."):
        t = t + "."
    return t


def _predict_grounding_sam2(
    bgr: np.ndarray,
    model_id: str,
    *,
    text_prompt: str,
    simplify_eps: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    import torch
    from PIL import Image

    bundle = _LOADED.get(model_id)
    if bundle is None:
        ensure_model(model_id)
        bundle = _LOADED[model_id]
    assert isinstance(bundle, _GroundingBundle)

    rgb = cv2.cvtColor(bgr[:, :, :3], cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    prompt = _normalize_prompt(text_prompt)
    h, w = rgb.shape[:2]

    inputs = bundle.detector_processor(images=pil, text=prompt, return_tensors="pt")
    inputs = {k: v.to(bundle.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bundle.detector(**inputs)

    results = _post_process_grounding(
        bundle.detector_processor, outputs, inputs, (h, w),
    )
    boxes = results.get("boxes")
    if boxes is None or len(boxes) == 0:
        return None, None

    scores = results.get("scores")
    if scores is not None and len(scores) == len(boxes):
        idx = int(torch.argmax(scores).item())
    else:
        idx = 0
    box = boxes[idx]
    if hasattr(box, "detach"):
        box = box.detach().float().cpu().numpy()
    else:
        box = np.asarray(box, dtype=np.float32)

    mask = _sam_mask_from_box(bundle, pil, box, h, w)
    if mask is None:
        return None, None
    alpha = (mask.astype(np.uint8) * 255)
    return alpha_to_polygon(alpha, simplify_eps), alpha


def _post_process_grounding(processor, outputs, inputs, target_size):
    """Compatibility shim across transformers Grounding DINO APIs."""
    kwargs_list = [
        dict(
            outputs=outputs,
            input_ids=inputs["input_ids"],
            box_threshold=0.25,
            text_threshold=0.25,
            target_sizes=[target_size],
        ),
        dict(
            outputs=outputs,
            input_ids=inputs["input_ids"],
            threshold=0.25,
            text_threshold=0.25,
            target_sizes=[target_size],
        ),
    ]
    last_exc = None
    for kwargs in kwargs_list:
        try:
            return processor.post_process_grounded_object_detection(**kwargs)[0]
        except TypeError as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"Grounding DINO post_process failed: {last_exc}")


def _sam_mask_from_box(
    bundle: _GroundingBundle,
    pil_image,
    box_xyxy: np.ndarray,
    height: int,
    width: int,
) -> Optional[np.ndarray]:
    import torch

    box = np.asarray(box_xyxy, dtype=np.float32).reshape(1, 4)
    processor = bundle.segmenter_processor
    model = bundle.segmenter

    try:
        if bundle.use_sam2:
            inputs = processor(pil_image, input_boxes=[box.tolist()], return_tensors="pt")
        else:
            inputs = processor(pil_image, input_boxes=[[box.tolist()]], return_tensors="pt")
    except Exception:
        inputs = processor(images=pil_image, input_boxes=[[box.tolist()]], return_tensors="pt")

    inputs = {
        k: v.to(bundle.device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }
    with torch.no_grad():
        outputs = model(**inputs)

    # Prefer official post_process_masks when available
    post = getattr(processor, "post_process_masks", None)
    if callable(post):
        try:
            masks = post(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs.get("reshaped_input_sizes", inputs["original_sizes"]).cpu(),
            )
            # masks: list of tensors
            m = masks[0]
            if hasattr(m, "detach"):
                m = m.detach().float().cpu().numpy()
            while isinstance(m, (list, tuple)):
                m = m[0]
            m = np.asarray(m)
            while m.ndim > 2:
                m = m[0]
            binary = m > 0.5 if m.dtype != np.bool_ else m
            if binary.any():
                if binary.shape[0] != height or binary.shape[1] != width:
                    binary = cv2.resize(
                        binary.astype(np.uint8), (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                return binary.astype(np.uint8)
        except Exception:
            pass

    masks = getattr(outputs, "pred_masks", None)
    if masks is None:
        return None
    m = masks.detach().float().cpu().numpy()
    while m.ndim > 2:
        m = m[0]
    if m.ndim != 2:
        return None
    if m.shape[0] != height or m.shape[1] != width:
        m = cv2.resize(m, (width, height), interpolation=cv2.INTER_LINEAR)
    binary = m > 0.0
    if not binary.any():
        binary = (1.0 / (1.0 + np.exp(-np.clip(m, -50, 50)))) > 0.5
    if not binary.any():
        return None
    return binary.astype(np.uint8)


def reset_loaded_models() -> None:
    """Test helper: drop cached sessions."""
    with _LOCK:
        _LOADED.clear()
        _READY.clear()
