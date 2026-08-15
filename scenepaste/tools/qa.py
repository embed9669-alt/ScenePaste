"""ScenePaste dataset QA report + standalone HTML dashboard."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..core.distribution import DistributionProfile, compare_profiles, learn_distribution_profile
from ..core.models import IMAGE_SUFFIXES
from ..core.similarity import diversity_summary, iter_dataset_images
from .analyze import count_synthetic
from .leakage import detect_split_leakage


def _image_paths(root: Path):
    for split in ("train", "val", "test"):
        d = root / "images" / split
        if d.exists():
            yield from sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _phash64(image: np.ndarray) -> int:
    """Return a compact 64-bit perceptual hash using low-frequency DCT."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8].reshape(-1)
    # Exclude DC from threshold so global brightness changes do not dominate.
    median = float(np.median(dct[1:])) if len(dct) > 1 else float(dct[0])
    bits = dct > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return int(value)


def _hamming64(a: int, b: int) -> int:
    return bin(int(a) ^ int(b)).count("1")


def _integrity(root: Path, duplicate_limit: int = 5000, near_duplicate_threshold: int = 6) -> dict:
    unreadable = 0
    duplicates = 0
    seen = {}
    checked = 0
    duplicate_examples = []
    near_pairs = []
    near_count = 0
    # Eight exact 8-bit bands guarantee that any 64-bit hashes with Hamming
    # distance <=7 share at least one band, avoiding an O(N^2) full scan.
    buckets = [dict() for _ in range(8)]
    phashes = []
    paths = []
    threshold = max(0, min(7, int(near_duplicate_threshold)))
    for p in _image_paths(root):
        try:
            data = p.read_bytes()
            arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            arr = None; data = b""
        if arr is None:
            unreadable += 1
            continue
        if checked < duplicate_limit:
            digest = hashlib.sha1(data).hexdigest()
            is_exact_duplicate = digest in seen
            if is_exact_duplicate:
                duplicates += 1
                if len(duplicate_examples) < 10:
                    duplicate_examples.append([seen[digest], str(p)])
            else:
                seen[digest] = str(p)

            ph = _phash64(arr)
            candidates = set()
            for band in range(8):
                key = (ph >> (band * 8)) & 0xFF
                candidates.update(buckets[band].get(key, []))
            matched = None
            best_distance = 65
            # Keep exact byte duplicates and perceptual near-duplicates as two
            # separate QA signals. Differently encoded copies still count as
            # perceptual duplicates because their byte hashes differ.
            if not is_exact_duplicate:
                for idx in candidates:
                    distance = _hamming64(ph, phashes[idx])
                    if distance <= threshold and distance < best_distance:
                        best_distance = distance
                        matched = idx
            if matched is not None:
                near_count += 1
                if len(near_pairs) < 20:
                    near_pairs.append({"a": paths[matched], "b": str(p), "hamming": best_distance})
            idx = len(phashes)
            phashes.append(ph); paths.append(str(p))
            for band in range(8):
                key = (ph >> (band * 8)) & 0xFF
                buckets[band].setdefault(key, []).append(idx)
            checked += 1
    phash_unique = len(set(phashes))
    diversity_ratio = (phash_unique / checked) if checked else None
    return {
        "unreadable_images": unreadable,
        "duplicate_images": duplicates,
        "duplicate_checked": checked,
        "duplicate_examples": duplicate_examples,
        "near_duplicate_images": near_count,
        "near_duplicate_threshold": threshold,
        "near_duplicate_examples": near_pairs,
        "perceptual_unique": phash_unique,
        "perceptual_diversity_ratio": diversity_ratio,
    }


def _profile_summary(profile: DistributionProfile) -> dict:
    classes = {}
    for name, row in profile.classes.items():
        classes[name] = {
            "count": int(row.get("count", 0)),
            "height_hist": row.get("height", {}).get("counts", []),
            "center_x_hist": row.get("center_x", {}).get("counts", []),
            "bottom_y_hist": row.get("bottom_y", {}).get("counts", []),
            "overlap_iou_hist": row.get("overlap_iou", {}).get("counts", []),
            "visible_shape_fraction_hist": row.get("visible_shape_fraction", {}).get("counts", []),
        }
    return {"image_count": profile.data.get("image_count", 0),
            "object_count_total": profile.data.get("object_count_total", 0),
            "object_count": profile.data.get("object_count", {}), "classes": classes}


def build_qa_report(root: Path, duplicate_limit: int = 5000,
                    target_profile: Optional[Path] = None,
                    near_duplicate_threshold: int = 6,
                    embedding_limit: int = 500,
                    leakage_embedding_threshold: float = 0.995,
                    embedding_backend: str = "cv-lite-v1") -> dict:
    root = Path(root)
    base = count_synthetic(root)
    integrity = _integrity(root, duplicate_limit=duplicate_limit, near_duplicate_threshold=near_duplicate_threshold)
    warnings = list(base.get("warnings", []))
    if integrity["unreadable_images"]:
        warnings.append(f"unreadable images: {integrity['unreadable_images']}")
    if integrity["duplicate_images"]:
        warnings.append(f"duplicate images: {integrity['duplicate_images']} / checked {integrity['duplicate_checked']}")
    if integrity.get("near_duplicate_images", 0):
        warnings.append(
            f"near-duplicate images: {integrity['near_duplicate_images']} / checked {integrity['duplicate_checked']} "
            f"(pHash Hamming <= {integrity['near_duplicate_threshold']})"
        )
    learned = None
    try:
        learned_profile = learn_distribution_profile(root)
        learned = _profile_summary(learned_profile)
    except Exception:
        learned_profile = None
    target_path = Path(target_profile) if target_profile else (root / "target_distribution_profile.json")
    comparison = None
    if target_path.exists() and learned_profile is not None:
        try:
            comparison = compare_profiles(DistributionProfile.load(target_path), learned_profile)
            target_images = int(DistributionProfile.load(target_path).data.get("image_count", 0))
            generated_images = int((learned_profile.data if learned_profile else {}).get("image_count", 0))
            # Distribution drift is statistically noisy on tiny smoke-test datasets.
            if min(target_images, generated_images) >= 50:
                if comparison["class_total_variation"] > 0.20:
                    warnings.append(f"target/generated class distribution drift: {comparison['class_total_variation']:.3f}")
                pd = comparison.get("placement_histogram_distance")
                if pd is not None and pd > 0.25:
                    warnings.append(f"target/generated placement drift: {pd:.3f}")
            else:
                comparison["note"] = "drift warning suppressed for <50 images"
        except Exception as exc:
            comparison = {"error": str(exc)}
    diagnostics = None
    diagnostics_path = root / "latest_generation_diagnostics.json"
    if diagnostics_path.exists():
        try:
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        except Exception:
            diagnostics = None
    image_paths = [p for _split, p in iter_dataset_images(root)]
    diversity = diversity_summary(image_paths, limit=max(0, int(embedding_limit)), backend=embedding_backend) if embedding_limit else {"samples": 0}
    try:
        leakage = detect_split_leakage(
            root, phash_threshold=near_duplicate_threshold,
            embedding_threshold=float(leakage_embedding_threshold),
            embedding_limit_per_split=max(0, int(embedding_limit)),
            max_examples=20,
            embedding_backend=embedding_backend,
        )
    except Exception as exc:
        leakage = {"health": "unknown", "error": str(exc), "exact_cross_split": 0,
                   "near_cross_split": 0, "embedding_cross_split": 0}
    if leakage.get("exact_cross_split", 0):
        warnings.append(f"cross-split exact leakage: {leakage['exact_cross_split']}")
    if leakage.get("near_cross_split", 0):
        warnings.append(f"cross-split perceptual leakage: {leakage['near_cross_split']}")
    if leakage.get("embedding_cross_split", 0):
        warnings.append(f"cross-split visual-embedding leakage: {leakage['embedding_cross_split']}")
    return {
        "schema": "scenepaste/qa-report", "version": 2,
        "dataset": str(root.resolve()), "health": "ok" if not warnings else "warning",
        "warnings": warnings, "summary": base, "integrity": integrity,
        "distribution": learned, "target_comparison": comparison,
        "generation_diagnostics": diagnostics,
        "curation": {"diversity": diversity, "cross_split_leakage": leakage},
    }


def _bars(values, max_width=100):
    vals = [float(v) for v in values]
    m = max(vals) if vals else 0.0
    if m <= 0: return '<span class="muted">no data</span>'
    return '<div class="spark">' + ''.join(
        f'<i style="height:{max(2, int(v/m*max_width))}%" title="{int(v)}"></i>' for v in vals
    ) + '</div>'


def render_qa_html(report: dict) -> str:
    summary = report.get("summary", {})
    yolo = summary.get("yolo", {})
    integrity = report.get("integrity", {})
    reuse = summary.get("reuse", {})
    distro = report.get("distribution") or {}
    warnings = report.get("warnings", [])
    curation = report.get("curation", {})
    diagnostics = report.get("generation_diagnostics") or {}
    diversity = curation.get("diversity", {})
    leakage = curation.get("cross_split_leakage", {})
    class_rows = []
    classes = distro.get("classes", {})
    if classes:
        for name, row in sorted(classes.items(), key=lambda kv: -kv[1].get("count", 0)):
            class_rows.append(
                f"<tr><td>{html.escape(str(name))}</td><td>{row.get('count',0)}</td>"
                f"<td>{_bars(row.get('height_hist',[]))}</td>"
                f"<td>{_bars(row.get('center_x_hist',[]))}</td>"
                f"<td>{_bars(row.get('bottom_y_hist',[]))}</td>"
                f"<td>{_bars(row.get('overlap_iou_hist',[]))}</td>"
                f"<td>{_bars(row.get('visible_shape_fraction_hist',[]))}</td></tr>"
            )
    else:
        dist = yolo.get("by_class_id", {})
        class_rows = [
            f"<tr><td>class {k}</td><td>{v}</td><td colspan=5>—</td></tr>"
            for k, v in dist.items()
        ]
    scene_effect_counts = diagnostics.get("scene_effect_counts", {}) if isinstance(diagnostics, dict) else {}
    object_effect_counts = diagnostics.get("object_effect_counts", {}) if isinstance(diagnostics, dict) else {}
    generated_images = max(1, int(diagnostics.get("images", 0) or 0)) if isinstance(diagnostics, dict) else 1
    generated_objects = max(1, int(diagnostics.get("objects", 0) or 0)) if isinstance(diagnostics, dict) else 1

    def effect_rows(counts, denominator, unit):
        if not counts:
            return '<tr><td colspan="4" class="muted">No recorded effects.</td></tr>'
        rows = []
        for name, count in sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))):
            pct = 100.0 * int(count) / max(1, denominator)
            rows.append(
                f"<tr><td>{html.escape(str(name))}</td><td>{int(count)}</td>"
                f"<td>{pct:.1f}%</td><td>{html.escape(unit)}</td></tr>"
            )
        return ''.join(rows)

    scene_effect_rows = effect_rows(scene_effect_counts, generated_images, "images")
    object_effect_rows = effect_rows(object_effect_counts, generated_objects, "objects")

    comparison = report.get("target_comparison")
    comp_html = '<span class="muted">No target distribution profile</span>'
    if isinstance(comparison, dict) and "error" not in comparison:
        comp_html = (f"Class TV distance: <b>{comparison.get('class_total_variation',0):.3f}</b> · "
                     f"Placement distance: <b>{comparison.get('placement_histogram_distance') if comparison.get('placement_histogram_distance') is not None else 'n/a'}</b> · "
                     f"Matched classes: {comparison.get('matched_classes',0)}")
    warning_html = ''.join(f'<li>{html.escape(str(w))}</li>' for w in warnings) or '<li class="ok">No blocking QA warnings.</li>'
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>ScenePaste QA Dashboard</title>
<style>
body{{font-family:Inter,Segoe UI,Arial,sans-serif;margin:0;background:#101215;color:#e9edf1}} .wrap{{max-width:1280px;margin:auto;padding:28px}}
h1{{margin:0 0 4px}} .sub,.muted{{color:#9099a3}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:22px 0}}
.card{{background:#181c21;border:1px solid #2a3038;border-radius:12px;padding:16px}} .big{{font-size:28px;font-weight:700;margin-top:6px}}
table{{width:100%;border-collapse:collapse;background:#181c21;border-radius:12px;overflow:hidden}} th,td{{padding:11px;border-bottom:1px solid #2a3038;text-align:left}} th{{background:#20252c}}
.spark{{height:48px;display:flex;gap:2px;align-items:flex-end;min-width:160px}} .spark i{{display:block;flex:1;min-height:2px;background:#66b3ff;border-radius:2px 2px 0 0}}
ul{{background:#181c21;border:1px solid #2a3038;border-radius:12px;padding:16px 34px}} li{{margin:7px 0;color:#ffcf70}} li.ok{{color:#83d69b}} .okbadge{{color:#83d69b}} .warnbadge{{color:#ffcf70}}
code{{color:#9dd1ff}}
</style></head><body><div class="wrap">
<h1>ScenePaste QA Dashboard</h1><div class="sub">{html.escape(report.get('dataset',''))}</div>
<div class="grid">
<div class="card">Health<div class="big {'okbadge' if report.get('health')=='ok' else 'warnbadge'}">{html.escape(report.get('health','unknown').upper())}</div></div>
<div class="card">Images<div class="big">{summary.get('images',0)}</div></div>
<div class="card">Objects<div class="big">{yolo.get('total_objects', (distro or {}).get('object_count_total',0))}</div></div>
<div class="card">Invalid labels<div class="big">{yolo.get('invalid_lines',0)}</div></div>
<div class="card">Exact duplicates<div class="big">{integrity.get('duplicate_images',0)}</div><div class="muted">checked {integrity.get('duplicate_checked',0)}</div></div>
<div class="card">Near duplicates<div class="big">{integrity.get('near_duplicate_images',0)}</div><div class="muted">pHash ≤ {integrity.get('near_duplicate_threshold',6)}</div></div>
<div class="card">pHash diversity<div class="big">{(f"{integrity.get('perceptual_diversity_ratio')*100:.1f}%" if integrity.get('perceptual_diversity_ratio') is not None else 'n/a')}</div><div class="muted">perceptual unique / checked</div></div>
<div class="card">Embedding uniqueness<div class="big">{(f"{diversity.get('mean_uniqueness'):.3f}" if diversity.get('mean_uniqueness') is not None else 'n/a')}</div><div class="muted">{html.escape(str(diversity.get('backend', 'cv-lite-v1')))} nearest-neighbor distance</div></div>
<div class="card">Cross-split leakage<div class="big">{int(leakage.get('exact_cross_split',0))+int(leakage.get('near_cross_split',0))+int(leakage.get('embedding_cross_split',0))}</div><div class="muted">exact / pHash / embedding</div></div>
<div class="card">Source max reuse<div class="big">{reuse.get('max_source_reuse',0)}</div></div>
<div class="card">Background max reuse<div class="big">{reuse.get('max_background_reuse',0)}</div></div>
</div>
<h2>Warnings</h2><ul>{warning_html}</ul>
<h2>Target vs generated distribution</h2><div class="card">{comp_html}</div>
<h2>Class / scale / position distributions</h2>
<table><thead><tr><th>Class</th><th>Objects</th><th>Height</th><th>Center X</th><th>Bottom Y</th><th>Overlap IoU</th><th>Visible shape</th></tr></thead><tbody>{''.join(class_rows)}</tbody></table>
<h2>Appearance coverage</h2>
<div class="grid">
<div class="card"><h3>Object-level effects</h3><table><thead><tr><th>Effect</th><th>Applied</th><th>Coverage</th><th>Base</th></tr></thead><tbody>{object_effect_rows}</tbody></table></div>
<div class="card"><h3>Scene-level effects</h3><table><thead><tr><th>Effect</th><th>Applied</th><th>Coverage</th><th>Base</th></tr></thead><tbody>{scene_effect_rows}</tbody></table></div>
</div>
<h2>Generation diagnostics</h2><div class="card"><pre>{html.escape(json.dumps(diagnostics, ensure_ascii=False, indent=2))}</pre></div>
<h2>Data curation</h2><div class="card"><pre>{html.escape(json.dumps(curation, ensure_ascii=False, indent=2))}</pre></div>
<h2>Integrity</h2><div class="card"><pre>{html.escape(json.dumps(integrity, ensure_ascii=False, indent=2))}</pre></div>
</div></body></html>'''


def write_qa_dashboard(root: Path, html_path: Optional[Path] = None,
                       json_path: Optional[Path] = None, duplicate_limit: int = 5000,
                       target_profile: Optional[Path] = None,
                       near_duplicate_threshold: int = 6,
                       embedding_limit: int = 500,
                       leakage_embedding_threshold: float = 0.995,
                       embedding_backend: str = "cv-lite-v1") -> dict:
    report = build_qa_report(root, duplicate_limit=duplicate_limit, target_profile=target_profile,
                             near_duplicate_threshold=near_duplicate_threshold,
                             embedding_limit=embedding_limit,
                             leakage_embedding_threshold=leakage_embedding_threshold,
                             embedding_backend=embedding_backend)
    root = Path(root)
    html_path = Path(html_path) if html_path else root / "qa_dashboard.html"
    json_path = Path(json_path) if json_path else root / "qa_report.json"
    html_path.write_text(render_qa_html(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["html_path"] = str(html_path)
    report["json_path"] = str(json_path)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate a ScenePaste QA JSON + HTML dashboard")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--html", type=Path, default=None)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    parser.add_argument("--target-profile", type=Path, default=None)
    parser.add_argument("--duplicate-limit", type=int, default=5000)
    parser.add_argument("--near-duplicate-threshold", type=int, default=6,
                        help="pHash Hamming threshold; smaller is stricter")
    parser.add_argument("--embedding-limit", type=int, default=500,
                        help="max images per dataset/split for lightweight visual embedding QA")
    parser.add_argument("--leakage-embedding-threshold", type=float, default=0.995,
                        help="cross-split embedding cosine similarity threshold")
    parser.add_argument("--embedding-backend", choices=("cv-lite-v1","clip","dinov2"), default="cv-lite-v1")
    args = parser.parse_args(argv)
    report = write_qa_dashboard(args.dataset, args.html, args.json_path,
                                max(0, args.duplicate_limit), args.target_profile,
                                max(0, min(12, args.near_duplicate_threshold)),
                                max(0, args.embedding_limit),
                                args.leakage_embedding_threshold,
                                args.embedding_backend)
    print(f"QA {report['health']}: {report['html_path']}")
    for w in report.get("warnings", []): print(f"  ⚠ {w}")
    return 0 if report["health"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
