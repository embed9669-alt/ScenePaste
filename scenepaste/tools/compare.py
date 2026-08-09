"""Real-vs-synthetic dataset comparison report and HTML dashboard."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Optional, Sequence

from ..core.distribution import compare_profiles, learn_distribution_profile
from ..core.similarity import compare_embedding_domains, iter_dataset_images


def compare_datasets(real: Path, synthetic: Path, embedding_limit: int = 1000, embedding_backend: str = "cv-lite-v1") -> dict:
    real = Path(real); synthetic = Path(synthetic)
    real_profile = learn_distribution_profile(real)
    synth_profile = learn_distribution_profile(synthetic)
    profile_cmp = compare_profiles(real_profile, synth_profile)
    real_images = [p for _s, p in iter_dataset_images(real)]
    synth_images = [p for _s, p in iter_dataset_images(synthetic)]
    visual = compare_embedding_domains(real_images, synth_images, limit=embedding_limit, backend=embedding_backend)
    suggestions = []
    if profile_cmp.get("class_total_variation", 0) > 0.15:
        suggestions.append("Class distribution differs noticeably; consider a mixed/real distribution profile.")
    pd = profile_cmp.get("placement_histogram_distance")
    if pd is not None and pd > 0.20:
        suggestions.append("Scale/position distribution differs; increase profile_strength or refine scene templates/zones.")
    overlap_gap = (profile_cmp.get("metric_distances") or {}).get("overlap_iou")
    if overlap_gap is not None and overlap_gap > 0.20:
        suggestions.append("Crowding/overlap distribution differs; tune overlap-aware profiles or scene relation constraints.")
    centroid = visual.get("centroid_cosine_similarity")
    if centroid is not None and centroid < 0.90:
        suggestions.append("Global appearance/domain gap is large; tune augmentation recipe, background pool, or blending.")
    nn = visual.get("synthetic_to_real_mean_nn_similarity")
    if nn is not None and nn > 0.995:
        suggestions.append("Synthetic images are extremely close to real samples; check for leakage or over-reuse.")
    return {
        "schema": "scenepaste/domain-comparison", "version": 1,
        "real": str(real.resolve()), "synthetic": str(synthetic.resolve()),
        "distribution": profile_cmp, "visual": visual, "suggestions": suggestions,
    }


def render_compare_html(report: dict) -> str:
    d = report.get("distribution", {}); v = report.get("visual", {}); suggestions = report.get("suggestions", [])
    list_html = "".join(f"<li>{html.escape(str(x))}</li>" for x in suggestions) or "<li>No major automatic warnings.</li>"
    def fmt(x):
        return "n/a" if x is None else f"{float(x):.3f}"
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>ScenePaste Real vs Synthetic</title>
<style>body{{font-family:Inter,Segoe UI,Arial;background:#101215;color:#e9edf1;margin:0}}.wrap{{max-width:1100px;margin:auto;padding:28px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}}.card{{background:#181c21;border:1px solid #2a3038;border-radius:12px;padding:16px}}.big{{font-size:28px;font-weight:700;margin-top:6px}}.muted{{color:#929aa4}}li{{margin:8px 0}}</style></head><body><div class="wrap">
<h1>ScenePaste — Real vs Synthetic</h1><p class="muted">Real: {html.escape(report.get('real',''))}<br>Synthetic: {html.escape(report.get('synthetic',''))}</p>
<div class="grid">
<div class="card">Class TV distance<div class="big">{fmt(d.get('class_total_variation'))}</div><div class="muted">0 = same distribution</div></div>
<div class="card">Placement distance<div class="big">{fmt(d.get('placement_histogram_distance'))}</div><div class="muted">scale / x / bottom-y histograms</div></div>
<div class="card">Appearance centroid<div class="big">{fmt(v.get('centroid_cosine_similarity'))}</div><div class="muted">1 = visually similar domains</div></div>
<div class="card">Synthetic→real NN<div class="big">{fmt(v.get('synthetic_to_real_mean_nn_similarity'))}</div><div class="muted">mean nearest visual similarity</div></div>
<div class="card">Real uniqueness<div class="big">{fmt((v.get('real_diversity') or {}).get('mean_uniqueness'))}</div></div>
<div class="card">Synthetic uniqueness<div class="big">{fmt((v.get('synthetic_diversity') or {}).get('mean_uniqueness'))}</div></div>
</div><h2>Recommendations</h2><div class="card"><ul>{list_html}</ul></div>
<h2>Raw report</h2><div class="card"><pre>{html.escape(json.dumps(report, ensure_ascii=False, indent=2))}</pre></div>
</div></body></html>'''


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare a real dataset with a synthetic ScenePaste dataset")
    parser.add_argument("real", type=Path); parser.add_argument("synthetic", type=Path)
    parser.add_argument("--embedding-limit", type=int, default=1000)
    parser.add_argument("--embedding-backend", choices=("cv-lite-v1","clip","dinov2"), default="cv-lite-v1")
    parser.add_argument("-o", "--output", type=Path, default=None, help="output directory")
    args = parser.parse_args(argv)
    out = args.output or (args.synthetic / "comparison")
    out.mkdir(parents=True, exist_ok=True)
    report = compare_datasets(args.real, args.synthetic, args.embedding_limit, args.embedding_backend)
    json_path = out / "real_vs_synthetic.json"; html_path = out / "real_vs_synthetic.html"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(render_compare_html(report), encoding="utf-8")
    print(html_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
