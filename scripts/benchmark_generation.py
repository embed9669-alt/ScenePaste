#!/usr/bin/env python3
"""Benchmark ScenePaste generation with 1/4/8 workers.

The script uses the bundled public sample assets, launches each run in a fresh
subprocess, samples total RSS (parent + children) with psutil when available,
and prints Markdown suitable for docs/PERFORMANCE.md.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


def _monitor_rss(proc: subprocess.Popen, result: dict) -> None:
    try:
        import psutil
    except Exception:
        result["peak_rss"] = None
        return
    p = psutil.Process(proc.pid)
    peak = 0
    while proc.poll() is None:
        try:
            procs = [p] + p.children(recursive=True)
            rss = sum(x.memory_info().rss for x in procs if x.is_running())
            peak = max(peak, rss)
        except Exception:
            pass
        time.sleep(0.03)
    result["peak_rss"] = peak


def run_one(root: Path, samples: Path, count: int, workers: int) -> dict:
    out = root / f"w{workers}"
    cmd = [
        sys.executable, "-m", "scenepaste", "generate",
        "--objects", str(samples / "objects"),
        "--backgrounds", str(samples / "backgrounds"),
        "--output", str(out),
        "--count", str(count),
        "--min-objects", "1", "--max-objects", "3",
        "--class-map", "person=0,motorcycle=1,truck=2",
        "--output-format", "detect",
        "--workers", str(workers),
        "--preview-ratio", "0",
        "--run-id", f"benchmark-w{workers}",
        "--seed", "20260809",
    ]
    env = dict(os.environ)
    started = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=str(root), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    mem = {}
    th = threading.Thread(target=_monitor_rss, args=(proc, mem), daemon=True)
    th.start()
    stdout, stderr = proc.communicate()
    th.join(timeout=2)
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        raise RuntimeError(f"benchmark workers={workers} failed\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
    image_count = len(list((out / "images" / "train").glob("*")))
    return {
        "workers": workers,
        "images": image_count,
        "seconds": elapsed,
        "images_per_second": image_count / elapsed if elapsed else 0.0,
        "peak_rss_mb": (mem.get("peak_rss") / (1024 * 1024)) if mem.get("peak_rss") else None,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    from scenepaste.sample_data import bundled_samples_root
    samples = bundled_samples_root()
    with tempfile.TemporaryDirectory(prefix="scenepaste-benchmark-") as tmp:
        root = Path(tmp)
        rows = [run_one(root, samples, args.count, w) for w in args.workers]

    base = rows[0]["images_per_second"] if rows else 1.0
    print(f"Platform: {platform.platform()} | Python {platform.python_version()} | count={args.count} | 800x600 samples | Detect only | previews off")
    print("| Workers | images/s | Relative | Peak total RSS |")
    print("|---:|---:|---:|---:|")
    for r in rows:
        rss = "n/a" if r["peak_rss_mb"] is None else f"{r['peak_rss_mb']:.0f} MB"
        rel = r["images_per_second"] / base if base else 0
        print(f"| {r['workers']} | {r['images_per_second']:.2f} | {rel:.2f}x | {rss} |")
    payload = {"platform": platform.platform(), "python": platform.python_version(), "count": args.count, "rows": rows}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
