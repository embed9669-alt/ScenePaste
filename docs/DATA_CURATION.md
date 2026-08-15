# Dataset curation and leakage checks

ScenePaste includes a lightweight local-first data-curation layer inspired by common similarity/uniqueness workflows while avoiding a mandatory foundation-model download.

## Cross-split leakage

```bash
scenepaste curate leakage ./dataset
```

The report scans `train`, `val`, and `test` and separates three signals:

1. **exact** — identical SHA-1 content across splits;
2. **pHash near duplicate** — visually near-identical images under a bounded 64-bit perceptual hash distance;
3. **cv-lite embedding similarity** — very high cosine similarity using ScenePaste's built-in descriptor.

This is intentionally stricter than source-aware synthetic splitting: it can also catch accidental copied files, transcoded duplicates, or highly similar frames from video.

Useful controls:

```bash
scenepaste curate leakage ./dataset \
  --phash-threshold 6 \
  --embedding-threshold 0.995 \
  --embedding-limit 1000
```

## Lightweight diversity

```bash
scenepaste curate diversity ./dataset --limit 5000
```

`cv-lite-v1` uses local Lab color statistics, low-frequency DCT and gradient-orientation features. It is deterministic, offline, small and fast enough for curation. It is **not** a semantic CLIP/DINO-style embedding.

Select a diverse representative subset:

```bash
scenepaste curate diversity ./dataset \
  --select 1000 \
  --export-dataset ./diverse_1k
```

Selection uses greedy farthest-point sampling in the normalized descriptor space. When `--export-dataset` is used, ScenePaste copies matching Detect/Seg/OBB/Semantic labels and filters COCO annotations when available.

## Real vs synthetic comparison

```bash
scenepaste compare ./real ./synthetic -o ./comparison
```

Outputs:

```text
real_vs_synthetic.json
real_vs_synthetic.html
```

The comparison includes:

- class-frequency total variation;
- objects-per-image distribution distance;
- per-class center-X / center-Y / bottom-Y distributions;
- width / height / area / aspect-ratio distributions;
- cv-lite domain-centroid cosine similarity;
- synthetic-to-real nearest-neighbor similarity;
- real and synthetic uniqueness summaries.

The report is a diagnosis tool, not a proof that synthetic data is realistic. Always retain an independent real validation/test set.
