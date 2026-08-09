# WebDataset-compatible sharding

Large image directories become inconvenient to copy, checksum and stream. ScenePaste can package an existing generated split into deterministic tar shards.

## Create shards

```bash
scenepaste shard ./generated \
  --split train \
  --max-samples 10000 \
  -o ./shards
```

Optional uncompressed payload budget:

```bash
scenepaste shard ./generated \
  --split train \
  --max-samples 10000 \
  --max-bytes 1073741824 \
  -o ./shards
```

Outputs are numbered deterministically:

```text
shards/
├── train-000000.tar
├── train-000001.tar
├── train-shards.json
├── classes.txt
├── data.yaml
└── semantic_classes.json
```

## Sample layout

All available payloads belonging to one image share one basename inside a tar:

```text
sample_000001.jpg
sample_000001.detect.txt
sample_000001.seg.txt
sample_000001.obb.txt
sample_000001.semantic.png
sample_000001.coco.json
sample_000001.json
```

Only existing modalities are written. The per-sample JSON records the split, image name and included modalities.

## Manifest

`train-shards.json` records:

- source dataset and split;
- total sample count;
- configured sample/byte caps;
- detected primary label kind;
- each tar's filename;
- sample count;
- final tar bytes;
- SHA-256 checksum.

This makes transfer and object-storage verification easier.

## Current scope

Sharding is currently **post-generation**. Workers still write the normal ScenePaste dataset first, then `scenepaste shard` packages it. Direct worker-to-tar generation is intentionally deferred because crash-safe resume, COCO finalization, random-access visual QA and tar concurrency require a different write architecture.
