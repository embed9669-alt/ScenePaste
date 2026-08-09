# Data Loop Center

ScenePaste v1.0 groups post-training data work into one GUI window instead of requiring users to remember many CLI commands.

Launch it from the main editor with **🧠 数据闭环**, or directly:

```bash
scenepaste loop --project ./scenepaste.project.json
```

The center exposes six workflows:

1. **Distribution Profile** — learn a real-data geometry/class profile.
2. **Hard Mining** — mine Detect, Seg, or OBB model failures and produce a hard profile.
3. **QA / Leakage** — generate QA Dashboard and cross-split similarity checks.
4. **Real vs Synthetic** — compare class/count/geometry/appearance distributions.
5. **Diversity Curation** — export a representative labeled subset.
6. **Publishing / Sharding** — build WebDataset shards with manifests/checksums.

All long tasks are launched through the same tested ScenePaste CLI via `QProcess`, so GUI and CLI results stay consistent and the editor remains responsive.
