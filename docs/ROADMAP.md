# ScenePaste Roadmap

## v10.0.0 — Copy-Paste only asset studio (current)

V10 keeps the mature engineering baseline and makes **human-reviewed assets + controllable Copy-Paste** the only generation path.

Landed:

- Asset Studio for foreground / background-removal Mask review and clean-background export;
- explicit deterministic **Label-First Planner**;
- scene-region policy with LabelMe zones, class zones, forbidden zones, and ground-prior fallback;
- built-in active hard-case recipes (`small-object`, `far-occluded`, `crowded`);
- Detect/Seg/OBB/COCO/Semantic writers, resumable multiprocessing, profile/template;
- hard-mining, QA/leakage, curation and sharding workflows;
- `scenepaste factory plan` for dry-run placement plans.

Removed from the product surface (not planned as production features):

- generative backends (Diffusers / light-inpaint / Qwen / HTTP generative bridge / plugin generative path);
- `scenepaste model` / `scenepaste qwen` and the `[generative]` optional dependency.

## Next research layers (optional, not required for Copy-Paste)

- better semantic/depth/ground-plane scene-region analyzers;
- SAM2/refiner plugins for **cutout** quality (not generative paste);
- adapters that turn training/evaluation metrics into hard-case recipes;
- richer Asset Studio tools for mask QA and batch review.
