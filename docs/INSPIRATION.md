# Design references and inspiration

ScenePaste does not claim to invent Copy-Paste augmentation. Its contribution is a local-first, scene-oriented engineering workflow around controllable 2D synthetic data generation.

Several mature projects influenced the product direction:

- **Albumentations / CopyAndPaste** — annotation-aware Copy-Paste, visibility filtering, hard/gaussian blending and the separation between spatial and image-only transforms: https://albumentations.ai/explore/transform/CopyAndPaste/docs/
- **BlenderProc** — constrained object pose sampling, collision-aware placement and surface-oriented sampling: https://dlr-rm.github.io/BlenderProc/examples/advanced/object_pose_sampling/README.html
- **NVIDIA Omniverse Replicator** — domain randomization and constrained/scattered placement: https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/randomizer_details.html
- **FiftyOne Brain** — exact/near duplicate discovery, embedding-based curation and dataset QA workflows: https://docs.voxel51.com/brain/index.html

ScenePaste translates selected ideas into a lightweight 2D workflow based on real cutouts and real backgrounds rather than a 3D renderer or a training-time augmentation library.

- **Ultralytics YOLO** — prediction TXT conventions make it practical to feed model confidence/FN/FP results back into ScenePaste hard-example mining: https://docs.ultralytics.com/modes/predict/
- **WebDataset** — same-basename sample members and sequential tar-shard data access influenced ScenePaste's post-generation sharding format: https://github.com/webdataset/webdataset
