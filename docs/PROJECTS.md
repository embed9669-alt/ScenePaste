# ScenePaste Project Manifest

ScenePaste v1.0 can keep a complete data workflow in one portable JSON file: `scenepaste.project.json`.

A project records:

- object/cutout directory;
- background directory;
- generated dataset directory;
- class map;
- real/reference dataset;
- validation dataset and model prediction directory;
- active distribution profile and scene template;
- default generation options.

Paths are stored relative to the manifest whenever possible, so a project folder can be moved to another workstation without rewriting every path.

## Create a project

```bash
scenepaste project init . \
  --name factory-camera-A \
  --objects ./objects \
  --backgrounds ./backgrounds \
  --output ./generated \
  --class-map "person=0,forklift=1"
```

Inspect or validate it:

```bash
scenepaste project show ./scenepaste.project.json
scenepaste project validate ./scenepaste.project.json --generation
```

Update workflow paths:

```bash
scenepaste project set ./scenepaste.project.json \
  --real-dataset ./real_train \
  --validation-dataset ./real_val \
  --predictions ./runs/detect/predict/labels \
  --distribution-profile ./profiles/factory.json \
  --workers 8 --output-format all --preview-ratio 0.01
```

## Generate from a project

```bash
scenepaste generate --project ./scenepaste.project.json --count 100000 --workers 8
```

Explicit CLI values always override project defaults, including when an explicit value happens to equal ScenePaste's built-in default. Supported project defaults include worker count, output format, preview/profile/negative-scene ratios, augmentation recipe and blend mode. The GUI can also open/save project manifests from the main toolbar.

`project validate` checks path type as well as existence, validates contiguous class IDs, and rejects invalid generation defaults. An output directory is allowed not to exist yet, but an existing output path must be a directory.
