# Scene Template v2

ScenePaste templates separate a **nominal human-designed scene** from stochastic variation used during batch generation.

Old v1 templates are upgraded automatically. The editor always loads the nominal pose; parameter ranges affect generator sampling.

## Example

```json
{
  "schema": "scenepaste/scene-template",
  "version": 2,
  "canvas": {"width": 800, "height": 600},
  "parameters": {
    "position_jitter_x": 0.04,
    "position_jitter_y": 0.03,
    "scale_jitter": 0.12,
    "angle_jitter": 8.0,
    "instance_probability": 0.95,
    "flip_probability": 0.5,
    "same_class_random": true,
    "allow_overlap": true
  },
  "instances": [
    {
      "source": "sample_person.json#0",
      "source_name": "sample_person.json#0",
      "label": "person",
      "class_id": 0,
      "cx_ratio": 0.60,
      "cy_ratio": 0.58,
      "h_ratio": 0.17,
      "flip": true,
      "angle": 0.0,
      "variation": {
        "cx_range": [0.56, 0.64],
        "cy_range": [0.55, 0.61],
        "h_range": [0.15, 0.19],
        "angle_range": [-8, 8],
        "enabled_probability": 0.95,
        "flip_probability": 0.5,
        "same_class_random": true,
        "allow_overlap": true
      }
    }
  ]
}
```

Per-instance `variation` overrides the global parameter block. This allows one object to stay fixed while another varies widely.

## GUI

Arrange the nominal scene, click **📝 存模板**, then configure:

- X/Y normalized position jitter;
- scale jitter;
- angle jitter;
- instance appearance probability;
- flip probability;
- same-class random replacement;
- whether intentional template overlap is preserved.

## Batch generation

```bash
scenepaste generate \
  --objects ./objects \
  --backgrounds ./backgrounds \
  --output ./generated \
  --class-map "person=0,truck=1" \
  --scene-template ./scene_template.json \
  --count 10000 \
  --workers 8
```

Template class/placement rules take precedence over distribution-profile placement. A profile can still be kept alongside the project for QA comparison.

Bundled examples include `samples/templates/parameterized_mixed_traffic.json`.


## Relation constraints

Template slots have stable `id` values. A top-level `constraints` list may relate slots with `left_of`, `right_of`, `above`, `below`, `min_distance` or `max_distance`. Sampling is bounded by `parameters.constraint_tries`; if no valid scene is found, that sample is rejected instead of silently violating the rule.

See [PLACEMENT_CONSTRAINTS.md](PLACEMENT_CONSTRAINTS.md) and `samples/templates/constrained_person_truck.json`.
