# Placement Zones and Scene Constraints

ScenePaste supports two complementary kinds of placement control: **background zones** and **template relations**.

## Class-specific background zones

A background can have a sibling LabelMe JSON. Generic regions still use:

```text
paste_zone
ground
road
地面
可粘贴区域
```

ScenePaste also supports class-specific regions:

```text
paste_zone:person
paste_zone:truck
zone:forklift
ground:motorcycle
```

If `paste_zone:person` exists, person feet are sampled inside that mask. Other classes use their own class-specific mask when available, otherwise the generic zone, otherwise the configured ground band.

This is useful for fixed-camera scenes where, for example, pedestrians belong on a walkway while forklifts belong in an aisle.

## Template relation constraints

Each Scene Template instance is assigned a stable slot id. Add a `constraints` array to require relations between slots:

```json
{
  "constraints": [
    {"type": "right_of", "a": "person", "b": "truck", "margin": 0.05},
    {"type": "max_distance", "a": "person", "b": "truck", "value": 0.35}
  ]
}
```

Supported relation types:

- `left_of`
- `right_of`
- `above`
- `below`
- `min_distance`
- `max_distance`

Distances and margins are normalized image coordinates. `parameters.constraint_tries` controls how many stochastic template samples are attempted before the plan is rejected rather than silently violating the declared relation.

See `samples/templates/constrained_person_truck.json` for a complete example.
