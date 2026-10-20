---
license: cc-by-4.0
task_categories:
- robotics
tags:
- LeRobot
- imitation-learning
- vision-language-action
- real-robot
- reBot-B601-RS
configs:
- config_name: default
  data_files: data/*/*.parquet
---

# reBot B601-RS, Task 1: pick and place (`rebot_t1_v1`)

50 teleoperated demonstrations, 43,607 frames at 30 Hz, LeRobot v3.0 format.
Instruction (one task string): *"Pick up the blue cube and place it in the white box."*

A 6 cm blue foam cube starts on one of the nine points printed in the black frame of a white A3 mat
(rows A–C near to far, columns 1–3 left to right) at a yaw of 0° or 45°, and is placed in a white lunch
box on the table to the right of the mat. The schedule covers each point five times plus five extra
starts, alternating the yaw per point (`protocols/t1_protocol.json` in the repository).

Used to train ACT, Diffusion Policy and SmolVLA, evaluated closed-loop on the arm: in-distribution
SmolVLA 13/15, Diffusion Policy 12/15 (separate day, trained on a copy with the idle start trimmed),
ACT 9/15; see the repository for the protocol, every trial and the caveats.

## Recording

- **Robot**: Seeed Studio reBot Arm B601-RS (6 arm joints + gripper, RobStride motors), teleoperated
  1:1 in joint space from a Fashion Star Star Arm 102-LD leader arm. The end effector carries a PaXini
  PX6D-75 force/torque sensor and two Daimon DM-Tac W2L fingertips; their data are **not** in this dataset.
- **Cameras**: `observation.images.wrist` = Intel RealSense D405 on a 30° wrist mount,
  `observation.images.scene` = RealSense D435i looking down at the table; RGB 640×480, 30 fps,
  stored as H.264 video.
- **State and action** (7 × float32, rad): `j1`…`j6`, `gripper` (J7). `observation.state` is the
  measured joint position, `action` the target the controller actually sent (after its speed and joint
  limits), both as `raw − reference` of the rig's joint-reference profile 08. J7 grows as the
  gripper opens.
- **Alignment**: the controller logs at 50 Hz; frames are resampled to a 30 Hz grid over the follow
  phase, still segments at both ends trimmed (0.02 rad threshold, 0.5 s pad). Each tick uses the newest
  camera frame that arrived **before** it (causal). An episode is rejected if more than 2% of ticks see
  a frame older than 0.1 s or any tick sees one older than 0.25 s; every kept episode passed.
- **Timestamps** are host arrival times on one monotonic clock, not exposure times; there is no
  hardware synchronisation, and RealSense frames arrive some tens of ms after exposure.
- Episodes were kept only when the operator accepted them on the spot (task done, at most one regrasp,
  no collision, controller ended normally); rejected attempts are not included.

## Use

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # lerobot 0.4.x, codebase v3.0
ds = LeRobotDataset("bowenszhu/rebot-b601-t1-pick-place")
```

Training presets, protocols, evaluation code and the logged closed-loop results are in
<https://github.com/bowenszhu/rebot-il-vla-benchmark>.

## Limitations

One operator, one scene, each task collected in a single sitting of about two hours; small (50 and
60 episodes). Joint-space teleoperation, so tool motions are not straight in Cartesian space.
Lighting was not controlled. No depth, touch or force streams.

## Licence

CC BY 4.0. Cite the repository above.
