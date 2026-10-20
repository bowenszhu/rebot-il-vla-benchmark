"""Where the rig-specific pieces live. They are not part of this repository.

Collection (episode.py) and evaluation (evaluate.py, eval_next.py) drive the arm through the
direct leader-follower controller of the rig (`stararm102_b601_joint_mirror.py` with
`run_mirror(args, policy=None)`), its joint-reference profile and its fault latch. That
controller, the motor driver and the safety parameters stay in the private rig repository;
README.md "Controller interface" lists what these scripts expect from it. Point the variables
below at a checkout of it to run on hardware. The offline parts (convert_episodes.py alignment,
policy_server.py, policy_link.py, frame_shm.py, analysis/) do not need any of this.

  REBOT_RIG_ROOT       controller checkout (default: current directory)
  REBOT_PYTHON         interpreter of the controller environment (default: this one)
  REBOT_LEADER_PORT    leader-arm serial port
  REBOT_SIDE_CAMERA    V4L2 device of the side-view camera (observation video only)
  REBOT_OBS_STREAMS    JSON {"name": "http://<device>:4747/video"} of extra observation streams
"""
import os
import sys
from pathlib import Path

RIG_ROOT = Path(os.environ.get('REBOT_RIG_ROOT', '.')).resolve()
REBOT_PYTHON = Path(os.environ.get('REBOT_PYTHON', sys.executable))
MIRROR = RIG_ROOT / 'stararm102_b601_joint_mirror.py'
# Profile 08: joint reference (q = raw - reference_raw_rad), limits and mapping used for all T1/T2 data.
PROFILE = RIG_ROOT / 'hardware/end_effectors/control_profiles/b601_px6d_w2l_wide_open_20261016_08.json'
MOTION_HOLD = RIG_ROOT / 'datasets/commissioning/mirror_motion_hold.json'
LEADER_PORT = os.environ.get('REBOT_LEADER_PORT', '/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0')
SIDE_CAMERA = os.environ.get('REBOT_SIDE_CAMERA', '/dev/video0')
OBS_STREAMS_CONFIG = Path(os.environ.get('REBOT_OBS_STREAMS', RIG_ROOT / 'datasets/il_vla_benchmark/obs_cameras.json'))


def import_rig():
    """Make the controller checkout importable (stararm102_b601_joint_mirror, hardware.*)."""
    if str(RIG_ROOT) not in sys.path:
        sys.path.insert(0, str(RIG_ROOT))
