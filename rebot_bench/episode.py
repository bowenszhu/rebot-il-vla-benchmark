"""Record one teleoperated episode: mark it for the sensor recorder, run the
direct profile-07 mirror (same supervised flags as the 2026-10-15 run03),
then ask the operator to keep or reject it.

Running this ENABLES AND MOVES THE ARM through the existing mirror; it adds no
new motion path, limits or gates. Use only when the operator is at the arm.

    python3 -B rebot_bench/episode.py \
        --session sessions/t1_20261015 --episode 0 \
        --tactile-status <W2L status.json>

T2 (language-conditioned colour squares): add --protocol protocols/t2_protocol.json;
each schedule row then carries its own instruction, which goes into meta.json as the task.
"""
import argparse
import hashlib
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROTOCOL = ROOT / 'protocols/t1_protocol.json'
# Profile 08 = 07 with the J7 opening widened to 5.92 for the 6 cm cube. Controller paths: rig.py.
from rebot_bench.rig import (LEADER_PORT, MIRROR, MOTION_HOLD, OBS_STREAMS_CONFIG, PROFILE,  # noqa: E402
                             REBOT_PYTHON, RIG_ROOT, SIDE_CAMERA)
# Normal ends: duration reached, or operator Ctrl+C after which the safe return completed.
GOOD_OUTCOMES = ('completed', 'operator_stop_returned')
# DJI Action 4 in UVC webcam mode (observation/website video only, never training input): SIDE_CAMERA.
# Optional DroidCam phones/tablets: OBS_STREAMS_CONFIG {"name": "http://IP:4747/video"}; local only, device IPs stay out of Git.


def _start_ffmpeg(episode_dir, name, input_args):
    """Stream-copy one source to <name>.mkv; own session so Ctrl+C reaches only the mirror."""
    out = episode_dir / f'{name}.mkv'
    log = (episode_dir / f'{name}_ffmpeg.log').open('w')
    try:
        proc = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'warning', *input_args,
                                 '-c', 'copy', str(out)],
                                stdin=subprocess.PIPE, stdout=log, stderr=log, start_new_session=True)
    except OSError as exc:
        print(f'WARNING: {name} recording not started ({exc}); episode continues without it')
        return None
    return {'proc': proc, 'file': out.name, 'started_monotonic': time.monotonic(), 'started_wall': time.time()}


def start_obs_video(episode_dir):
    """DJI Action 4 side view (1080p30 MJPEG over UVC)."""
    return _start_ffmpeg(episode_dir, 'action4', ['-f', 'v4l2', '-input_format', 'mjpeg',
                                                  '-video_size', '1920x1080', '-framerate', '30',
                                                  '-i', SIDE_CAMERA])


def start_stream_videos(episode_dir, config=OBS_STREAMS_CONFIG):
    """DroidCam MJPEG-over-HTTP streams listed in the local config (one client per device)."""
    try:
        sources = json.loads(Path(config).read_text())
    except FileNotFoundError:
        return []
    videos = []
    for name, url in sources.items():
        if name.startswith('_'):
            continue
        video = _start_ffmpeg(episode_dir, name, ['-use_wallclock_as_timestamps', '1', '-reconnect', '1',
                                                  '-reconnect_streamed', '1', '-reconnect_delay_max', '2',
                                                  '-f', 'mjpeg', '-i', url])
        if video is not None:
            videos.append({'name': name, **video})
    return videos


def stop_obs_video(video):
    if video is None:
        return None
    proc = video.pop('proc')
    if proc.poll() is None:
        try:
            proc.communicate(b'q', timeout=10)  # ffmpeg finalizes the file on 'q'
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(5)
    video.update(stopped_monotonic=time.monotonic(), ffmpeg_exit=proc.returncode)
    if proc.returncode not in (0, 255):
        name = video['file'].rsplit('.', 1)[0]
        print(f"WARNING: {name} recording exit {proc.returncode}; see {name}_ffmpeg.log")
    return video


def start_home_arg(protocol):
    """Protocol 'start_home_q' {joint: q} -> mirror --start-home value ('' = start from the rest pose)."""
    return ','.join(f'{int(j)}={float(q):g}' for j, q in sorted(protocol.get('start_home_q', {}).items(), key=lambda kv: int(kv[0])))


def mirror_command(episode_dir, camera_status, tactile_status, duration, start_home=''):
    """Run03 flags with profile 08; J4 rate 0.25 -> 0.75 rad/s (mirror default, user request 2026-10-16)."""
    return [str(REBOT_PYTHON), '-u', '-B', str(MIRROR),
            '--mirror', '--monitored-contact', '--wrist-observe-only', '--align-arm-start', '--align-gripper-start',
            '--camera-status', str(camera_status), '--tactile-status', str(tactile_status),
            '--control-profile', str(PROFILE), '--leader-port', LEADER_PORT,
            '--joints', '1,2,3,4,5,6,7', '--speed-scale', '1.25', '--joint-velocities', '1=0.9,2=1.0,4=0.75,7=2.0',
            '--command-period', '.02', '--return-velocity', '.15', '--duration', f'{duration:g}',
            '--armed', '--supervised-test', '--confirm', 'SUPERVISED_PX6D_W2L_MIRROR',
            '--operator-status', str(episode_dir / 'operator_status.json'),
            '--trace', str(episode_dir / 'mirror_trace.json')] + (['--start-home', start_home] if start_home else [])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fresh_status(path, max_age_s=1.0):
    data = json.loads(Path(path).read_text())
    if data.get('state') != 'running' or not 0 <= time.time() - data.get('updated_at', 0) <= max_age_s:
        raise SystemExit(f'sensor recorder status {path} is not running/fresh; start sensor_recorder.py first')
    if data.get('active_episode'):
        raise SystemExit(f"recorder still marks {data['active_episode']} active; remove the stale ACTIVE_EPISODE")
    for name in ('d405', 'd435i'):
        if data.get('cameras', {}).get(name, {}).get('state') != 'live':
            raise SystemExit(f'{name} not live in {path}')
    if data.get('px6d', {}).get('state') != 'live':
        raise SystemExit(f'PX6D not live in {path}')


def update_meta(path, **fields):
    meta = json.loads(path.read_text())
    meta.update(fields)
    path.write_text(json.dumps(meta, indent=1, ensure_ascii=False))


def run_with_child_ctrl_c(command, log_path=None, **kwargs):
    """Run the mirror so Ctrl+C reaches IT (graceful stop + height-checked return).

    The wrapper must survive Ctrl+C but must not use SIG_IGN: an ignored SIGINT is
    inherited across exec and Python keeps it ignored, so the mirror would never see
    Ctrl+C (2026-10-16 bug). A Python-level no-op handler is reset to default in the child.
    With ``log_path`` the mirror's output is echoed to the terminal and saved there.
    """
    previous = signal.signal(signal.SIGINT, lambda *_: None)
    try:
        if log_path is None:
            return subprocess.run(command, **kwargs).returncode
        with open(log_path, 'w') as log, subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                         text=True, bufsize=1, **kwargs) as proc:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
            return proc.wait()
    finally:
        signal.signal(signal.SIGINT, previous)


def next_episode(session, count):
    """First schedule index with no accepted episode; a rejected attempt is retried."""
    accepted = set()
    for meta_path in (session / 'episodes').glob('ep*_*/meta.json'):
        try:
            if json.loads(meta_path.read_text()).get('accepted'):
                accepted.add(int(meta_path.parent.name[2:5]))
        except (ValueError, OSError):
            continue
    for index in range(count):
        if index not in accepted:
            return index
    raise SystemExit(f'all {count} scheduled episodes already accepted in {session}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL,
                        help='task protocol JSON (default: t1_protocol.json; T2: t2_protocol.json)')
    parser.add_argument('--episode', type=int,
                        help='index in the protocol schedule; default: first index without an accepted episode in this session')
    parser.add_argument('--tactile-status', type=Path, required=True, help='W2L dual preview status.json')
    parser.add_argument('--duration', type=float, help="mirror duration, (0, 120] s (default: the protocol's episode_duration_s, else 55); still tails are trimmed offline")
    parser.add_argument('--cell', help='override schedule cell (pilot / evaluation)')
    parser.add_argument('--yaw', type=int, help='override schedule yaw in degrees')
    parser.add_argument('--mat-offset-cm', type=float,
                        help='signed lateral mat offset, + = mat to the arm\'s left (cm); record-only, may be filled after the pilot')
    parser.add_argument('--obs-video', action='store_true',
                        help='also record the Action 4 side view (1080p MJPEG copy) and any DroidCam streams listed in '
                             'REBOT_OBS_STREAMS into the episode, for website video')
    parser.add_argument('--print-command', action='store_true', help='print the mirror command and exit')
    args = parser.parse_args()

    session = args.session.resolve()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    if args.episode is None:
        args.episode = next_episode(session, len(protocol['schedule']))
    plan = dict(protocol['schedule'][args.episode])
    if args.duration is None:
        args.duration = float(protocol.get('episode_duration_s', 55.0))
    if args.cell:
        plan.update(cell=args.cell, override=True)
    if args.yaw is not None:
        plan.update(yaw_deg=args.yaw, override=True)
    task = plan.get('task', protocol['task'])
    camera_status = session / 'sensor_status.json'
    stamp = time.strftime('%Y%m%d_%H%M%S')
    episode_dir = session / 'episodes' / f"ep{args.episode:03d}_{stamp}"
    command = mirror_command(episode_dir, camera_status, args.tactile_status.resolve(), args.duration,
                             start_home_arg(protocol))
    if args.print_command:
        print(' '.join(command))
        return 0
    if not 0 < args.duration <= 120:
        raise SystemExit('duration must be in (0, 120] s')
    if MOTION_HOLD.exists():
        raise SystemExit(f'{MOTION_HOLD} exists: previous run faulted; handle it on site before recording')
    fresh_status(camera_status)
    running = subprocess.run(['pgrep', '-af', 'stararm102_b601_joint_mirror'], capture_output=True, text=True).stdout
    if running.strip():
        raise SystemExit(f'another mirror process is running:\n{running}')

    print(f"\nEpisode {args.episode} (batch {plan.get('batch')}): cube at {plan['cell']}, yaw {plan['yaw_deg']} deg, "
          f"mat offset {args.mat_offset_cm} cm" + (f", layer {plan['layer']}" if 'layer' in plan else '')
          + (' - REBUILD THE TOWER FIRST' if plan.get('rebuild_before') else ''))
    print(f'Task: {task}')
    print(protocol.get('setup_prompt', 'Arm at supported zero pose, gripper near-closed, leader neutral, box in place.'))
    print(protocol.get('demo_prompt', 'Demo: first lift straight up/back to the fixed look pose (~1 s), '
                                      'then reach, grasp, place, return leader to neutral.'))
    input('Press Enter to start the mirror (Ctrl+C to abort)... ')

    episode_dir.mkdir(parents=True)
    meta_path = episode_dir / 'meta.json'
    meta_path.write_text(json.dumps({
        'protocol_id': protocol['protocol_id'], 'task': task, 'plan': plan,
        'duration_s': args.duration, 'mat_offset_cm': args.mat_offset_cm, 'profile': str(PROFILE.relative_to(RIG_ROOT)), 'profile_sha256': sha256(PROFILE),
        'protocol': str(protocol_path.relative_to(ROOT)), 'protocol_sha256': sha256(protocol_path), 'started_wall': time.time(), 'command': command,
    }, indent=1, ensure_ascii=False))
    marker = session / 'ACTIVE_EPISODE'
    tmp = marker.with_suffix('.tmp')
    tmp.write_text(str(episode_dir))
    tmp.replace(marker)
    video = start_obs_video(episode_dir) if args.obs_video else None
    streams = start_stream_videos(episode_dir) if args.obs_video else []
    try:
        code = run_with_child_ctrl_c(command, log_path=episode_dir / 'mirror_console.log', cwd=RIG_ROOT)
    finally:
        marker.unlink(missing_ok=True)
        video = stop_obs_video(video)
        streams = [stop_obs_video(stream) for stream in streams]
    if video is not None:
        update_meta(meta_path, obs_video=video)
    if streams:
        update_meta(meta_path, obs_streams=streams)
    time.sleep(0.5)  # let queued frames reach disk
    outcome = None
    trace = episode_dir / 'mirror_trace.json'
    if trace.exists():
        outcome = json.loads(trace.read_text()).get('outcome')
    print(f'\nmirror exit={code} outcome={outcome}')
    if code != 0 or outcome not in GOOD_OUTCOMES:
        update_meta(meta_path, accepted=False, reason=f'mirror exit={code} outcome={outcome}', ended_wall=time.time())
        print('Episode REJECTED automatically. Check the arm state on site before the next episode.')
        return 1
    answer = input('Keep this episode? [y = success / anything else = reject]: ').strip().lower()
    reason = '' if answer == 'y' else input('Reject reason (short): ').strip()
    update_meta(meta_path, accepted=answer == 'y', reason=reason, ended_wall=time.time())
    print('ACCEPTED' if answer == 'y' else 'REJECTED', episode_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
