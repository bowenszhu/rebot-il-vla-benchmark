"""One closed-loop policy trial on the real arm (control process, rebot env).

The policy runs in policy_server.py (separate process, GPU env). This process
connects to it, then calls the direct mirror's run_mirror() with the policy as
the target source instead of the leader: the same profile-08 limits and J7
start rule, CSPTargetLimiter envelope and 2 s no-progress stop,
check_mirror_feedback_limits, require_sensors freshness, height-checked and
staged return, fault latch and mirror_motion_hold.json. Speeds are the
collection flags (episode.py). A policy timeout holds the last command and a
lost/crashed policy ends follow through the normal return, and so does the
policy bringing J1-J6 back to the start (outcome policy_rest_returned); Ctrl+C
is the mirror's usual operator stop.

Running this ENABLES AND MOVES THE ARM. Operator at the arm, hand near Ctrl+C.

    "$REBOT_PYTHON" -u -B rebot_bench/evaluate.py \\
        --session sessions/t1_eval_20261017 --policy-tag act --trial 0 \\
        --tactile-status <W2L status.json>

T2: add --protocol protocols/t2_protocol.json. The trial's
instruction is sent to the policy server for this trial only, and the result
categories come from the protocol (wrong_square etc.).
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rebot_bench.episode import (MOTION_HOLD, PROFILE, PROTOCOL, RIG_ROOT, fresh_status,  # noqa: E402
                                mirror_command, sha256, start_home_arg, start_obs_video, start_stream_videos,
                                stop_obs_video, update_meta)
from rebot_bench.policy_link import (DEFAULT_SOCKET, POLICY_LOST_S, POLICY_TIMEOUT_S,  # noqa: E402
                                    PolicyClient)
from rebot_bench.rig import import_rig  # noqa: E402

# Operator-entered result categories (one per trial); a protocol may define its own.
OUTCOMES = {
    '1': ('success', 'cube released inside the box'),
    '2': ('grasp_miss', 'never held the cube (missed / pushed it)'),
    '3': ('drop', 'grasped, dropped before the box'),
    '4': ('place_miss', 'released, but not inside the box'),
    '5': ('collision', 'hit box / table / fixture (operator stop)'),
    '6': ('stall', 'no task progress before the end (hesitation / freeze)'),
    '7': ('safety_stop', 'controller fault, hold latch or policy lost'),
    '8': ('other', 'anything else; describe in the note'),
}
# J3 never starts a policy trial lower than the demonstrations did: rebot_t1_v1
# frame-0 observation.state j3 median is +0.0139 rad (range -0.011..+0.026).
# Resting lower on the zero support (it varies after power cycles), ACT
# stalled at zero (10-16: J3 q <= 0 in 7 of 8 slow or no starts).
POLICY_MIN_Q = {3: 0.0139}
# The leader-only start options; everything else equals the collection command.
LEADER_ONLY = ('--align-arm-start', '--align-gripper-start')


def outcome_table(protocol):
    """{'1': (name, text), ...} from protocol['outcome_categories'], else the T1 table."""
    categories = protocol.get('outcome_categories')
    if not categories:
        return OUTCOMES
    return {str(k): (name, text) for k, (name, text) in enumerate(categories, 1)}


def policy_mirror_argv(trial_dir, camera_status, tactile_status, duration, start_home=''):
    """episode.py's run03 mirror flags minus the leader port and leader alignment."""
    argv = mirror_command(trial_dir, camera_status, tactile_status, duration, start_home)[4:]
    port = argv.index('--leader-port')
    del argv[port:port+2]
    return [a for a in argv if a not in LEADER_ONLY]


def eval_trials(protocol):
    """Trial rows in index order: evaluation['trials'] (each row names its own split) when present, else ID then OOD."""
    evaluation = protocol['evaluation']
    if 'trials' in evaluation:
        return [dict(row, split=row.get('split', 'in_distribution')) for row in evaluation['trials']]
    return ([dict(row, split='in_distribution') for row in evaluation['in_distribution']]
            + [dict(row, split='out_of_distribution') for row in evaluation['out_of_distribution']])


def trace_summary(trace):
    """Latency / hold statistics from the mirror trace's per-command policy rows."""
    rows = [r['policy'] for r in trace.get('commands', []) if r.get('phase') == 'follow' and r.get('policy')]
    def stats(values):
        values = sorted(v for v in values if v is not None)
        if not values:
            return None
        return {'n': len(values), 'median': round(statistics.median(values), 2),
                'p95': round(values[min(len(values)-1, int(.95*len(values)))], 2), 'max': round(values[-1], 2)}
    states = {}
    for row in rows:
        states[row['state']] = states.get(row['state'], 0) + 1
    return {'commands': len(rows), 'states': states,
            'infer_ms': stats(r.get('infer_ms') for r in rows),
            'obs_to_action_ms': stats(r.get('obs_to_action_ms') for r in rows),
            'action_age_ms': stats(1000*r['age_s'] for r in rows if r.get('age_s') is not None),
            'unique_actions': len({r.get('action_seq') for r in rows if r.get('action_seq') is not None})}


def ask_outcome(suggest=None, outcomes=OUTCOMES):
    print('\nTrial result:')
    for key, (name, text) in outcomes.items():
        print(f'  {key} {name:12} {text}')
    while True:
        answer = input(f"Category number{f' [Enter = {suggest}]' if suggest else ''}: ").strip()
        if not answer and suggest:
            return suggest
        if answer in outcomes:
            return outcomes[answer][0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--session', type=Path, required=True, help='recorder session (sensor_status.json lives here)')
    parser.add_argument('--policy-tag', required=True, help='short name, e.g. act / dp / smolvla; groups trials')
    parser.add_argument('--protocol', type=Path, default=PROTOCOL,
                        help='task protocol JSON (default: t1_protocol.json; T2: t2_protocol.json)')
    parser.add_argument('--trial', type=int, required=True,
                        help='index into the protocol evaluation (T1: 0-14 ID, 15-19 OOD; T2: 0-19)')
    parser.add_argument('--tactile-status', type=Path, required=True, help='W2L dual preview status.json')
    parser.add_argument('--duration', type=float, help='follow duration, (0, 120] s (default: protocol episode_duration_s)')
    parser.add_argument('--pos', help='override the trial position label')
    parser.add_argument('--yaw', type=int, help='override the trial yaw in degrees')
    parser.add_argument('--socket', default=DEFAULT_SOCKET)
    parser.add_argument('--policy-timeout', type=float, default=POLICY_TIMEOUT_S, help='hold after this action age, s')
    parser.add_argument('--policy-lost', type=float, default=POLICY_LOST_S, help='normal return after this action age, s')
    parser.add_argument('--no-frames', action='store_true', help='do not mark the trial for the recorder (no saved frames)')
    parser.add_argument('--obs-video', action='store_true',
                        help='also record the Action 4 side view and the DroidCam streams (website video, as episode.py)')
    parser.add_argument('--print-command', action='store_true', help='print the mirror arguments and exit')
    args = parser.parse_args()

    session = args.session.resolve()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    outcomes = outcome_table(protocol)
    if args.duration is None:
        args.duration = float(protocol['episode_duration_s'])
    plan = dict(eval_trials(protocol)[args.trial], trial=args.trial)
    if args.pos:
        plan.update(pos=args.pos, override=True)
    if args.yaw is not None:
        plan.update(yaw_deg=args.yaw, override=True)
    task = plan.get('task', protocol['task'])
    camera_status = session / 'sensor_status.json'
    stamp = time.strftime('%Y%m%d_%H%M%S')
    trial_dir = session / 'evals' / args.policy_tag / f'trial{args.trial:02d}_{stamp}'
    mirror_argv = policy_mirror_argv(trial_dir, camera_status, args.tactile_status.resolve(), args.duration,
                                     start_home_arg(protocol))
    if args.print_command:
        print(' '.join(mirror_argv))
        return 0
    if not 0 < args.duration <= 120:
        raise SystemExit('duration must be in (0, 120] s')
    if MOTION_HOLD.exists():
        raise SystemExit(f'{MOTION_HOLD} exists: previous run faulted; handle it on site before evaluating')
    fresh_status(camera_status)
    others = [line for line in subprocess.run(
        ['pgrep', '-af', 'stararm102_b601_joint_mirror|il_vla_benchmark/(episode|evaluate).py'],
        capture_output=True, text=True).stdout.splitlines() if not line.startswith(f'{os.getpid()} ')]
    if others:
        raise SystemExit('another mirror/episode/evaluate process is running:\n' + '\n'.join(others))

    import_rig()
    import stararm102_b601_joint_mirror as mirror
    mirror_args = mirror.parse_args(mirror_argv)
    mirror_args.policy_min_q = POLICY_MIN_Q
    client = PolicyClient(args.socket, args.policy_timeout, args.policy_lost, log_path=None, task=task)
    hello = client.connect()
    print(f"\nPolicy: {hello['policy'].get('type')} {hello['policy'].get('path')}")
    print(f"Trial {args.trial} ({plan['split']}): {plan['pos']} yaw {plan['yaw_deg']} deg"
          + (f" xy_cm={plan['xy_cm']}" if plan.get('xy_cm') else '') + (f" - {plan['note']}" if plan.get('note') else ''))
    print(f'Task sent to the policy: {task}')
    print('Arm at supported zero pose, gripper near-closed, cube placed, box in place, leader untouched.')
    print('Ctrl+C at any time = normal operator stop and return (also when the task is done).')
    try:
        input('Press Enter to start (Ctrl+C to abort)... ')
    except KeyboardInterrupt:
        client.close()
        print('\naborted; nothing enabled')
        return 1

    trial_dir.mkdir(parents=True)
    client.log_path = str(trial_dir / 'policy_server_log.jsonl')
    meta_path = trial_dir / 'meta.json'
    meta_path.write_text(json.dumps({
        'kind': 'policy_eval', 'protocol_id': protocol['protocol_id'], 'task': task, 'plan': plan,
        'policy_tag': args.policy_tag, 'policy': hello.get('policy'), 'duration_s': args.duration,
        'profile': str(PROFILE.relative_to(RIG_ROOT)), 'profile_sha256': sha256(PROFILE),
        'protocol': str(protocol_path.relative_to(ROOT)), 'protocol_sha256': sha256(protocol_path), 'mirror_argv': mirror_argv,
        'policy_timeout_s': args.policy_timeout, 'policy_lost_s': args.policy_lost, 'started_wall': time.time(),
        'policy_min_q': POLICY_MIN_Q,
    }, indent=1, ensure_ascii=False))
    marker = session / 'ACTIVE_EPISODE'
    if not args.no_frames:
        tmp = marker.with_suffix('.tmp')
        tmp.write_text(str(trial_dir))
        tmp.replace(marker)
    video = start_obs_video(trial_dir) if args.obs_video else None
    streams = start_stream_videos(trial_dir) if args.obs_video else []
    code, error = None, None
    try:
        code = mirror.run_mirror(mirror_args, policy=client)
    except BaseException as exc:  # trace and hold latch are already written by run_mirror
        error = repr(exc)
        print(f'\nCONTROL EXIT WITH ERROR: {error}')
    finally:
        if not args.no_frames:
            marker.unlink(missing_ok=True)
        client.close()
        video = stop_obs_video(video)
        streams = [stop_obs_video(stream) for stream in streams]
    if video is not None:
        update_meta(meta_path, obs_video=video)
    if streams:
        update_meta(meta_path, obs_streams=streams)
    time.sleep(.5)  # let queued frames reach disk
    trace_path = trial_dir / 'mirror_trace.json'
    trace = json.loads(trace_path.read_text()) if trace_path.exists() else {}
    outcome = trace.get('outcome')
    summary = trace_summary(trace)
    print(f'\ncontrol exit={code} outcome={outcome} error={error}')
    print(f"policy steps: {summary['states']}; infer_ms {summary['infer_ms']}; obs->action ms {summary['obs_to_action_ms']}")
    if MOTION_HOLD.exists():
        print(f'FAULT LATCH {MOTION_HOLD} present: arm may be energized; handle on site before the next trial.')
    normal = outcome in ('completed', 'operator_stop_returned', 'policy_rest_returned')
    category = ask_outcome(None if normal else 'safety_stop', outcomes)
    note = input('Note (optional): ').strip()
    result = {'category': category, 'success': category == 'success', 'note': note,
              'control_exit': code, 'control_error': error, 'mirror_outcome': outcome,
              'policy_lost': trace.get('policy_lost'), 'latency': summary, 'ended_wall': time.time()}
    update_meta(meta_path, result=result)
    with (session / 'evals' / 'results.jsonl').open('a') as handle:
        handle.write(json.dumps({'trial_dir': str(trial_dir.relative_to(session)), 'policy_tag': args.policy_tag,
                                 'trial': args.trial, 'split': plan['split'], 'pos': plan['pos'],
                                 'yaw_deg': plan['yaw_deg'], 'protocol_id': protocol['protocol_id'], 'task': task,
                                 **{k: plan[k] for k in ('colour', 'dimension', 'layer') if k in plan}, **result}, ensure_ascii=False) + '\n')
    print(f'{category.upper()} -> {trial_dir}')
    return 0 if error is None else 1


if __name__ == '__main__':
    sys.exit(main())
