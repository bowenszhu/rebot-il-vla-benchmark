"""Convert accepted mirror episodes to a LeRobot v3 dataset (offline, no hardware).

Inputs per episode (written by episode.py + sensor_recorder.py):
  meta.json, mirror_trace.json (50 Hz follow commands, CLOCK_MONOTONIC),
  d405_frames.jsonl / d435i_frames.jsonl + frames/<cam>/<arrival_ns>.jpg

Output frames on a 30 Hz grid over the follow phase (leading/trailing still
segments trimmed):
  observation.state  = measured joint position, q = raw - profile reference (rad)
  action             = target actually sent by the mirror, same coordinates
  observation.images.wrist / .scene = latest D405 / D435i frame that arrived
                       at or before the tick (causal; stale ticks reject the episode)

    conda run -n lerobot-blackwell python -B rebot_bench/convert_episodes.py \
        --session sessions/t1_20261015 --out data/rebot_t1_v1 --check-only
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rebot_bench.rig import RIG_ROOT  # noqa: E402

JOINTS = ('1', '2', '3', '4', '5', '6', '7')
NAMES = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'gripper']
CAMERAS = {'d405': 'observation.images.wrist', 'd435i': 'observation.images.scene'}


def load_follow(trace):
    """Follow-phase rows -> (t seconds, actual raw Nx7, sent raw Nx7)."""
    rows = [row for row in trace['commands'] if row.get('phase') == 'follow'
            and all(j in row.get('actual_raw_rad', {}) and j in row.get('sent_raw_rad', {}) for j in JOINTS)]
    if len(rows) < 2:
        raise ValueError('trace has fewer than two complete follow commands')
    t = np.array([row['monotonic'] for row in rows], dtype=float)
    if np.any(np.diff(t) <= 0):
        raise ValueError('follow timestamps are not strictly increasing')
    actual = np.array([[row['actual_raw_rad'][j] for j in JOINTS] for row in rows], dtype=float)
    sent = np.array([[row['sent_raw_rad'][j] for j in JOINTS] for row in rows], dtype=float)
    return t, actual, sent


def trim_window(t, actual, threshold=0.02, pad_s=0.5):
    """Drop still time before the first and after the last motion > threshold rad."""
    moved_from_start = np.max(np.abs(actual - actual[0]), axis=1) > threshold
    moved_from_end = np.max(np.abs(actual - actual[-1]), axis=1) > threshold
    if not moved_from_start.any():
        raise ValueError('arm never moved more than threshold')
    first = t[np.argmax(moved_from_start)]
    last = t[len(t) - 1 - np.argmax(moved_from_end[::-1])]
    return max(t[0], first - pad_s), min(t[-1], last + pad_s)


def load_frames(episode_dir, camera):
    path = episode_dir / f'{camera}_frames.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda row: row['arrival_ns'])
    return np.array([row['arrival_ns'] / 1e9 for row in rows], dtype=float), [row['file'] for row in rows]


def build_timeline(t, actual, sent, reference, frames, fps, window, max_age_s):
    """Resample to fps ticks; returns state/action arrays, per-camera frame indices and age stats."""
    start, end = window
    ticks = np.arange(start, end + 1e-9, 1.0 / fps)
    state = np.stack([np.interp(ticks, t, actual[:, j]) for j in range(7)], axis=1) - reference
    action = np.stack([np.interp(ticks, t, sent[:, j]) for j in range(7)], axis=1) - reference
    indices, stats = {}, {}
    for camera, (arrival, _files) in frames.items():
        idx = np.searchsorted(arrival, ticks, side='right') - 1
        valid = idx >= 0
        age = np.where(valid, ticks - arrival[np.clip(idx, 0, None)], np.inf)
        stale = age > max_age_s
        stats[camera] = {'stale_ticks': int(stale.sum()), 'max_age_s': float(np.max(age)),
                         'median_age_s': float(np.median(age[valid])) if valid.any() else None,
                         'unique_frames': int(len(np.unique(idx[valid])))}
        indices[camera] = idx
    return ticks, state.astype(np.float32), action.astype(np.float32), indices, stats


def profile_reference(meta):
    path = RIG_ROOT / meta['profile']   # the controller's joint-reference profile (rig.py)
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta['profile_sha256']:
        raise ValueError(f'profile {path} changed since recording (sha256 mismatch)')
    joints = json.loads(path.read_text())['joints']
    return np.array([joints[j]['reference_raw_rad'] for j in JOINTS], dtype=float)


def episode_dirs(sessions):
    for session in sessions:
        for meta_path in sorted((session / 'episodes').glob('*/meta.json')):
            yield meta_path.parent


def prepare_episode(episode_dir, fps, max_age_s, max_stale_fraction, max_gap_s=0.25):
    meta = json.loads((episode_dir / 'meta.json').read_text())
    trace = json.loads((episode_dir / 'mirror_trace.json').read_text())
    if trace.get('outcome') not in ('completed', 'operator_stop_returned'):  # Ctrl+C after a finished demo
        raise ValueError(f"mirror outcome {trace.get('outcome')}")
    t, actual, sent = load_follow(trace)
    frames = {camera: load_frames(episode_dir, camera) for camera in CAMERAS}
    window = trim_window(t, actual)
    reference = profile_reference(meta)
    ticks, state, action, indices, stats = build_timeline(t, actual, sent, reference, frames, fps, window, max_age_s)
    worst = max(s['stale_ticks'] for s in stats.values()) / len(ticks)
    report = {'episode_dir': str(episode_dir), 'plan': meta.get('plan'), 'mat_offset_cm': meta.get('mat_offset_cm'),
              'frames': len(ticks),
              'duration_s': float(ticks[-1] - ticks[0]), 'trim_window_rel_s': [window[0] - t[0], window[1] - t[0]],
              'cameras': stats, 'stale_fraction': worst,
              'max_abs_sent_minus_actual_rad': np.max(np.abs(sent - actual), axis=0).round(4).tolist(),
              'state_min': state.min(axis=0).round(3).tolist(), 'state_max': state.max(axis=0).round(3).tolist()}
    report['ok'] = worst <= max_stale_fraction and max(s['max_age_s'] for s in stats.values()) <= max_gap_s
    return meta, ticks, state, action, frames, indices, report


def features(height, width):
    vector = {'dtype': 'float32', 'shape': (7,), 'names': NAMES}
    image = {'dtype': 'video', 'shape': (height, width, 3), 'names': ['height', 'width', 'channels']}
    return {'observation.state': dict(vector), 'action': dict(vector),
            **{key: dict(image) for key in CAMERAS.values()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--session', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True, help='new dataset root (must not exist)')
    parser.add_argument('--repo-id', default='local/rebot_t1_pick_place_v1')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--max-age', type=float, default=0.1, help='max camera frame age at a tick, s')
    parser.add_argument('--max-stale-fraction', type=float, default=0.02)
    parser.add_argument('--max-gap', type=float, default=0.25, help='reject if any tick sees a frame older than this, s')
    parser.add_argument('--check-only', action='store_true', help='write only the alignment report')
    args = parser.parse_args()

    prepared, reports = [], []
    for episode_dir in episode_dirs(args.session):
        meta = json.loads((episode_dir / 'meta.json').read_text())
        if not meta.get('accepted'):
            reports.append({'episode_dir': str(episode_dir), 'skipped': meta.get('reason', 'not accepted')})
            continue
        try:
            item = prepare_episode(episode_dir, args.fps, args.max_age, args.max_stale_fraction, args.max_gap)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            reports.append({'episode_dir': str(episode_dir), 'error': repr(exc), 'ok': False})
            continue
        reports.append(item[-1])
        if item[-1]['ok']:
            prepared.append(item)
    for row in reports:
        status = 'SKIP' if 'skipped' in row else ('OK' if row.get('ok') else 'BAD')
        detail = row.get('skipped') or row.get('error') or (
            f"{row['frames']} frames {row['duration_s']:.1f}s stale={row['stale_fraction']:.3f} "
            f"max_age={[round(s['max_age_s'], 3) for s in row['cameras'].values()]} "
            f"age_med={[round(s['median_age_s'] or 0, 3) for s in row['cameras'].values()]}")
        print(f"{status:4} {Path(row['episode_dir']).name}: {detail}")
    print(f'{len(prepared)} episodes convertible')
    report_path = args.out.with_name(args.out.name + '_report.json')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(reports, indent=1, ensure_ascii=False))
    print(f'report: {report_path}')
    if args.check_only or not prepared:
        return 0

    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    first_report, first_frames = prepared[0][-1], prepared[0][4]
    sample = Image.open(Path(first_report['episode_dir']) / first_frames['d405'][1][0])
    dataset = LeRobotDataset.create(repo_id=args.repo_id, fps=args.fps, root=args.out,
                                    robot_type='rebot_b601_rs_px6d_w2l', features=features(sample.height, sample.width),
                                    use_videos=True, image_writer_threads=4, vcodec='h264')
    for meta, ticks, state, action, frames, indices, report in prepared:
        episode_dir = Path(report['episode_dir'])
        for k in range(len(ticks)):
            frame = {'observation.state': state[k], 'action': action[k], 'task': meta['task']}
            for camera, key in CAMERAS.items():
                frame[key] = np.asarray(Image.open(episode_dir / frames[camera][1][indices[camera][k]]).convert('RGB'))
            dataset.add_frame(frame)
        dataset.save_episode()
        print(f'saved {episode_dir.name}: {len(ticks)} frames')
    dataset.finalize()
    print(f'dataset: {args.out} ({len(prepared)} episodes)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
