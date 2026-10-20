#!/usr/bin/env python3
"""Offline replay of chunk takeovers: how far the commanded gripper tip jumps when a new chunk takes over.

Feeds recorded episodes (one dataset frame per 30 Hz tick, open loop: the recorded robot does not follow
the policy) through policy_server.LeRobotEngine and measures the tip-height step between consecutive
actions with the rig's nominal kinematics (return_planner.ArmModel, the arm's MJCF plus the 26 cm tool,
on raw = action + profile reference). Both live in the rig checkout (rig.py), not in this repository. Steps are split into chunk takeovers
and within-chunk steps.

Variants (repeat --variant):
  sync[:N]           the default server path (N: n_action_steps override, option (c) without blending): play n_action_steps, then infer (blocking; the world waits,
                     as the robot holds during the real 0.3 s stall)
  rtc:REFILL         background chunk launched with REFILL actions left, real-time chunking
  blend:REFILL:N     background chunk, N-tick linear crossfade
Background chunks are inferred at launch but merged --delay-ticks later (simulated latency), so every
variant sees the same frames. --realtime TICKS also drives the real threaded engine at 30 Hz on the
first episode's frames and reports tick time, chunk latency and peak GPU memory.

  conda run -n lerobot-pi python -B rebot_bench/chunk_replay.py \\
      --policy <pi05 run>/checkpoints/020000/pretrained_model \\
      --dataset-root data/rebot_t2_v1 --episodes 0 10 20 --variant sync --variant rtc:25 \\
      --rename-map '{"observation.images.wrist": "observation.images.left_wrist_0_rgb", \\
                     "observation.images.scene": "observation.images.base_0_rgb"}'
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rebot_bench.convert_episodes import CAMERAS  # noqa: E402
from rebot_bench.policy_server import ChunkStream, LeRobotEngine  # noqa: E402
from rebot_bench.rig import PROFILE, RIG_ROOT, import_rig  # noqa: E402

import_rig()
from hardware.end_effectors.return_planner import MODEL, ArmModel  # noqa: E402


def parse_variant(text):
    mode, *rest = text.split(':')
    if mode == 'sync' and len(rest) <= 1:
        return {'name': text, 'mode': 'sync', 'n_action_steps': int(rest[0]) if rest else None}
    if mode == 'rtc' and len(rest) == 1:
        return {'name': text, 'mode': 'rtc', 'refill': int(rest[0]), 'blend': 0}
    if mode == 'blend' and len(rest) == 2:
        return {'name': text, 'mode': 'blend', 'refill': int(rest[0]), 'blend': int(rest[1])}
    raise argparse.ArgumentTypeError('variant: sync[:N] | rtc:REFILL | blend:REFILL:N')


class TickWorker:
    """Stands in for the inference thread: 'alive' until `delay` ticks after launch."""

    def __init__(self, stream, launched, delay):
        self.stream, self.launched, self.delay = stream, launched, delay

    def is_alive(self):
        return self.stream.tick - self.launched < self.delay

    def join(self):
        pass


def set_action_steps(engine, variant):
    if engine.default_steps is None:
        engine.default_steps = engine.policy.config.n_action_steps
    engine.policy.config.n_action_steps = variant.get('n_action_steps') or engine.default_steps


def use_variant(engine, variant, delay):
    """Switch one loaded engine between variants (rtc_config only toggles guidance)."""
    set_action_steps(engine, variant)
    engine.chunk_mode = variant['mode']
    engine.policy.config.rtc_config.enabled = variant['mode'] == 'rtc'
    engine.stream = None if variant['mode'] == 'sync' else ChunkStream(variant['refill'], variant['blend'])
    engine.worker, engine.result, engine.delays, engine.chunk_ms = None, None, [], None
    engine.chunk_log = []

    def launch(observation):
        launched, leftover = engine.stream.tick, engine.stream.leftover()
        actions, raw, ms = engine.infer_chunk(observation, leftover, delay)
        engine.chunk_log.append(ms)
        engine.result = (launched, actions, raw, ms)
        engine.worker = TickWorker(engine.stream, launched, delay)
    engine.launch = launch


def load_episode(dataset, frames):
    import torch
    n = len(dataset) if frames <= 0 else min(frames, len(dataset))
    states, actions, images = [], [], {key: [] for key in CAMERAS.values()}
    for i in range(n):
        item = dataset[i]
        states.append(item['observation.state'].numpy())
        actions.append(item['action'].numpy())
        for key in images:
            images[key].append((item[key].permute(1, 2, 0)*255).round().to(torch.uint8).numpy())
    task = item['task']
    return np.stack(states), np.stack(actions), {k: np.stack(v) for k, v in images.items()}, task


def run_episode(engine, variant, episode, seed):
    import torch
    states, recorded, images, _ = episode
    torch.manual_seed(seed)
    engine.reset()
    out, chunks = [], []
    for t in range(len(states)):
        out.append(engine.step(states[t], {key: images[key][t] for key in images}))
        if variant['mode'] == 'sync':
            chunks.append(t // engine.policy.config.n_action_steps)
        else:
            chunks.append(engine.step_info['chunk'])
    return np.array(out), np.array(chunks)


def tip_z_cm(model, reference, actions):
    return np.array([model.tool({j: float(a[j-1] + reference[j-1]) for j in range(1, 7)})[0][2]*100
                     for a in actions])


def stats(values):
    values = np.abs(np.asarray(values, dtype=float))
    if not values.size:
        return None
    return {'n': int(values.size), 'median': round(float(np.median(values)), 3),
            'p90': round(float(np.percentile(values, 90)), 3), 'p99': round(float(np.percentile(values, 99)), 3),
            'max': round(float(values.max()), 3)}


def realtime(engine, variant, episode, ticks, seed, fps=30):
    """The threaded engine at a fixed tick, as policy_server.serve_client calls it."""
    import torch
    states, _, images, _ = episode
    use_variant_real(engine, variant)
    torch.manual_seed(seed)
    engine.reset()
    torch.cuda.reset_peak_memory_stats()
    step_ms, merges, period, next_tick = [], [], 1/fps, time.monotonic()
    for t in range(min(ticks, len(states))):
        now = time.monotonic()
        if now < next_tick:
            time.sleep(next_tick - now)
        next_tick = next_tick + period if next_tick + period > time.monotonic() else time.monotonic() + period
        start = time.monotonic()
        engine.step(states[t], {key: images[key][t] for key in images})
        step_ms.append(1000*(time.monotonic() - start))
        if engine.stream is not None and engine.step_info['chunk'] != (merges[-1][0] if merges else 0):
            merges.append((engine.step_info['chunk'], engine.chunk_ms, list(engine.delays)[-1]))
    engine.reset()
    out = {'variant': variant['name'], 'ticks': len(step_ms), 'step_ms': stats(step_ms),
           'cuda_peak_allocated_gib': round(torch.cuda.max_memory_allocated()/2**30, 2),
           'cuda_peak_reserved_gib': round(torch.cuda.max_memory_reserved()/2**30, 2)}
    if merges:
        out['chunk_infer_ms'] = stats([m[1] for m in merges[1:]])
        out['takeover_delay_ticks'] = [m[2] for m in merges[1:]]
    return out


def use_variant_real(engine, variant):
    engine.__dict__.pop('launch', None)   # back to the threaded LeRobotEngine.launch
    set_action_steps(engine, variant)
    engine.chunk_mode = variant['mode']
    engine.policy.config.rtc_config.enabled = variant['mode'] == 'rtc'
    engine.stream = None if variant['mode'] == 'sync' else ChunkStream(variant['refill'], variant['blend'])
    engine.worker, engine.result, engine.delays, engine.chunk_ms = None, None, [], None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--episodes', type=int, nargs='+', default=[0])
    parser.add_argument('--frames', type=int, default=0, help='per episode; 0 = whole episode')
    parser.add_argument('--variant', type=parse_variant, action='append', required=True)
    parser.add_argument('--delay-ticks', type=int, default=10, help='simulated background inference latency')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--realtime', type=int, default=0, help='also run the threaded engine for this many ticks')
    parser.add_argument('--rename-map', type=json.loads, default={}, help='as policy_server.py --rename-map')
    parser.add_argument('--mjcf', type=Path, default=MODEL, help='arm MJCF (ReBot_Arm_web_RS submodule)')
    parser.add_argument('--json-out', type=Path)
    args = parser.parse_args()
    model = ArmModel(args.mjcf)

    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    # One load for every variant: rtc_config present (guidance toggled per variant), weights frozen.
    engine = LeRobotEngine(args.policy, 'cuda', rename_map=args.rename_map, chunk_mode='rtc', refill=max(
        [v.get('refill', 0) for v in args.variant] + [1]))
    engine.default_steps = None
    print(f'loaded; cuda allocated {torch.cuda.memory_allocated()/2**30:.2f} GiB', flush=True)
    reference = np.array([v['reference_raw_rad'] for v in
                          (json.loads(PROFILE.read_text())['joints'][str(j)] for j in range(1, 8))])
    root = args.dataset_root.resolve()
    per_variant = {v['name']: {'switch': [], 'within': [], 'mae': [], 'chunk_ms': []} for v in args.variant}
    demo_steps, report = [], {'policy': str(args.policy.resolve()), 'dataset': str(root),
                              'episodes': args.episodes, 'delay_ticks': args.delay_ticks, 'seed': args.seed,
                              'profile': str(PROFILE.relative_to(RIG_ROOT))}
    first = None
    for ep in args.episodes:
        episode = load_episode(LeRobotDataset(f'local/{root.name}', root=root, episodes=[ep]), args.frames)
        if first is None and args.realtime:
            first = episode
        engine.task = episode[3]
        demo_steps += list(np.diff(tip_z_cm(model, reference, episode[1])))
        for variant in args.variant:
            use_variant(engine, variant, args.delay_ticks)
            start = time.monotonic()
            actions, chunks = run_episode(engine, variant, episode, args.seed*1000 + ep)
            dz = np.diff(tip_z_cm(model, reference, actions))
            switch = chunks[1:] != chunks[:-1]
            acc = per_variant[variant['name']]
            acc['switch'] += list(dz[switch])
            acc['within'] += list(dz[~switch])
            acc['mae'].append(np.abs(actions - episode[1]).mean(axis=0))
            acc['chunk_ms'] += engine.chunk_log
            print(f'ep {ep} {variant["name"]}: {len(actions)} ticks, {int(switch.sum())} takeovers, '
                  f'switch |dz| median {np.median(np.abs(dz[switch])):.2f} cm, {time.monotonic()-start:.0f} s',
                  flush=True)
    report['demo_step_cm'] = stats(demo_steps)
    report['variants'] = {}
    for name, acc in per_variant.items():
        mae = np.mean(acc['mae'], axis=0)
        report['variants'][name] = {
            'switch_step_cm': stats(acc['switch']), 'within_step_cm': stats(acc['within']),
            'all_step_cm': stats(acc['switch'] + acc['within']),
            'mae_vs_demo_rad': [round(float(x), 4) for x in mae], 'mae_vs_demo_rad_mean': round(float(mae.mean()), 4),
            'background_chunk_ms': stats(acc['chunk_ms'])}
    if args.realtime:
        report['realtime'] = [realtime(engine, v, first, args.realtime, args.seed) for v in args.variant]
    text = json.dumps(report, indent=2)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
