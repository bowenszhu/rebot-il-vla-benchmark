"""Policy process for closed-loop evaluation (lerobot-blackwell, GPU; no CAN/motor imports).

Reads the newest D405/D435i frames that sensor_recorder.py --publish-frames
shares (frame_shm.py; this process never opens a camera) and the measured joint
state that the control process streams over policy_link.py, runs a LeRobot
policy at the dataset rate (30 Hz) and returns one action per tick. Inputs and
outputs use the convert_episodes.py definitions: observation.state / action =
raw - profile reference (rad; j1..j6, gripper), observation.images.wrist =
D405, observation.images.scene = D435i.

Stale state (control gone) means no inference; stale or missing frames mean a
'skip' instead of an action, so the control side holds and then returns. The
server outlives individual trials; Ctrl+C stops it.

    conda run --no-capture-output -n lerobot-blackwell python -B rebot_bench/policy_server.py \\
        --policy outputs/train/act_t1/checkpoints/last/pretrained_model

pi0.5 needs the lerobot-pi env (transformers fix/lerobot_openpi branch, pi05_env_patch.sh).

--chunk-mode (default sync: one chunk, n_action_steps actions, then a blocking inference) can instead
infer the next chunk in the background while the current one keeps playing (ChunkStream): rtc steers
the new chunk onto the old one's leftover with lerobot's real-time chunking, blend crossfades them.
"""
import argparse
import hashlib
import json
import math
import select
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rebot_bench.convert_episodes import CAMERAS, NAMES  # noqa: E402
from rebot_bench.frame_shm import DEFAULT_DIR, read_frame  # noqa: E402
from rebot_bench.policy_link import (ACTION_DIM, DEFAULT_SOCKET, LinkServer,  # noqa: E402
                                                      finite_action)

PROTOCOL = ROOT / 'protocols/t1_protocol.json'
ROBOT_TYPE = 'rebot_b601_rs_px6d_w2l'
# Control streams state at 50 Hz; an older state means its loop is not running.
OBS_MAX_AGE_S = .1
# convert_episodes --max-gap: a demo with an older frame at any tick is rejected,
# so the policy never saw one in training.
FRAME_MAX_AGE_S = .25


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def parse_override(text):
    key, _, value = text.partition('=')
    if not key or not _:
        raise argparse.ArgumentTypeError('override must be key=value')
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


class ChunkStream:
    """One action per tick from overlapping chunks (no torch; ticks are calls, not wall time).

    A chunk launched at tick t from that tick's observation has its index j belong to tick t+j. It is
    merged at the first tick after it is ready, k ticks after launch, and plays from its own index k on;
    the old chunk's actions for ticks t..t+k-1 were played meanwhile. blend > 0 crossfades the first
    `blend` merged actions linearly from the old leftover to the new chunk.
    """

    def __init__(self, refill, blend=0):
        if refill < 0 or blend < 0:
            raise ValueError('refill and blend must be >= 0')
        self.refill, self.blend = refill, blend
        self.reset()

    def reset(self):
        self.actions, self.raw, self.pos, self.tick, self.chunk = None, None, 0, 0, 0

    def remaining(self):
        return 0 if self.actions is None else len(self.actions) - self.pos

    def due(self):
        """Time to launch the next chunk."""
        return self.remaining() <= self.refill

    def leftover(self):
        """Unplayed model-space actions, the first one being this tick's (rtc prev_chunk_left_over)."""
        return None if self.raw is None or self.remaining() == 0 else self.raw[self.pos:]

    def merge(self, actions, raw, launched):
        k = min(self.tick - launched, len(actions) - 1)
        new = np.array(actions[k:], dtype=float)
        n = min(self.blend, self.remaining(), len(new))
        if n:
            weight = (np.arange(1, n + 1) / (n + 1))[:, None]   # of the new chunk
            new[:n] = (1 - weight)*self.actions[self.pos:self.pos + n] + weight*new[:n]
        self.actions, self.raw, self.pos = new, None if raw is None else raw[k:], 0
        self.chunk += 1
        return k

    def pop(self):
        action = self.actions[self.pos]
        self.pos += 1
        self.tick += 1
        return action


class LeRobotEngine:
    """ACT / Diffusion / SmolVLA / pi0.5 through the generic LeRobot inference path."""

    def __init__(self, path, device='cuda', task='', rename_map=None, overrides=None,
                 chunk_mode='sync', refill=25, blend=0, fps=30):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        from lerobot.utils.control_utils import predict_action
        self.torch, self.predict_action = torch, predict_action
        self.path, self.task, self.rename_map = Path(path), task, dict(rename_map or {})
        self.overrides = dict(overrides or {})
        config = PreTrainedConfig.from_pretrained(self.path)
        for key, value in self.overrides.items():
            if not hasattr(config, key):
                raise ValueError(f'{config.type} config has no field {key!r}')
            setattr(config, key, value)
        config.device = device
        self.config = config
        self.device = torch.device(device)
        self.chunk_mode, self.stream = chunk_mode, None
        if chunk_mode != 'sync':
            if refill >= config.chunk_size:
                raise ValueError(f'refill {refill} must be below chunk_size {config.chunk_size}')
            self.stream = ChunkStream(refill, blend if chunk_mode == 'blend' else 0)
            self.worker, self.result, self.delays, self.step_info, self.chunk_ms = None, None, [], None, None
            self.fps = fps
        if chunk_mode == 'rtc':
            if not hasattr(config, 'rtc_config'):
                raise ValueError(f'{config.type} has no real-time chunking')
            from lerobot.policies.rtc.configuration_rtc import RTCConfig
            # Guide over the whole leftover: refill ticks remain when a chunk is launched.
            config.rtc_config = RTCConfig(enabled=True, execution_horizon=refill)
        provided = {self.rename_map.get(key, key) for key in ('observation.state', *CAMERAS.values())}
        if config.type == 'pi05':
            self.policy = self.load_pi05(get_policy_class(config.type), config, provided)
        else:
            self.policy = get_policy_class(config.type).from_pretrained(self.path, config=config)
        if chunk_mode == 'rtc':
            # RTC denoises under enable_grad; frozen weights keep that from recording a graph.
            self.policy.requires_grad_(False)
        self.policy.to(self.device)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            config, pretrained_path=str(self.path),
            preprocessor_overrides={'device_processor': {'device': device},
                                    'rename_observations_processor': {'rename_map': self.rename_map}})
        self.input_keys = sorted(getattr(config, 'input_features', {}) or {})
        unfed = [key for key in self.input_keys if key not in provided]
        if unfed:  # e.g. SmolVLA's unused camera3 slot; the policy decides whether that is valid
            print(f'WARNING: policy inputs not provided by this rig: {unfed}', flush=True)

    def load_pi05(self, policy_cls, config, provided):
        """pi0.5 in bf16 next to W2L's CUDA backend on the 8 GB card (2026-10-18: 6.5 GiB + W2L 0.5 GiB).

        Loading straight onto the GPU peaks with the unused action-expert lm_head and runs out of memory
        while W2L is up, so load on the CPU, drop what inference never touches, then move. Camera slots this
        rig never feeds (pi05_base right_wrist_0_rgb) are dropped too: they were blank images with a zero
        mask in training, padding-masked out of the prefix, so the actions are unchanged.
        """
        if config.dtype != 'bfloat16':
            print(f'pi05: dtype {config.dtype} -> bfloat16 (float32 does not fit in 8 GB)', flush=True)
            config.dtype = 'bfloat16'
        for key in [key for key in config.image_features if key not in provided]:
            print(f'pi05: dropping unfed camera slot {key}', flush=True)
            del config.input_features[key]
        config.device = 'cpu'
        # strict=False: the checkpoint stores the tied PaliGemma embedding once (lm_head), so the loader
        # reports embed_tokens as missing; the tie is checked below instead.
        policy = policy_cls.from_pretrained(self.path, config=config, strict=False)
        paligemma = policy.model.paligemma_with_expert.paligemma
        if paligemma.model.language_model.embed_tokens.weight is not paligemma.lm_head.weight:
            raise RuntimeError('pi05: language embedding is not tied to lm_head; it may be uninitialised')
        policy.model.paligemma_with_expert.gemma_expert.lm_head = None  # ~0.5 GiB, sampling never uses it
        config.device = str(self.device)
        return policy

    def reset(self):
        if self.stream is not None:
            if self.worker is not None:   # a chunk for the previous trial; let it finish, then drop it
                self.worker.join()
            self.worker, self.result, self.step_info = None, None, None
            self.stream.reset()
        self.policy.reset()
        self.pre.reset()
        self.post.reset()

    def step(self, state, images):
        observation = {'observation.state': np.asarray(state, dtype=np.float32), **images}
        if self.stream is not None:
            return self.step_chunked(observation)
        action = self.predict_action(observation, self.policy, self.device, self.pre, self.post,
                                     use_amp=False, task=self.task, robot_type=ROBOT_TYPE)
        return action.squeeze(0).to('cpu').numpy().astype(float).tolist()

    def infer_chunk(self, observation, leftover=None, delay=0):
        """(actions (chunk_size, 7) float ndarray, model-space chunk tensor, ms) for one observation."""
        from copy import copy
        from lerobot.policies.utils import prepare_observation_for_inference
        torch = self.torch
        start = time.monotonic()
        # no_grad, not inference_mode: rtc needs autograd for its guidance step.
        with torch.no_grad():
            batch = prepare_observation_for_inference(copy(observation), self.device, self.task, ROBOT_TYPE)
            batch = self.pre(batch)
            kwargs = {}
            if self.chunk_mode == 'rtc':
                kwargs = {'inference_delay': delay, 'prev_chunk_left_over': leftover,
                          'execution_horizon': self.stream.refill}
            raw = self.policy.predict_action_chunk(batch, **kwargs)[0]
            actions = self.post(raw).to('cpu').numpy().astype(float)
        return actions, raw, 1000*(time.monotonic() - start)

    def delay_estimate(self):
        """Ticks a background chunk is expected to take (rtc inference_delay): the slowest of the last few
        takeovers; before any, this trial's first (blocking, already warm) chunk."""
        return max(self.delays, default=math.ceil(self.chunk_ms/1000*self.fps))

    def launch(self, observation):
        stream, launched = self.stream, self.stream.tick
        leftover, delay = stream.leftover(), self.delay_estimate()

        def run():
            try:
                self.result = (launched, *self.infer_chunk(observation, leftover, delay))
            except BaseException as exc:  # re-raised on the serving thread
                self.result = (launched, exc)
        self.worker = threading.Thread(target=run, name='chunk-inference', daemon=True)
        self.worker.start()

    def collect(self, wait=False):
        if self.worker is None or (self.worker.is_alive() and not wait):
            return
        self.worker.join()
        result, self.worker, self.result = self.result, None, None
        if isinstance(result[1], BaseException):
            raise result[1]
        launched, actions, raw, ms = result
        self.delays = self.delays[-4:] + [self.stream.merge(actions, raw, launched)]
        self.chunk_ms = round(ms, 1)

    def step_chunked(self, observation):
        stream = self.stream
        self.collect()
        if self.worker is None and stream.remaining() == 0:
            # Start of a trial: nothing to play while waiting, so infer here (as sync mode does).
            actions, raw, ms = self.infer_chunk(observation)
            stream.merge(actions, raw, stream.tick)
            self.chunk_ms = round(ms, 1)
        elif self.worker is None and stream.due():
            self.launch(observation)
        if stream.remaining() == 0:   # background chunk is late: hold here until it lands
            self.collect(wait=True)
        self.step_info = {'chunk': stream.chunk, 'chunk_pos': stream.pos, 'chunk_infer_ms': self.chunk_ms}
        return [float(x) for x in stream.pop()]

    def describe(self):
        if getattr(self, 'weights_sha256', None) is None:  # hashed once per server
            weights = self.path / 'model.safetensors'
            self.weights_sha256 = sha256(weights) if weights.exists() else 'missing'
        return {'type': self.config.type, 'path': str(self.path.resolve()),
                'model_sha256': self.weights_sha256,
                'n_action_steps': getattr(self.config, 'n_action_steps', None),
                'chunk_size': getattr(self.config, 'chunk_size', None),
                'device': str(self.device), 'task': self.task, 'rename_map': self.rename_map,
                'overrides': self.overrides, 'input_keys': self.input_keys, 'action_names': NAMES,
                'chunk_mode': self.chunk_mode,
                **({} if self.stream is None else {'chunk_refill': self.stream.refill, 'chunk_blend': self.stream.blend})}


class PolicyServer:
    def __init__(self, engine, frames_dir=DEFAULT_DIR, fps=30, clock=time.monotonic):
        self.engine, self.frames_dir, self.period, self.clock = engine, Path(frames_dir), 1.0/fps, clock
        self.fps = fps
        self.default_task = getattr(engine, 'task', None)

    def use_task(self, message):
        """Per-trial instruction from warmup/reset; the startup --task otherwise."""
        task = message.get('task') or self.default_task
        if task != getattr(self.engine, 'task', None):
            print(f'task: {task}', flush=True)
        self.engine.task = task

    def frames(self):
        """{observation.images.*: rgb}, {camera: age_s}, or a reason string."""
        images, ages = {}, {}
        for camera, key in CAMERAS.items():
            try:
                item = read_frame(self.frames_dir / f'{camera}.frame')
            except (OSError, RuntimeError) as exc:
                return None, None, f'{camera} frame unavailable: {exc}'
            if item is None:
                return None, None, f'{camera} has no frame yet'
            meta, rgb = item
            ages[camera] = time.monotonic_ns()/1e9 - meta['arrival_ns']/1e9
            if not 0 <= ages[camera] <= FRAME_MAX_AGE_S:
                return None, ages, f'{camera} frame age {ages[camera]:.3f}s'
            images[key] = rgb
        return images, ages, None

    def infer(self, q):
        images, ages, reason = self.frames()
        if reason:
            return None, ages, None, reason
        start = self.clock()
        action = finite_action([float(x) for x in self.engine.step(np.asarray(q, dtype=np.float32), images)])
        return action, ages, round(1000*(self.clock()-start), 2), None

    def warmup(self, conn, message):
        try:
            q = finite_action(message.get('q'))
            self.use_task(message)
            self.engine.reset()
            action, ages, infer_ms, reason = self.infer(q)
            self.engine.reset()
        except Exception as exc:
            traceback.print_exc()
            reason = f'warmup inference failed: {exc!r}'
        if reason:
            LinkServer.send(conn, {'type': 'error', 'message': reason})
        else:
            LinkServer.send(conn, {'type': 'warmup_ok', 'q': action, 'infer_ms': infer_ms, 'frame_age_s': ages})
        print(f'warmup: {reason or f"ok {infer_ms} ms"}', flush=True)

    def serve_client(self, conn, stop=None):
        """Serve one control connection until bye/close; returns why it ended."""
        LinkServer.send(conn, {'type': 'hello', 'policy': self.engine.describe(), 'fps': self.fps,
                               'frames_dir': str(self.frames_dir), 'obs_max_age_s': OBS_MAX_AGE_S,
                               'frame_max_age_s': FRAME_MAX_AGE_S})
        active, latest, seq, next_tick, log, last_skip = False, None, 0, 0., None, 0.
        try:
            while stop is None or not stop.is_set():
                for message in LinkServer.drain(conn):
                    if message is None:
                        return 'closed'
                    kind = message['type']
                    if kind == 'warmup':
                        self.warmup(conn, message)
                    elif kind == 'reset':
                        self.use_task(message)
                        self.engine.reset()
                        active, latest, seq, next_tick = True, None, 0, self.clock()
                        if log is not None:
                            log.close()
                        log = open(message['log'], 'a') if message.get('log') else None
                    elif kind == 'obs':
                        latest = message
                    elif kind == 'bye':
                        return 'bye'
                now = self.clock()
                if active and now >= next_tick:
                    # Fixed cadence; an overrun starts a new period, never a burst.
                    next_tick = next_tick + self.period if next_tick + self.period > now else now + self.period
                    if latest is not None and 0 <= now - latest['t'] <= OBS_MAX_AGE_S:
                        try:
                            action, ages, infer_ms, reason = self.infer(latest['q'])
                        except Exception as exc:
                            traceback.print_exc()
                            action, ages, infer_ms, reason = None, None, None, f'inference failed: {exc!r}'
                            active = False   # needs a new reset; the control side holds and returns
                        row = {'t': self.clock(), 'obs_seq': latest['seq'], 'obs_t': latest['t'],
                               'frame_age_s': ages, 'infer_ms': infer_ms}
                        if reason:
                            if now - last_skip > .5 or not active:
                                LinkServer.send(conn, {'type': 'skip', 'reason': reason})
                                last_skip = now
                            row['skip'] = reason
                        else:
                            seq += 1
                            row.update(seq=seq, q=action, **(getattr(self.engine, 'step_info', None) or {}))
                            LinkServer.send(conn, {'type': 'action', **row})
                        if log is not None:
                            log.write(json.dumps(row) + '\n')
                wait = max(0., next_tick - self.clock()) if active else .05
                select.select([conn], [], [], wait)
        finally:
            if log is not None:
                log.close()
        return 'stopped'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--policy', type=Path, required=True, help='pretrained_model directory (config + weights + processors)')
    parser.add_argument('--socket', default=DEFAULT_SOCKET)
    parser.add_argument('--frames', type=Path, default=DEFAULT_DIR, help='sensor_recorder.py --publish-frames directory')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--task', help='language instruction when the control side sends none (default: t1_protocol.json task)')
    parser.add_argument('--rename-map', type=json.loads, default={},
                        help='observation key renames, e.g. SmolVLA base: \'{"observation.images.wrist": "observation.images.camera1", '
                             '"observation.images.scene": "observation.images.camera2"}\'')
    parser.add_argument('--override', type=parse_override, action='append', default=[],
                        help='policy config field, e.g. n_action_steps=50 or num_inference_steps=10 (repeatable)')
    parser.add_argument('--fps', type=int, help='policy rate (default: t1_protocol.json fps)')
    parser.add_argument('--chunk-mode', choices=('sync', 'rtc', 'blend'), default='sync',
                        help='sync: play n_action_steps, then infer (blocking); rtc / blend: infer the next chunk in the '
                             'background once --chunk-refill actions are left, joined by real-time chunking / a crossfade')
    parser.add_argument('--chunk-refill', type=int, default=25,
                        help='rtc/blend: launch the next chunk when this many actions of the current one are left')
    parser.add_argument('--chunk-blend', type=int, default=10, help='blend: crossfade length (ticks)')
    args = parser.parse_args()

    protocol = json.loads(PROTOCOL.read_text())
    fps = args.fps or protocol['fps']
    engine = LeRobotEngine(args.policy, args.device, args.task or protocol['task'], args.rename_map, dict(args.override),
                           args.chunk_mode, args.chunk_refill, args.chunk_blend, fps)
    # Load CUDA kernels before any client: one throwaway step on blank inputs.
    blank = {key: np.zeros((480, 640, 3), np.uint8) for key in CAMERAS.values()}
    start = time.monotonic()
    engine.reset()
    engine.step(np.zeros(ACTION_DIM, np.float32), blank)
    engine.reset()
    print(f'policy loaded: {json.dumps(engine.describe(), ensure_ascii=False)}', flush=True)
    print(f'first blank step {1000*(time.monotonic()-start):.0f} ms', flush=True)
    server = PolicyServer(engine, args.frames, fps)
    link = LinkServer(args.socket)
    print(f'POLICY SERVER READY on {args.socket} (frames {args.frames}); Ctrl+C to stop', flush=True)
    try:
        while True:
            conn = link.accept(timeout=1.0)
            if conn is None:
                continue
            print('control connected', flush=True)
            try:
                print(f'control link ended: {server.serve_client(conn)}', flush=True)
            finally:
                conn.close()
    except KeyboardInterrupt:
        print('POLICY SERVER STOPPED', flush=True)
    finally:
        link.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
