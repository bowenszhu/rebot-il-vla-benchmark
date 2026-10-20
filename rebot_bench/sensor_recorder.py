"""Read-only D405/D435i + PX6D recorder for imitation-learning episodes.

No CAN or motor imports. Replaces the repair preview agent during collection:
it publishes the same status schema that the direct mirror's
``--camera-status`` freshness check reads, and additionally stores every RGB
frame (JPEG) and PX6D sample while ``<session>/ACTIVE_EPISODE`` names an
episode directory. Timestamps are ``time.monotonic_ns()`` (CLOCK_MONOTONIC),
the same clock the mirror trace uses, so episodes are aligned offline.
With ``--publish-frames`` it also shares each camera's newest frame
(frame_shm.py) for policy_server.py, which never opens the cameras itself.

    conda run -n lerobot python -B rebot_bench/sensor_recorder.py \
        --session sessions/t1_20261015 --d405-serial <serial> --d435i-serial <serial> --px6d-port <port>

The PX6D wire protocol (read_request / SampleStream) comes from the rig checkout (rig.py);
without --px6d-port the recorder runs cameras only and reports the PX6D as disabled.
"""
import argparse
import json
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CAMERAS = ('d405', 'd435i')


def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    tmp.replace(path)


def active_episode(session):
    """Episode directory named by ACTIVE_EPISODE, or None."""
    try:
        text = (session / 'ACTIVE_EPISODE').read_text().strip()
    except FileNotFoundError:
        return None
    return Path(text) if text else None


class EpisodeSink:
    """Per-stream writer thread so JPEG encoding never blocks capture."""

    def __init__(self, name, quality):
        self.name = name
        self.quality = quality
        self.queue = queue.Queue(maxsize=120)
        self.dropped = 0
        self.written = 0
        self.errors = []
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def put(self, episode, row, rgb):
        try:
            self.queue.put_nowait((episode, row, rgb))
        except queue.Full:
            self.dropped += 1

    def _run(self):
        from PIL import Image
        index_files = {}
        while True:
            item = self.queue.get()
            if item is None:
                break
            episode, row, rgb = item
            try:
                folder = episode / 'frames' / self.name
                if episode not in index_files:
                    folder.mkdir(parents=True, exist_ok=True)
                    index_files[episode] = (episode / f'{self.name}_frames.jsonl').open('a')
                name = f"{row['arrival_ns']}.jpg"
                Image.fromarray(rgb).save(folder / name, format='JPEG', quality=self.quality)
                index_files[episode].write(json.dumps({**row, 'file': f'frames/{self.name}/{name}'}) + '\n')
                index_files[episode].flush()
                self.written += 1
            except Exception as exc:  # recorded in status; capture keeps running
                self.errors.append(repr(exc))
        for handle in index_files.values():
            handle.close()

    def close(self):
        self.queue.put(None)
        self.thread.join(10)


def camera_worker(name, args, session, latest, lock, stop, state, sink, publisher=None):
    import numpy as np
    import pyrealsense2 as rs
    pipe = rs.pipeline()
    started = False
    state.update(errors=[], timeouts=0)
    try:
        cfg = rs.config()
        cfg.enable_device(getattr(args, f'{name}_serial'))
        cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
        profile = pipe.start(cfg)
        started = True
        dev = profile.get_device()
        state['device'] = {k: dev.get_info(getattr(rs.camera_info, k))
                           for k in ('name', 'serial_number', 'firmware_version', 'usb_type_descriptor')}
        while not stop.is_set():
            ok, frames = pipe.try_wait_for_frames(timeout_ms=100)
            arrival = time.monotonic_ns()
            if not ok:
                state['timeouts'] += 1
                continue
            color = frames.get_color_frame()
            if not color:
                continue
            row = {'camera': name, 'arrival_ns': arrival, 'frame_number': color.get_frame_number(),
                   'device_timestamp_ms': color.get_timestamp()}
            rgb = np.asanyarray(color.get_data()).copy()
            with lock:
                latest[name] = (row, rgb)
            if publisher is not None:
                try:
                    publisher.write(arrival, row['frame_number'], rgb)
                except Exception as exc:  # never let the shared copy stop capture
                    state['publish_errors'] = state.get('publish_errors', 0) + 1
                    state['publish_error'] = repr(exc)
            episode = active_episode(session)
            if episode is not None:
                sink.put(episode, row, rgb)
    except Exception as exc:
        state['errors'].append(repr(exc))
    finally:
        if started:
            pipe.stop()


def px6d_worker(session, stop, px, port_path):
    import serial
    from rebot_bench.rig import import_rig
    import_rig()
    from hardware.px6d_wrist_adapter.sensor.protocol import read_request, SampleStream
    count = 0
    logs = {}
    try:
        with serial.Serial(port_path, 921600, timeout=.25, exclusive=True) as port:
            if stop.wait(1.0):
                return
            port.reset_input_buffer()
            parser = SampleStream(127)
            while not stop.is_set():
                start = time.monotonic()
                port.write(read_request(127))
                port.flush()
                samples = []
                while time.monotonic() - start < .5 and not samples and not stop.is_set():
                    samples = parser.feed(port.read(max(1, port.in_waiting)))
                if stop.is_set():
                    break
                if not samples:
                    raise TimeoutError('No valid PX6D frame')
                count += 1
                row = {'sample_monotonic': time.monotonic(), 'wall_time': time.time(),
                       'channels_raw': samples[-1]['channels']}
                # Replace the whole dict: the status loop must never see a half-built sample.
                px['v'] = {**row, 'samples': count, 'state': 'live',
                           'axis_order': 'raw channels only; no calibration, no tare'}
                episode = active_episode(session)
                if episode is not None:
                    if episode not in logs:
                        logs[episode] = (episode / 'px6d.jsonl').open('a')
                    logs[episode].write(json.dumps(row) + '\n')
                stop.wait(max(0, .02 - (time.monotonic() - start)))
    except Exception as exc:
        px['v'] = {'state': 'error', 'error': repr(exc), 'samples': count, 'wall_time': time.time()}
    finally:
        for handle in logs.values():
            handle.close()


def px6d_retry_worker(session, stop, px, port_path):
    while not stop.is_set():
        px6d_worker(session, stop, px, port_path)
        if not stop.is_set():
            stop.wait(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--session', type=Path, required=True, help='session directory (created if absent)')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--jpeg-quality', type=int, default=92)
    parser.add_argument('--d405-serial', required=True, help='RealSense serial of the wrist D405')
    parser.add_argument('--d435i-serial', required=True, help='RealSense serial of the scene D435i')
    parser.add_argument('--px6d-port', help='PX6D RS485 adapter serial port (omit: no force logging)')
    parser.add_argument('--publish-frames', type=Path, nargs='?', const=Path('/dev/shm/rebot_il_frames'),
                        help='share the newest frame per camera for policy_server.py (default dir /dev/shm/rebot_il_frames)')
    args = parser.parse_args()

    session = args.session.resolve()
    session.mkdir(parents=True, exist_ok=True)
    (session / 'preview').mkdir(exist_ok=True)
    if (session / 'ACTIVE_EPISODE').exists():
        raise SystemExit(f'{session}/ACTIVE_EPISODE exists; finish or remove the stale episode marker first')
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    lock = threading.Lock()
    latest, px = {}, {'v': {}}
    states = {name: {} for name in CAMERAS}
    sinks = {name: EpisodeSink(name, args.jpeg_quality) for name in CAMERAS}
    publishers = {name: None for name in CAMERAS}
    if args.publish_frames:
        from rebot_bench.frame_shm import FrameWriter
        publishers = {name: FrameWriter(args.publish_frames / f'{name}.frame', args.height, args.width)
                      for name in CAMERAS}
    threads = [threading.Thread(target=camera_worker, daemon=True,
                                args=(name, args, session, latest, lock, stop, states[name], sinks[name], publishers[name]))
               for name in CAMERAS]
    if args.px6d_port:
        threads.append(threading.Thread(target=px6d_retry_worker, args=(session, stop, px, args.px6d_port), daemon=True))
    else:
        px['v'] = {'state': 'disabled'}
    for thread in threads:
        thread.start()
    status_path = session / 'sensor_status.json'
    atomic_json(session / 'recorder_owner.json', {'pid': os.getpid(), 'stop_file': str(session / 'STOP_RECORDER'),
                                                  'width': args.width, 'height': args.height, 'fps': args.fps})
    print(f'RECORDER RUNNING: status={status_path}; Ctrl+C or touch {session}/STOP_RECORDER to stop', flush=True)
    last_preview = 0.0
    try:
        from PIL import Image
        while not stop.is_set() and not (session / 'STOP_RECORDER').exists():
            with lock:
                samples = dict(latest)
            episode = active_episode(session)
            status = {'updated_at': time.time(), 'state': 'running', 'cameras': {}, 'px6d': dict(px['v']),
                      'active_episode': None if episode is None else str(episode),
                      'frame_shm': None if not args.publish_frames else str(args.publish_frames)}
            write_preview = time.monotonic() - last_preview > .5
            for name in CAMERAS:
                cs = states[name]
                info = {'device': cs.get('device'), 'errors': cs.get('errors', []), 'state': 'starting',
                        'saved': sinks[name].written, 'dropped': sinks[name].dropped,
                        'writer_errors': sinks[name].errors[-3:],
                        'publish_errors': cs.get('publish_errors', 0), 'publish_error': cs.get('publish_error')}
                if name in samples:
                    row, rgb = samples[name]
                    age = (time.monotonic_ns() - row['arrival_ns']) / 1e9
                    info.update(state='live' if age < 1 else 'stale', age_s=age, frame=row['frame_number'],
                                arrival_ns=row['arrival_ns'])
                    if write_preview:
                        tmp = session / 'preview' / f'{name}.tmp'
                        Image.fromarray(rgb).save(tmp, format='JPEG', quality=80)
                        tmp.replace(session / 'preview' / f'{name}.jpg')
                status['cameras'][name] = info
            if write_preview:
                last_preview = time.monotonic()
            atomic_json(status_path, status)
            stop.wait(.1)
    finally:
        stop.set()
        for thread in threads:
            thread.join(5)
        for sink in sinks.values():
            sink.close()
        atomic_json(status_path, {'state': 'stopped', 'updated_at': time.time(), 'px6d': dict(px['v']),
                                  'cameras': {n: {'errors': s.get('errors', []), 'saved': sinks[n].written,
                                                  'dropped': sinks[n].dropped} for n, s in states.items()}})
        print('RECORDER STOPPED', flush=True)


if __name__ == '__main__':
    main()
