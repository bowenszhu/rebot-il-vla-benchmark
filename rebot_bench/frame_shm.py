"""Newest RGB frame per camera, shared from sensor_recorder.py to policy_server.py.

The recorder keeps exclusive ownership of the D405/D435i (and PX6D, and the
sensor_status.json the mirror's freshness check reads); with
``--publish-frames DIR`` it also overwrites ``DIR/<camera>.frame`` with each
new frame. The file is a 64-byte header plus one HxWx3 uint8 frame under a
sequence lock: the writer makes the sequence odd while copying, readers retry
on a torn read. ``arrival_ns`` is the recorder's time.monotonic_ns() at arrival,
the same stamp the episode frame index uses, so readers compute frame age on
the shared CLOCK_MONOTONIC.
"""
import mmap
import os
import struct
import time
from pathlib import Path

import numpy as np

DEFAULT_DIR = Path('/dev/shm/rebot_il_frames')
MAGIC = b'RBFRAME1'
HEADER = struct.Struct('<8sQqqIII')   # magic, seq, arrival_ns, frame_number, height, width, channels
HEADER_SIZE = 64
SEQ_OFFSET = 8


class FrameWriter:
    def __init__(self, path, height, width, channels=3):
        self.path = Path(path)
        self.shape = (height, width, channels)
        self.size = height * width * channels
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.tmp')
        with open(tmp, 'wb') as handle:
            handle.truncate(HEADER_SIZE + self.size)
        self.handle = open(tmp, 'r+b')
        self.map = mmap.mmap(self.handle.fileno(), HEADER_SIZE + self.size)
        self.seq = 0
        HEADER.pack_into(self.map, 0, MAGIC, 0, 0, -1, height, width, channels)
        os.replace(tmp, self.path)  # readers of an old file see it go stale, never torn

    def write(self, arrival_ns, frame_number, rgb):
        if rgb.shape != self.shape or rgb.dtype != np.uint8:
            raise ValueError(f'frame {rgb.shape}/{rgb.dtype} does not match {self.shape}/uint8')
        self.seq += 1
        struct.pack_into('<Q', self.map, SEQ_OFFSET, self.seq)          # odd: write in progress
        self.map[HEADER_SIZE:HEADER_SIZE + self.size] = np.ascontiguousarray(rgb).tobytes()
        HEADER.pack_into(self.map, 0, MAGIC, self.seq, int(arrival_ns), int(frame_number), *self.shape)
        self.seq += 1
        struct.pack_into('<Q', self.map, SEQ_OFFSET, self.seq)          # even: consistent

    def close(self):
        self.map.close()
        self.handle.close()


def read_frame(path, retries=20):
    """(meta, rgb copy) of the newest frame, or None before the first frame.

    Raises FileNotFoundError when no recorder publishes there and RuntimeError
    when every attempt overlapped a write.
    """
    with open(path, 'rb') as handle:
        view = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    try:
        for _ in range(retries):
            magic, seq, arrival_ns, frame_number, height, width, channels = HEADER.unpack_from(view, 0)
            if magic != MAGIC:
                raise RuntimeError(f'{path} is not a frame file')
            if seq == 0:
                return None
            if seq % 2:
                time.sleep(.0005)
                continue
            size = height * width * channels
            rgb = np.frombuffer(view, dtype=np.uint8, count=size, offset=HEADER_SIZE).reshape(height, width, channels).copy()
            if struct.unpack_from('<Q', view, SEQ_OFFSET)[0] == seq:
                return {'arrival_ns': arrival_ns, 'frame_number': frame_number, 'seq': seq}, rgb
        raise RuntimeError(f'{path}: every read overlapped a write')
    finally:
        view.close()
