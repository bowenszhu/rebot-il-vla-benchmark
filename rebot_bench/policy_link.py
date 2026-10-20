"""Control <-> policy process link (stdlib only; rebot py3.11 and lerobot-blackwell py3.10).

One AF_UNIX SOCK_SEQPACKET socket, one JSON object per packet. The policy
server listens; the control process (evaluate.py -> run_mirror) connects.
Sends from the control loop use MSG_DONTWAIT, so a hung policy can never block
a command tick: the packet is dropped and the action simply goes stale.

Coordinates match convert_episodes.py: q = raw - profile reference_raw_rad,
7 values ordered j1..j6, gripper(J7), radians.

control -> policy                         policy -> control
  {'type': 'warmup', 'q': [7], 'task'?}     {'type': 'hello', 'policy': {...}}
  {'type': 'reset', 'log': path|None,       {'type': 'warmup_ok'|'error', ...}
   'task'?}
  {'type': 'obs', 'seq', 't', 'q': [7]}     {'type': 'action', 'seq', 'obs_seq', 'obs_t',
  {'type': 'bye'}                              'q': [7], 'infer_ms', 'frame_age_s'}
                                            {'type': 'skip', 'reason'}
't' values are time.monotonic() (CLOCK_MONOTONIC is shared on one host).
'task' (optional) is this trial's language instruction; without it the server
uses its startup --task.
"""
import json
import math
import os
import socket
import threading
import time

DEFAULT_SOCKET = '/tmp/rebot_il_policy.sock'
ACTION_DIM = 7
MAX_PACKET = 65536
# Hold after this long without a new action; leave follow (normal return) after
# POLICY_LOST_S. A healthy ACT run delivers every ~33 ms; Diffusion DDIM10 stalls
# ~0.2 s at chunk boundaries, SmolVLA ~0.11 s (README measurements).
POLICY_TIMEOUT_S = .5
POLICY_LOST_S = 2.0


def encode(message):
    data = json.dumps(message, separators=(',', ':')).encode()
    if len(data) > MAX_PACKET:
        raise ValueError('policy link packet too large')
    return data


def decode(packet):
    message = json.loads(packet.decode())
    if not isinstance(message, dict) or not isinstance(message.get('type'), str):
        raise ValueError('policy link packet is not a typed object')
    return message


def finite_action(values):
    """Exactly 7 finite floats or ValueError; never repaired or clipped here."""
    if not isinstance(values, list) or len(values) != ACTION_DIM:
        raise ValueError(f'policy action must have {ACTION_DIM} values')
    out = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('policy action contains a non-finite value')
        out.append(float(value))
    return out


def link_state(now, last_action_at, begun_at, dead, timeout_s, lost_s):
    """'live' | 'waiting' (no action yet) | 'hold' | 'lost' from action age alone."""
    since = now - (begun_at if last_action_at is None else last_action_at)
    if dead or since > lost_s:
        return 'lost', since
    if since > timeout_s:
        return 'hold', since
    return ('waiting' if last_action_at is None else 'live'), since


class PolicyClient:
    """Control-process side. Only connect()/preflight() raise (before enabling)."""

    def __init__(self, path=DEFAULT_SOCKET, timeout_s=POLICY_TIMEOUT_S, lost_s=POLICY_LOST_S,
                 log_path=None, clock=time.monotonic, task=None):
        if not 0 < timeout_s < lost_s <= 5:
            raise ValueError('policy timeout/lost must satisfy 0 < timeout < lost <= 5 s')
        self.path, self.timeout_s, self.lost_s = str(path), timeout_s, lost_s
        self.log_path = None if log_path is None else str(log_path)
        self.clock = clock
        self.task = task
        self.sock = None
        self.hello = None
        self.lock = threading.Lock()
        self.latest = None          # (receive time, action message)
        self.replies = []           # warmup replies
        self.reply_event = threading.Event()
        self.dead = None            # reason string once the link is unusable
        self.begun_at = None
        self.obs_seq = 0
        self.counts = {'obs_sent': 0, 'obs_dropped': 0, 'actions': 0, 'bad_actions': 0, 'skips': 0,
                       'hold_steps': 0, 'waiting_steps': 0}
        self.last_skip = None
        self.reader = None

    # -- before enabling (may raise) ------------------------------------------------
    def connect(self, timeout=5.0):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        sock.settimeout(timeout)
        try:
            sock.connect(self.path)
            hello = decode(sock.recv(MAX_PACKET))
        except (OSError, ValueError) as exc:
            sock.close()
            raise RuntimeError(f'policy server not reachable at {self.path}: {exc}') from exc
        if hello['type'] != 'hello':
            sock.close()
            raise RuntimeError(f'unexpected policy greeting {hello}')
        sock.settimeout(None)
        self.sock, self.hello = sock, hello
        self.reader = threading.Thread(target=self._read, daemon=True, name='policy-link')
        self.reader.start()
        return hello

    def preflight(self, q, timeout=15.0):
        """One warm inference on the startup state and live frames; raises before enabling."""
        if self.sock is None:
            self.connect()
        self.reply_event.clear()
        message = {'type': 'warmup', 'q': [q[j] for j in range(1, ACTION_DIM+1)]}
        if self.task is not None:
            message['task'] = self.task
        self.sock.sendall(encode(message))
        if not self.reply_event.wait(timeout):
            raise RuntimeError('policy warmup timed out; nothing enabled')
        with self.lock:
            reply = self.replies.pop() if self.replies else None
        if reply is None or self.dead:
            raise RuntimeError(f'policy link closed during warmup: {self.dead}')
        if reply['type'] != 'warmup_ok':
            raise RuntimeError(f"policy warmup failed: {reply.get('message')}")
        finite_action(reply.get('q'))
        return reply

    # -- control loop (never raises) -----------------------------------------------------
    def begin(self):
        with self.lock:
            self.latest = None
            self.begun_at = self.clock()
        message = {'type': 'reset', 'log': self.log_path}
        if self.task is not None:
            message['task'] = self.task
        if not self._send(message, control=True):
            self._kill('reset not delivered')

    def step(self, now, q):
        """Send the measured state; return the newest action and the link state."""
        self.obs_seq += 1
        if self.dead is None:
            sent = self._send({'type': 'obs', 'seq': self.obs_seq, 't': now,
                               'q': [q[j] for j in range(1, ACTION_DIM+1)]})
            self.counts['obs_sent' if sent else 'obs_dropped'] += 1
        with self.lock:
            latest = self.latest
            begun = self.begun_at if self.begun_at is not None else now
        state, age = link_state(self.clock(), None if latest is None else latest[0], begun,
                                self.dead is not None, self.timeout_s, self.lost_s)
        if state in ('hold', 'waiting'):
            self.counts[f'{state}_steps'] += 1
        row = {'state': state, 'age_s': round(age, 4), 'obs_seq': self.obs_seq,
               'obs_q': [q[j] for j in range(1, ACTION_DIM+1)]}
        if latest is not None:
            message = latest[1]
            row.update(action=message['q'], action_seq=message.get('seq'), action_obs_seq=message.get('obs_seq'),
                       infer_ms=message.get('infer_ms'),
                       obs_to_action_ms=None if message.get('obs_t') is None else round(1000*(latest[0]-message['obs_t']), 2))
        if state == 'lost':
            row['reason'] = self.dead or f'no action for {age:.2f}s'
        elif state == 'hold' and self.last_skip:
            row['reason'] = self.last_skip
        return row

    def summary(self):
        return {'socket': self.path, 'hello': self.hello, 'task': self.task, 'timeout_s': self.timeout_s, 'lost_s': self.lost_s,
                'dead': self.dead, 'counts': dict(self.counts), 'last_skip': self.last_skip}

    def close(self):
        if self.sock is None:
            return
        self._send({'type': 'bye'}, control=True)
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()
        if self.reader is not None:
            self.reader.join(1)
        self.sock = None

    # -- internals ---------------------------------------------------------------------------
    def _send(self, message, control=False):
        """Non-blocking whole-packet send; control messages get a few short retries."""
        data = encode(message)
        for attempt in range(20 if control else 1):
            try:
                self.sock.send(data, socket.MSG_DONTWAIT)
                return True
            except BlockingIOError:
                if control:
                    time.sleep(.005)
            except OSError as exc:
                self._kill(f'send failed: {exc}')
                return False
        return False

    def _kill(self, reason):
        if self.dead is None:
            self.dead = reason
        self.reply_event.set()

    def _read(self):
        while True:
            try:
                packet = self.sock.recv(MAX_PACKET)
            except OSError as exc:
                self._kill(f'receive failed: {exc}')
                return
            if not packet:
                self._kill('policy server closed the link')
                return
            try:
                message = decode(packet)
            except ValueError as exc:
                self._kill(f'bad packet: {exc}')
                return
            kind = message['type']
            if kind == 'action':
                try:
                    message['q'] = finite_action(message.get('q'))
                except ValueError as exc:
                    # A malformed action is a policy failure, not a value to clip.
                    self.counts['bad_actions'] += 1
                    self._kill(f'invalid action: {exc}')
                    return
                with self.lock:
                    self.latest = (self.clock(), message)
                self.counts['actions'] += 1
            elif kind in ('warmup_ok', 'error'):
                with self.lock:
                    self.replies.append(message)
                self.reply_event.set()
            elif kind == 'skip':
                self.counts['skips'] += 1
                self.last_skip = message.get('reason')


class LinkServer:
    """Policy-process side socket: one control client at a time."""

    def __init__(self, path=DEFAULT_SOCKET):
        self.path = str(path)
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            probe.connect(self.path)
        except OSError:
            pass
        else:
            probe.close()
            raise RuntimeError(f'another policy server is listening on {self.path}')
        finally:
            probe.close()
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.sock.bind(self.path)
        self.sock.listen(1)

    def accept(self, timeout=None):
        self.sock.settimeout(timeout)
        try:
            conn, _ = self.sock.accept()
        except socket.timeout:
            return None
        conn.settimeout(None)
        return conn

    @staticmethod
    def drain(conn):
        """All pending packets without blocking; None marks a closed client."""
        messages = []
        while True:
            try:
                packet = conn.recv(MAX_PACKET, socket.MSG_DONTWAIT)
            except BlockingIOError:
                return messages
            except OSError:
                messages.append(None)
                return messages
            if not packet:
                messages.append(None)
                return messages
            messages.append(decode(packet))

    @staticmethod
    def send(conn, message):
        try:
            conn.send(encode(message), socket.MSG_DONTWAIT)
            return True
        except (BlockingIOError, OSError):
            return False

    def close(self):
        self.sock.close()
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass
