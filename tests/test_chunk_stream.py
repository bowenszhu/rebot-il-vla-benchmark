"""policy_server --chunk-mode rtc/blend scheduling (offline; a stub chunk model, no torch or GPU).

The stub's chunk launched from the observation of tick L holds L + j at index j, so a correctly
aligned stream plays exactly the tick number on every tick, whichever chunk it comes from and
however late a background chunk lands.
"""
import time
import unittest

import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:  # the analysis tests run on the standard library alone
    raise unittest.SkipTest('needs numpy')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rebot_bench.policy_server import ChunkStream, LeRobotEngine  # noqa: E402

CHUNK = 50


def chunk_from(tick, offset=0.):
    values = tick + np.arange(CHUNK, dtype=float) + offset
    return np.repeat(values[:, None], 7, axis=1)


class StubEngine(LeRobotEngine):
    """LeRobotEngine's scheduling with infer_chunk replaced; observation.state[0] is the tick."""

    def __init__(self, mode, refill, blend=0, latency_s=0., offset_after_first=0.):
        class Resettable:
            def reset(self):
                pass
        self.policy = self.pre = self.post = Resettable()
        self.chunk_mode, self.fps = mode, 30
        self.stream = ChunkStream(refill, blend)
        self.worker, self.result, self.delays, self.step_info, self.chunk_ms = None, None, [], None, None
        self.latency_s, self.offset_after_first, self.calls, self.fail = latency_s, offset_after_first, [], False

    def infer_chunk(self, observation, leftover=None, delay=0):
        tick = float(observation['observation.state'][0])
        self.calls.append({'tick': tick, 'leftover': None if leftover is None else leftover.copy(), 'delay': delay})
        if self.fail:
            raise RuntimeError('stub inference failed')
        if self.calls[1:]:
            time.sleep(self.latency_s)
        actions = chunk_from(tick, self.offset_after_first if self.calls[1:] else 0.)
        return actions, actions[:, :1].copy(), 1000*self.latency_s


def play(engine, ticks, period=0.):
    out = []
    for t in range(ticks):
        out.append(engine.step(np.full(7, t, np.float32), {})[0])
        time.sleep(period)
    return out


class ChunkStreamTest(unittest.TestCase):
    def test_late_chunk_plays_from_its_own_index(self):
        stream = ChunkStream(refill=20)
        stream.merge(chunk_from(0), chunk_from(0)[:, :1], launched=0)
        played = [stream.pop()[0] for _ in range(30)]
        self.assertTrue(stream.due())
        launched = stream.tick
        np.testing.assert_array_equal(stream.leftover()[:, 0], np.arange(30, 50))
        played += [stream.pop()[0] for _ in range(7)]   # inference takes 7 ticks
        self.assertEqual(stream.merge(chunk_from(launched), chunk_from(launched)[:, :1], launched), 7)
        played += [stream.pop()[0] for _ in range(stream.remaining())]
        self.assertEqual(played, list(range(len(played))))
        self.assertEqual(stream.chunk, 2)

    def test_blend_crossfades_linearly_into_the_new_chunk(self):
        stream = ChunkStream(refill=20, blend=4)
        stream.merge(chunk_from(0), None, 0)
        for _ in range(30):
            stream.pop()
        stream.merge(chunk_from(30, offset=5.), None, launched=30)   # new chunk sits 5 above the old one
        steps = [stream.pop()[0] - (30 + i) for i in range(6)]
        np.testing.assert_allclose(steps, [1., 2., 3., 4., 5., 5.])
        self.assertIsNone(stream.leftover())

    def test_rejects_negative_settings(self):
        with self.assertRaises(ValueError):
            ChunkStream(refill=-1)


class ChunkedEngineTest(unittest.TestCase):
    def test_background_chunks_stay_aligned_with_ticks(self):
        for mode, blend in (('rtc', 0), ('blend', 10)):
            with self.subTest(mode=mode):
                engine = StubEngine(mode, refill=25, blend=blend, latency_s=.06)
                engine.reset()
                played = play(engine, 150, period=.01)
                np.testing.assert_allclose(played, np.arange(150))
                self.assertGreaterEqual(engine.stream.chunk, 4)
                background = engine.calls[1:]
                # rtc leftover starts at the launch tick's own action; inference_delay is the last takeovers' max.
                for call in background:
                    self.assertEqual(call['leftover'][0, 0], call['tick'])
                    self.assertEqual(len(call['leftover']), 25)
                self.assertTrue(all(0 < call['delay'] <= 20 for call in background))
                info = engine.step_info
                self.assertEqual(set(info), {'chunk', 'chunk_pos', 'chunk_infer_ms'})

    def test_late_chunk_holds_the_tick_instead_of_running_dry(self):
        engine = StubEngine('blend', refill=2, latency_s=.08)
        engine.reset()
        start = time.monotonic()
        played = play(engine, 100)
        self.assertEqual(played, list(range(100)))
        self.assertGreater(time.monotonic() - start, .08)

    def test_reset_drops_the_previous_trials_chunk(self):
        engine = StubEngine('rtc', refill=25, latency_s=.1)
        engine.reset()
        play(engine, 26)
        self.assertIsNotNone(engine.worker)   # launched at tick 25, still running
        engine.reset()
        self.assertIsNone(engine.worker)
        self.assertEqual(play(engine, 3), [0., 1., 2.])
        self.assertIsNone(engine.calls[-1]['leftover'])   # fresh trial: plain inference, no guidance

    def test_background_failure_reaches_the_serving_thread(self):
        engine = StubEngine('rtc', refill=25, latency_s=0.)
        engine.reset()
        play(engine, 25)
        engine.fail = True
        with self.assertRaisesRegex(RuntimeError, 'stub inference failed'):
            for t in range(25, 60):
                engine.step(np.full(7, t, np.float32), {})
                time.sleep(.005)
        self.assertIsNone(engine.worker)


if __name__ == '__main__':
    unittest.main()
