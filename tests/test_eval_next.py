"""eval_next.py ordering and labels (offline; nothing is executed)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rebot_bench import eval_next  # noqa: E402


class EvalNextTest(unittest.TestCase):
    def test_single_policy_runs_every_trial_once_in_order(self):
        for order in ('blocked', 'alternate'):
            self.assertEqual(eval_next.run_order(3, ['dp'], order), [(0, 'dp'), (1, 'dp'), (2, 'dp')])

    def test_two_policy_orders_are_unchanged(self):
        self.assertEqual(eval_next.run_order(2, ['a', 'b'], 'alternate'), [(0, 'a'), (0, 'b'), (1, 'b'), (1, 'a')])
        self.assertEqual(eval_next.run_order(2, ['a', 'b'], 'blocked'), [(0, 'a'), (1, 'a'), (0, 'b'), (1, 'b')])

    def test_t1_positions_including_free_form_ood_labels(self):
        self.assertEqual(eval_next.describe_start('B12'), '中排（十字那排） · 左列与中列之间')
        self.assertEqual(eval_next.describe_start('A2 -5 cm near'), 'A2 -5 cm near')
        self.assertIn('dp', eval_next.POLICIES)


if __name__ == '__main__':
    unittest.main()
