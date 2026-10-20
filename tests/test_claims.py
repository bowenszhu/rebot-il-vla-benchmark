"""Every published number (analysis/claims.json) is what the analysis recomputes from results/."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))
import reproduce_counts  # noqa: E402
from stats import fisher, wilson  # noqa: E402

CLAIMS = json.loads((ROOT / 'analysis/claims.json').read_text())


def lookup(data, path):
    for key in path.split('.'):
        data = data[key]
    return data


class ClaimsTest(unittest.TestCase):
    def check(self, computed, claims):
        for item in claims:
            with self.subTest(item['claim']):
                self.assertEqual(lookup(computed, item['path']), item['value'])

    def test_counts(self):
        self.check(reproduce_counts.compute(), CLAIMS['counts'])

    def test_chunks(self):
        try:
            import reproduce_chunks
        except ImportError as exc:  # numpy is the only dependency outside the standard library
            self.skipTest(f'needs numpy: {exc}')
        self.check(reproduce_chunks.compute(), CLAIMS['chunks'])

    def test_every_scored_trial_is_counted_once(self):
        sessions = reproduce_counts.load()
        scored = [(name, r['policy_tag'], r['trial']) for name, rows in sessions.items() for r in rows if r['scored']]
        self.assertEqual(len(scored), len(set(scored)))
        rules = json.loads((ROOT / 'results/adjudication.json').read_text())
        logged = {(name, r['trial_dir']) for name, rows in sessions.items() for r in rows}
        for run in rules['invalid_runs'] + rules['category_fixes']:
            self.assertIn((run['session'], run['trial_dir']), logged)


class StatsTest(unittest.TestCase):
    def test_wilson_reference_values(self):
        # The intervals stated in the validation records (0.39-0.78, 0.84-1.00), to four places.
        self.assertEqual([round(x, 4) for x in wilson(12, 20)], [0.3866, 0.7812])
        self.assertEqual([round(x, 4) for x in wilson(20, 20)], [0.8389, 1.0])
        self.assertEqual(wilson(0, 0), (0.0, 1.0))

    def test_fisher_reference_values(self):
        # scipy.stats.fisher_exact on the same tables.
        two, one = fisher(12, 20, 4, 20)
        self.assertAlmostEqual(two, 0.0224774273717544, places=12)
        self.assertAlmostEqual(one, 0.0112387136858772, places=12)
        self.assertAlmostEqual(fisher(20, 20, 12, 20)[0], 0.003276003276003276, places=12)
        self.assertAlmostEqual(fisher(20, 20, 4, 20)[1], 7.708572780875339e-08, places=18)


if __name__ == '__main__':
    unittest.main()
