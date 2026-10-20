#!/usr/bin/env python3
"""Recompute every trial count, interval and test reported for this benchmark from the logged results.

Inputs (all in this repository): results/sessions/*.jsonl, the evaluation runner's own per-trial logs,
copied unchanged; results/adjudication.json, which rows are scored (invalid runs, smoke tests and
separately reported checks, as decided and recorded on the day); results/datasets/*, the dataset
metadata and conversion reports. Standard library only.

    python3 analysis/reproduce_counts.py           # tables
    python3 analysis/reproduce_counts.py --json    # the same numbers as JSON (tests compare these)
"""
import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stats import fisher, wilson  # noqa: E402

RESULTS = HERE.parent / 'results'
T1 = 't1_eval_20261016'
T2 = 't2_eval_20261017'
PI05_SYNC = 't2_eval_20261019'
PI05_RTC = 't2_eval_pi05rtc_20261019'
PI05_GEN = 't2_gen_pi05_20261019'
NAMES = {'act': 'ACT', 'smolvla': 'SmolVLA', 'dp': 'Diffusion Policy', 'act_t2': 'ACT', 'smolvla_t2': 'SmolVLA',
         'pi05_t2': 'pi0.5'}
COLOURS = ('red', 'green', 'yellow', 'purple')


def load(results=RESULTS):
    """{session: [row]} of logged rows, with 'scored' / 'excluded' / 'separate' flags and fixes applied."""
    rules = json.loads((results / 'adjudication.json').read_text())
    invalid = {(r['session'], r['trial_dir']) for r in rules['invalid_runs']}
    fixes = {(r['session'], r['trial_dir']): r for r in rules['category_fixes']}
    sessions = {}
    for path in sorted((results / 'sessions').glob('*.jsonl')):
        session = path.stem
        rows = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (session, row['trial_dir'])
            if key in fixes:
                assert row['category'] == fixes[key]['from'], key
                row['category'] = fixes[key]['to']
            separate = row['split'] in rules['separately_reported_splits'].get(session, [])
            row['excluded'] = key in invalid
            row['separate'] = separate or row['policy_tag'].endswith('_c2far')
            row['scored'] = (row['policy_tag'] in rules['scored_tags'][session]
                             and not row['excluded'] and not row['separate'])
            rows.append(row)
        sessions[session] = rows
    return sessions


def scored(rows, tag=None, **match):
    out = [r for r in rows if r['scored'] and (tag is None or r['policy_tag'] == tag)
           and all(r.get(k) == v for k, v in match.items())]
    keys = [(r['policy_tag'], r['trial']) for r in out]
    assert len(keys) == len(set(keys)), 'a scored trial appears twice'
    return out


def rate(rows):
    k, n = sum(r['success'] for r in rows), len(rows)
    lo, hi = wilson(k, n)
    return {'k': k, 'n': n, 'wilson95': [round(lo, 2), round(hi, 2)]}


def failures(rows):
    return dict(sorted(Counter(r['category'] for r in rows if not r['success']).items()))


def median_of_medians(rows, field):
    values = [r['latency'][field]['median'] for r in rows if r.get('latency', {}).get(field)]
    return round(statistics.median(values), 1)


def fisher_row(a, b):
    two, greater = fisher(a['k'], a['n'], b['k'], b['n'])
    return {'two_sided': float(f'{two:.2g}'), 'one_sided': float(f'{greater:.2g}')}


def compute(results=RESULTS):
    s = load(results)
    out = {}

    # Demonstrations: every accepted episode passed the alignment rule and went into the dataset.
    demos = {}
    for name in ('rebot_t1_v1', 'rebot_t2_v1'):
        info = json.loads((results / 'datasets' / f'{name}_info.json').read_text())
        report = json.loads((results / 'datasets' / f'{name}_conversion_report.json').read_text())
        converted = [r for r in report if 'skipped' not in r]
        demos[name] = {'episodes': info['total_episodes'], 'frames': info['total_frames'], 'fps': info['fps'],
                       'accepted': len(converted), 'passed_alignment': sum(bool(r.get('ok')) for r in converted),
                       'rejected_attempts': len(report) - len(converted),
                       'max_stale_fraction': max(r['stale_fraction'] for r in converted),
                       'max_frame_age_s': round(max(c['max_age_s'] for r in converted for c in r['cameras'].values()), 3)}
    demos['total_episodes'] = sum(d['episodes'] for d in demos.values())
    out['demonstrations'] = demos

    # Task 1: 15 in-distribution + 5 out-of-distribution trials per policy.
    t1 = {}
    for tag in ('act', 'smolvla', 'dp'):
        rows = scored(s[T1], tag)
        ood = [r for r in rows if r['split'] == 'out_of_distribution']
        t1[tag] = {'in_distribution': rate([r for r in rows if r['split'] == 'in_distribution']),
                   'out_of_distribution': rate(ood), 'all': rate(rows),
                   'ood_shifted_positions': rate([r for r in ood if r['trial'] in (15, 16, 17)]),
                   'ood_unseen_yaw': rate([r for r in ood if r['trial'] == 18]),
                   'ood_duck': rate([r for r in ood if r['trial'] == 19]),
                   'failures': failures(rows),
                   'infer_ms_median': median_of_medians(rows, 'infer_ms'),
                   'obs_to_action_ms_median': median_of_medians(rows, 'obs_to_action_ms')}
    c2far = [r for r in s[T1] if r['policy_tag'].endswith('_c2far')]
    t1['extra_c2_far'] = {r['policy_tag'][:-6]: r['category'] for r in c2far}
    out['task1'] = t1

    # Task 2, fixed layout: ACT and SmolVLA (same evening), pi0.5 sync and rtc (own sessions).
    t2 = {}
    for session, tag, key in ((T2, 'act_t2', 'act'), (T2, 'smolvla_t2', 'smolvla'), (PI05_SYNC, 'pi05_t2', 'pi05_sync'),
                              (PI05_RTC, 'pi05_t2', 'pi05_rtc')):
        rows = scored(s[session], tag)
        wrong = [r for r in rows if r['category'] == 'wrong_square']
        t2[key] = {**rate(rows), 'by_colour': {c: rate([r for r in rows if r['colour'] == c]) for c in COLOURS},
                   'failures': failures(rows), 'wrong_square': len(wrong),
                   'wrong_square_on_green': sum(r['note'].strip() == 'green' for r in wrong),
                   'success_colours': dict(Counter(r['colour'] for r in rows if r['success'])),
                   'failed_starts': sorted(r['pos'] for r in rows if not r['success']),
                   'infer_ms_median': median_of_medians(rows, 'infer_ms'),
                   'obs_to_action_ms_median': median_of_medians(rows, 'obs_to_action_ms')}
    act, smol, sync, rtc = (t2[k] for k in ('act', 'smolvla', 'pi05_sync', 'pi05_rtc'))
    # ACT trials where it released the cube on some square (success or wrong_square), and how many of those were green.
    t2['act']['placed'] = act['k'] + act['wrong_square']
    t2['act']['placed_on_green'] = act['success_colours'].get('green', 0) + act['wrong_square_on_green']
    t2['smolvla']['grasp_related_failures'] = smol['failures'].get('grasp_miss', 0) + smol['failures'].get('drop', 0)
    t2['fisher'] = {'smolvla_vs_act': fisher_row(smol, act), 'pi05_sync_vs_smolvla': fisher_row(sync, smol),
                    'pi05_sync_vs_act': fisher_row(sync, act), 'pi05_sync_vs_rtc': fisher_row(sync, rtc),
                    'pi05_rtc_vs_smolvla': fisher_row(rtc, smol)}
    out['task2'] = t2

    # Task 2 generalisation, pi0.5 sync: 36 scored trials in four dimensions, 6 exploratory reported apart.
    gen_rows = scored(s[PI05_GEN], 'pi05_t2')
    gen = {d: rate([r for r in gen_rows if r['dimension'] == d]) for d in ('control', 'spatial', 'language', 'clutter')}
    gen['formal'] = rate(gen_rows)
    explore = [r for r in s[PI05_GEN] if r['separate'] and not r['excluded']]
    gen['exploratory'] = {d: {**rate([r for r in explore if r['dimension'] == d]),
                              'categories': dict(Counter(r['category'] for r in explore if r['dimension'] == d))}
                          for d in ('exploratory_chinese', 'exploratory_object')}
    gen['spatial_failures'] = failures([r for r in gen_rows if r['dimension'] == 'spatial'])
    gen['language_wrong_colour'] = sum(r['category'] == 'wrong_square' for r in gen_rows if r['dimension'] == 'language')
    gen['fisher_control_vs_pi05_sync'] = fisher_row(gen['control'], sync)
    out['generalisation'] = gen

    # pi0.5 that night (sync, rtc, generalisation controls): A2 against the other training starts.
    night = scored(s[PI05_SYNC], 'pi05_t2') + scored(s[PI05_RTC], 'pi05_t2') + [r for r in gen_rows if r['dimension'] == 'control']
    training_starts = ('A2', 'B2', 'C2', 'B12', 'B23')
    out['pi05_night'] = {'A2': rate([r for r in night if r['pos'] == 'A2']),
                         'other_training_starts': rate([r for r in night if r['pos'] in training_starts and r['pos'] != 'A2'])}

    # Totals.
    scored_all = [r for rows in s.values() for r in rows if r['scored']]
    separate = [r for rows in s.values() for r in rows if r['separate'] and not r['excluded']]
    out['totals'] = {'scored_trials': len(scored_all), 'separately_reported': len(separate),
                     'scored_by_session': {k: len([r for r in v if r['scored']]) for k, v in s.items()},
                     'invalid_reruns': sum(r['excluded'] for rows in s.values() for r in rows),
                     'smoke_tests': sum(r['policy_tag'].endswith('_smoke') for rows in s.values() for r in rows)}
    return out


def fmt(r):
    return f"{r['k']}/{r['n']}  ({r['wilson95'][0]:.2f}-{r['wilson95'][1]:.2f})"


def report(out):
    d = out['demonstrations']
    print('Demonstrations')
    for name in ('rebot_t1_v1', 'rebot_t2_v1'):
        x = d[name]
        print(f"  {name}: {x['episodes']} episodes, {x['frames']} frames at {x['fps']} Hz; {x['passed_alignment']}/{x['accepted']} "
              f"accepted episodes pass the alignment rule (worst stale fraction {x['max_stale_fraction']:.3f}, "
              f"oldest frame {x['max_frame_age_s']} s); {x['rejected_attempts']} attempts rejected at collection")
    print(f"  total {d['total_episodes']}")
    print('\nTask 1 (pick and place), n/N (95% Wilson)')
    print(f"  {'policy':18}{'in-distribution':24}{'out-of-distribution':24}failures")
    for tag in ('smolvla', 'dp', 'act'):
        x = out['task1'][tag]
        print(f"  {NAMES[tag]:18}{fmt(x['in_distribution']):24}{fmt(x['out_of_distribution']):24}{x['failures']}")
    print(f"  extra position C2 +5 cm (not scored): {out['task1']['extra_c2_far']}")
    print('\nTask 2 (language-conditioned placement)')
    for key, label in (('pi05_sync', 'pi0.5 sync'), ('pi05_rtc', 'pi0.5 rtc'), ('smolvla', 'SmolVLA'), ('act', 'ACT')):
        x = out['task2'][key]
        colours = ' '.join(f"{c[0].upper()}{x['by_colour'][c]['k']}/{x['by_colour'][c]['n']}" for c in COLOURS)
        print(f"  {label:12}{fmt(x):22}{colours:24}{x['failures']}")
    for name, f in out['task2']['fisher'].items():
        print(f"  Fisher {name}: two-sided p = {f['two_sided']}, one-sided p = {f['one_sided']}")
    print('\nTask 2 generalisation, pi0.5 sync')
    g = out['generalisation']
    for dim in ('control', 'spatial', 'language', 'clutter', 'formal'):
        print(f'  {dim:12}{fmt(g[dim])}')
    for dim, x in g['exploratory'].items():
        print(f"  {dim:22}{x['k']}/{x['n']}  {x['categories']}  (reported separately)")
    print(f"  control vs fixed-layout sync: Fisher two-sided p = {g['fisher_control_vs_pi05_sync']['two_sided']}")
    n = out['pi05_night']
    print(f"  pi0.5 that night: A2 {n['A2']['k']}/{n['A2']['n']}, other training starts "
          f"{n['other_training_starts']['k']}/{n['other_training_starts']['n']}")
    t = out['totals']
    print(f"\nScored closed-loop trials: {t['scored_trials']} {t['scored_by_session']}; "
          f"reported separately: {t['separately_reported']}; invalid and rerun: {t['invalid_reruns']}; "
          f"smoke tests: {t['smoke_tests']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    out = compute()
    if args.json:
        print(json.dumps(out, indent=1))
    else:
        report(out)


if __name__ == '__main__':
    main()
