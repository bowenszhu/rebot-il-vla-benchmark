#!/usr/bin/env python3
"""Recompute the π0.5 chunk-handover numbers (Task 2, sync against rtc) from results/chunk_handover/*.csv.gz.

Each CSV row is one 50 Hz follow command with a policy target (tools/export_chunk_traces.py). For every
new policy action the commanded gripper-tip height is compared with the previous action's: a step where
a new chunk takes over is a handover jump, any other step is a within-chunk step. rtc logs the chunk of
each action; sync does not, so there a new chunk is an action whose inference took more than 150 ms.
Release (successful trials): the J7 target first opens wide (> 5.0 rad), then stays in the holding band
(3.0-4.4 rad) for at least 1 s; release is the first tick after the tightest grip where the target rises
0.8 rad above the band's median. The drop is how far the sent (or measured) tip height falls within 2.5 s.

Heights come from nominal kinematics without calibration: only differences are meaningful. Needs numpy.

    python3 analysis/reproduce_chunks.py [--json]
"""
import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from reproduce_counts import PI05_RTC, PI05_SYNC, load, scored  # noqa: E402

DATA = HERE.parent / 'results' / 'chunk_handover'
WINDOW_S = 2.5


def read(path):
    trials = {}
    with gzip.open(path, 'rt', newline='') as handle:
        for row in csv.DictReader(handle):
            trials.setdefault(int(row['trial']), []).append(row)
    return trials


def release_index(j7):
    opened = next(i for i in range(len(j7)) if j7[i] > 5.0)
    run, start = 0, None
    for i in range(opened, len(j7)):
        run = run + 1 if 3.0 < j7[i] < 4.4 else 0
        if run >= 50:
            start = i - 49
            break
    end = start
    while end + 1 < len(j7) and j7[end + 1] < 4.4:
        end += 1
    hold = float(np.median(j7[start:end + 1]))
    lo = start + int(np.argmin(j7[start:end + 1]))
    return next(i for i in range(lo, len(j7)) if j7[i] > hold + 0.8)


def analyse(path, success):
    take, within, trials = [], [], []
    for trial, rows in sorted(read(path).items()):
        acts, seen = [], set()
        for r in rows:
            if r['policy_state'] != 'live' or r['action_seq'] in seen:
                continue
            seen.add(r['action_seq'])
            acts.append((float(r['t_s']), float(r['tip_requested_cm']), r['chunk'],
                         float(r['infer_ms'] or 0)))
        z = np.array([a[1] for a in acts])
        if any(a[2] for a in acts):
            new = np.array([acts[i + 1][2] != acts[i][2] for i in range(len(acts) - 1)])
        else:
            new = np.array([acts[i + 1][3] > 150 for i in range(len(acts) - 1)])
        dz = np.diff(z)
        take += list(np.abs(dz[new]))
        within += list(np.abs(dz[~new]))
        tc = np.array([float(r['t_s']) for r in rows])
        row = {'trial': trial, 'success': success[trial], 'follow_s': round(float(tc[-1]), 1), 'takeovers': int(new.sum())}
        if success[trial]:
            j7 = np.array([float(r['j7_requested_rad']) for r in rows])
            zs = np.array([float(r['tip_sent_cm']) for r in rows])
            za = np.array([float(r['tip_actual_cm']) for r in rows])
            rel = release_index(j7)
            w = (tc >= tc[rel]) & (tc <= tc[rel] + WINDOW_S)
            row.update(release_s=round(float(tc[rel]), 1), sent_drop_cm=round(float(zs[rel] - zs[w].min()), 2),
                       act_drop_cm=round(float(za[rel] - za[w].min()), 2))
        trials.append(row)
    take, within = np.array(take), np.array(within)
    rel = [r for r in trials if 'sent_drop_cm' in r]
    drops = np.array([r['sent_drop_cm'] for r in rel])
    adrops = np.array([r['act_drop_cm'] for r in rel])
    return {
        'handover_jump_cm': {'median': round(float(np.median(take)), 2), 'p90': round(float(np.percentile(take, 90)), 2),
                             'max': round(float(take.max()), 2), 'n': int(take.size)},
        'within_chunk_step_cm': {'median': round(float(np.median(within)), 2),
                                 'p90': round(float(np.percentile(within, 90)), 2),
                                 'p99': round(float(np.percentile(within, 99)), 2), 'n': int(within.size)},
        'release': {'n': len(rel), 'sent_drop_median_cm': round(float(np.median(drops)), 2),
                    'sent_drop_max_cm': round(float(drops.max()), 2), 'sent_drop_ge_half_cm': int((drops >= 0.5).sum()),
                    'measured_drop_ge_half_cm': int((adrops >= 0.5).sum()),
                    'release_s_median': round(float(np.median([r['release_s'] for r in rel])), 1),
                    'release_s_range': [min(r['release_s'] for r in rel), max(r['release_s'] for r in rel)],
                    'follow_s_median_success': round(float(np.median([r['follow_s'] for r in trials if r['success']])), 1)},
    }


def compute():
    sessions = load()
    out = {}
    for name, session in (('sync', PI05_SYNC), ('rtc', PI05_RTC)):
        success = {r['trial']: r['success'] for r in scored(sessions[session], 'pi05_t2')}
        out[name] = analyse(DATA / f'{name}.csv.gz', success)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    out = compute()
    if args.json:
        print(json.dumps(out, indent=1))
        return
    print(f"{'':52}{'sync':>16}{'rtc':>16}")
    rows = [('handover jump, median / p90 / max (cm)', 'handover_jump_cm', ('median', 'p90', 'max')),
            ('within-chunk step, median / p90 / p99 (cm)', 'within_chunk_step_cm', ('median', 'p90', 'p99'))]
    for label, key, fields in rows:
        cells = ['/'.join(f"{out[m][key][f]:g}" for f in fields) for m in ('sync', 'rtc')]
        print(f'{label:52}{cells[0]:>16}{cells[1]:>16}  (n = {out["sync"][key]["n"]}, {out["rtc"][key]["n"]})')
    for label, field in (('sent target drops >= 0.5 cm after release', 'sent_drop_ge_half_cm'),
                         ('measured drops >= 0.5 cm after release', 'measured_drop_ge_half_cm'),
                         ('sent drop after release, median (cm)', 'sent_drop_median_cm'),
                         ('sent drop after release, max (cm)', 'sent_drop_max_cm'),
                         ('release time, median (s)', 'release_s_median'),
                         ('follow time of successful trials, median (s)', 'follow_s_median_success')):
        cells = [f"{out[m]['release'][field]}" + (f"/{out[m]['release']['n']}" if '_ge_' in field else '') for m in ('sync', 'rtc')]
        print(f'{label:52}{cells[0]:>16}{cells[1]:>16}')


if __name__ == '__main__':
    main()
