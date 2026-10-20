#!/usr/bin/env python3
"""Export the compact per-command tables that analysis/reproduce_chunks.py reads (run once by the author).

For every π0.5 trial of the Task 2 sync and rtc sessions it writes one CSV row per 50 Hz follow command
that carries a policy target: time since follow start, the J7 (gripper) target, the policy state and
action number, the chunk the action came from (rtc logs it; sync does not) and the policy inference
time, plus the nominal gripper-tip height (cm) of the requested, sent and measured joint positions.

The tip heights come from the rig's nominal kinematics (`return_planner.ArmModel`: the arm's MJCF with
a 26 cm tool on raw joint angles). That model and the raw traces stay in the private rig checkout, so
this script cannot run from a clone of this repository; it is here so the export is inspectable. The
absolute heights are not calibrated; only differences between consecutive commands are used.

    REBOT_RIG_ROOT=<rig checkout> python3 tools/export_chunk_traces.py \\
        --session sync=<rig>/datasets/il_vla_benchmark/t2_eval_20261019 \\
        --session rtc=<rig>/datasets/il_vla_benchmark/t2_eval_pi05rtc_20261019 --out results/chunk_handover
"""
import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rebot_bench.rig import import_rig  # noqa: E402

import_rig()
from hardware.end_effectors.return_planner import ArmModel  # noqa: E402

COLUMNS = ['trial', 't_s', 'j7_requested_rad', 'policy_state', 'action_seq', 'chunk', 'infer_ms',
           'tip_requested_cm', 'tip_sent_cm', 'tip_actual_cm']


def export(model, session, out):
    def tip(raw):
        return round(model.tool({j: raw[str(j)] for j in range(1, 7)})[0][2] * 100, 4)

    rows = []
    for trial_dir in sorted((session / 'evals/pi05_t2').glob('trial*')):
        trial = int(trial_dir.name[5:7])
        chunks = {}
        for line in (trial_dir / 'policy_server_log.jsonl').read_text().splitlines():
            entry = json.loads(line)
            if 'seq' in entry and 'chunk' in entry:
                chunks[entry['seq']] = entry['chunk']
        trace = json.loads((trial_dir / 'mirror_trace.json').read_text())
        follow = [c for c in trace['commands'] if c['phase'] == 'follow' and c.get('requested_raw_rad')]
        t0 = follow[0]['monotonic']
        for c in follow:
            policy = c.get('policy') or {}
            seq = policy.get('action_seq')
            rows.append([trial, round(c['monotonic'] - t0, 4), c['requested_raw_rad']['7'], policy.get('state', ''),
                         '' if seq is None else seq, chunks.get(seq, ''),
                         '' if policy.get('infer_ms') is None else policy['infer_ms'],
                         tip(c['requested_raw_rad']), tip(c['sent_raw_rad']), tip(c['actual_raw_rad'])])
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, 'wt', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    print(f'{out}: {len(rows)} rows')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--session', action='append', required=True, help='name=path of an evaluation session')
    parser.add_argument('--out', type=Path, default=ROOT / 'results/chunk_handover')
    args = parser.parse_args()
    model = ArmModel()
    for item in args.session:
        name, _, path = item.partition('=')
        export(model, Path(path), args.out / f'{name}.csv.gz')


if __name__ == '__main__':
    main()
