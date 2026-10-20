"""Run the next closed-loop evaluation trial of a session: pick (trial, policy) from
<session>/evals/results.jsonl, print what to set up, then hand over to evaluate.py.

Order: --order blocked (default) runs every trial of the first policy, then every trial of
the second, so the operator sees one policy at a time; --order alternate interleaves them
per trial index (even = first policy first, odd = second first; the T1 order). A trial
counts as done once evaluate.py has written its result line; an aborted start writes none,
so rerunning this command repeats it. Use --policy / --trial only to redo a specific run
(e.g. an invalid one).

Running this ENABLES AND MOVES THE ARM through evaluate.py. Operator at the arm.

    python3 -B rebot_bench/eval_next.py --session <session> --tactile-status <W2L status.json>
    python3 -B rebot_bench/eval_next.py ... --dry-run  # only show the next run
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from rebot_bench.rig import REBOT_PYTHON  # noqa: E402

# tag -> (display name, policy-server socket; None = evaluate.py default)
POLICIES = {'act_t2': ('ACT', None), 'smolvla_t2': ('SmolVLA', '/tmp/rebot_il_policy_smolvla.sock'),
            'dp': ('Diffusion', '/tmp/rebot_il_policy_dp.sock'),
            'pi05_t2': ('pi0.5', '/tmp/rebot_il_policy_pi05.sock'),
            'pi05_t3': ('pi0.5', '/tmp/rebot_il_policy_pi05.sock')}
ROWS = {'A': '近排（靠机械臂）', 'B': '中排（十字那排）', 'C': '远排'}
COLS = {'1': '左列', '2': '中列', '3': '右列'}
SQUARES = {'red': '红框：左侧、与十字同排', 'green': '绿框：右侧、与十字同排',
           'yellow': '黄框：左侧、远排', 'purple': '紫框：右侧、远排'}


def describe_start(pos):
    """'C2' -> 远排·中列; 'B12' -> 中排·左列与中列之间; other labels (T1 OOD) as given."""
    row, cols = pos[0], pos[1:]
    if row not in ROWS or not cols or len(cols) > 2 or any(c not in COLS for c in cols):
        return pos
    if len(cols) == 1:
        return f'{ROWS[row]} · {COLS[cols]}'
    return f'{ROWS[row]} · {COLS[cols[0]]}与{COLS[cols[1]]}之间'


def run_order(n_trials, tags, order='blocked'):
    if order == 'blocked' or len(tags) == 1:
        return [(t, tag) for tag in tags for t in range(n_trials)]
    first, second = tags
    return [(t, tag) for t in range(n_trials) for tag in ((first, second) if t % 2 == 0 else (second, first))]


def done_runs(results_path):
    done = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row['trial'], row['policy_tag']))
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'protocols/t2_protocol.json')
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--tactile-status', type=Path, required=True, help='W2L dual preview status.json')
    parser.add_argument('--policies', default='act_t2,smolvla_t2', help='first,second')
    parser.add_argument('--order', choices=('blocked', 'alternate'), default='blocked')
    parser.add_argument('--policy', help='redo: policy tag')
    parser.add_argument('--trial', type=int, help='redo: trial index')
    parser.add_argument('--dry-run', action='store_true', help='show the next run and exit')
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text())
    evaluation = protocol['evaluation']
    trials = evaluation.get('trials') or evaluation['in_distribution'] + evaluation.get('out_of_distribution', [])
    tags = args.policies.split(',')
    order = run_order(len(trials), tags, args.order)
    done = done_runs(args.session / 'evals' / 'results.jsonl')
    if (args.policy is None) != (args.trial is None):
        raise SystemExit('--policy and --trial go together')
    if args.policy:
        trial, tag = args.trial, args.policy
    else:
        pending = [run for run in order if run not in done]
        if not pending:
            print(f'全部 {len(order)} 次已完成：{args.session}/evals/results.jsonl')
            return 0
        trial, tag = pending[0]
    row = trials[trial]
    name, socket = POLICIES[tag]
    finished = len(done & set(order))
    remaining = len(order) - finished
    policy_done = len({t for t, p in done if p == tag and t < len(trials)})
    xy = row.get('xy_cm') or protocol.get('starts', {}).get('xy_cm', {}).get(row['pos'])
    colour = row.get('colour')

    bar = '=' * 64
    print(f'\n{bar}')
    print(f'  策略：{name}    第 {trial} 次（共 {len(trials)} 次，{name} 已完成 {policy_done} 次）'
          + ('  [重跑]' if args.policy else ''))
    if args.policy:
        print(f'  总进度：已完成 {finished}/{len(order)} 个运行，未跑的还剩 {remaining} 个（本次是重跑，不在其中）')
    else:
        print(f'  总进度：已完成 {finished}/{len(order)} 个运行，还剩 {remaining} 个（含本次）')
    if row.get('layer'):  # T3: fixed tower, the row only names the layer
        print(f"  目标：第 {row['layer']} 层（从下数）中间那块" + ('；开始前整塔重搭' if row.get('rebuild_before') else ''))
    else:
        print(f"  蓝块位置：{row['pos']}（{describe_start(row['pos'])}）"
              + (f'，桌垫坐标 {xy} cm' if xy else '') + f"，朝向 {row['yaw_deg']}°")
    if colour:
        print(f'  目标：{SQUARES.get(colour, colour)}')
    if row.get('dimension'):
        print(f"  维度：{row['dimension']}（{row.get('split', 'in_distribution')}）")
    if row.get('note'):
        print(f"  说明：{row['note']}")
    print(f"  指令：{row.get('task', protocol['task'])}")
    print(bar)
    command = [str(REBOT_PYTHON), '-u', '-B', str(HERE / 'evaluate.py'), '--protocol', str(args.protocol),
               '--session', str(args.session), '--policy-tag', tag, '--trial', str(trial),
               '--tactile-status', str(args.tactile_status), '--obs-video']
    if socket:
        command += ['--socket', socket]
    if args.dry_run:
        print(' '.join(command))
        return 0
    sys.stdout.flush()
    os.execv(command[0], command)


if __name__ == '__main__':
    sys.exit(main())
