#!/usr/bin/env python3
"""Upload the Task 1 / Task 2 LeRobot v3 datasets to the Hugging Face Hub with their data cards.

Dry run by default: prints what would be uploaded (files, sizes, target repo) and changes nothing.
--push creates the dataset repository (public unless --private) and uploads the dataset folder plus
dataset/cards/<name>.md as README.md. Needs huggingface_hub and a token (`hf auth login`).

    python3 dataset/upload_to_hub.py --dataset-root <dir with rebot_t1_v1 and rebot_t2_v1>
    python3 dataset/upload_to_hub.py --dataset-root <...> --only rebot_t2_v1 --push
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = {
    'rebot_t1_v1': 'bowenszhu/rebot-b601-t1-pick-place',
    'rebot_t2_v1': 'bowenszhu/rebot-b601-t2-language-squares',
}
# Rig-local or regenerable files that do not belong in the published dataset.
IGNORE = ['*_report.json', '*.log', '.cache/**', '**/.DS_Store']


def plan(root, name):
    folder = root / name
    info = json.loads((folder / 'meta/info.json').read_text())
    files = sorted(p for p in folder.rglob('*') if p.is_file())
    size = sum(p.stat().st_size for p in files)
    card = HERE / 'cards' / f'{name}.md'
    if not card.exists():
        raise SystemExit(f'missing data card {card}')
    return folder, card, info, files, size


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--only', choices=sorted(DATASETS), action='append')
    parser.add_argument('--repo-prefix', help='override the owner, e.g. another user or organisation')
    parser.add_argument('--private', action='store_true')
    parser.add_argument('--push', action='store_true', help='actually create the repositories and upload')
    args = parser.parse_args()

    for name in args.only or sorted(DATASETS):
        repo_id = DATASETS[name] if not args.repo_prefix else f"{args.repo_prefix}/{DATASETS[name].split('/')[1]}"
        folder, card, info, files, size = plan(args.dataset_root, name)
        print(f"{name} -> https://huggingface.co/datasets/{repo_id}  ({'private' if args.private else 'public'})")
        print(f"  {info['total_episodes']} episodes, {info['total_frames']} frames, {len(files)} files, {size / 1e6:.0f} MB; card {card.name}")
        for path in files:
            print(f'    {path.relative_to(folder)}  {path.stat().st_size / 1e6:.1f} MB')
        if not args.push:
            continue
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id, repo_type='dataset', private=args.private, exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type='dataset', folder_path=str(folder), ignore_patterns=IGNORE,
                          commit_message=f'Add {name} (LeRobot v3)')
        api.upload_file(repo_id=repo_id, repo_type='dataset', path_or_fileobj=str(card), path_in_repo='README.md',
                        commit_message='Add data card')
        # LeRobot loads a dataset by its codebase version tag.
        api.create_tag(repo_id, repo_type='dataset', tag=info['codebase_version'], exist_ok=True)
        print(f'  uploaded and tagged {info["codebase_version"]}')
    if not args.push:
        print('\ndry run: nothing uploaded (add --push)')


if __name__ == '__main__':
    main()
