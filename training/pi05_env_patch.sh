#!/usr/bin/env bash
# Turn a copy of the lerobot 0.4.4 env (lerobot-blackwell locally, the project env on a cluster) into a pi0.5 env.
# lerobot 0.4.4's pi05 needs the transformers fix/lerobot_openpi branch, which conflicts with the 4.57.6 that
# SmolVLA uses, so patch a clone and leave the original alone:
#   conda create -y -n lerobot-pi --clone lerobot-blackwell && conda run -n lerobot-pi bash pi05_env_patch.sh
#   cluster: conda create -y -p $P/env_pi --clone $P/env, activate it, then bash pi05_env_patch.sh
# Runs with whatever python is first on PATH. No GPU needed.
set -euo pipefail
# fix/lerobot_openpi head on 2026-10-18 (reports itself as 4.53.3); it needs tokenizers 0.21.
TRANSFORMERS_REV=dcddb970176382c0fcf4521b0c0e6fc15894dfe0
python -m pip install -q --no-deps "transformers @ git+https://github.com/huggingface/transformers.git@$TRANSFORMERS_REV"
python -m pip install -q "tokenizers>=0.21,<0.22" "scipy>=1.10.1,<1.15"
python -m pip check
python - <<'PY'
import importlib.metadata as m
from transformers.models.siglip import check
assert check.check_whether_transformers_replace_is_installed_correctly(), 'transformers replace check failed'
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: F401
print('PI05_ENV_OK transformers', m.version('transformers'), 'tokenizers', m.version('tokenizers'),
      'lerobot', m.version('lerobot'))
PY
