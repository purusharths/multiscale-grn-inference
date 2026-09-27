#!/usr/bin/env bash
# Vendors antonioterpin/jkonet-star into external/jkonet-star with its own venv.
#
# external/ is gitignored (see README.md in this folder for why: jkonet-star
# pins its own jax/torch/wandb stack that would conflict with this project's
# pyproject.toml), so every clone of this repo needs to run this once before
# compare_jkonet_star.py will work.
#
# Usage:
#   bash tests/diagnostics/jko-testing/setup_jkonet_star.sh
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
TARGET="$REPO_ROOT/external/jkonet-star"
PINNED_COMMIT="1741c53ae00da932e0841ce02eca2d842c04b813"  # 2025-03-18, main @ setup time

if [ -d "$TARGET" ]; then
    echo "already present: $TARGET (delete it to re-clone)"
else
    git clone https://github.com/antonioterpin/jkonet-star.git "$TARGET"
    git -C "$TARGET" checkout "$PINNED_COMMIT"
fi

cd "$TARGET"

# requirements.txt in this repo is UTF-16 with CRLF line endings; normalize
# before pip can read it, and drop the sphinx/doc-build deps we don't need.
iconv -f UTF-16 -t UTF-8 requirements.txt | tr -d '\r' | grep -v '^#' | grep -v '^sphinx' > requirements-core.txt

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-core.txt

echo "done: $TARGET/.venv"
