#!/usr/bin/env bash
# Runs once after the dev container is created.
set -euo pipefail

# The named volume mounted at ~/.claude can come back owned by root (Docker
# creates a new named volume as root unless the image seeds the mount point).
# Claude Code then fails with "Failed to retrieve auth status after login".
# Repair it before anything else, in case the volume predates the Dockerfile fix.
if [ -d "$HOME/.claude" ] && [ ! -w "$HOME/.claude" ]; then
  echo "==> Fixing ownership of $HOME/.claude"
  sudo chown -R "$(id -u):$(id -g)" "$HOME/.claude"
fi

echo "==> Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

if [ -f requirements-dev.txt ]; then
  echo "==> Installing dev requirements"
  python -m pip install -r requirements-dev.txt
fi

if [ -f requirements.txt ]; then
  echo "==> Installing runtime requirements"
  python -m pip install -r requirements.txt
fi

# Install the shared package in editable mode once it exists.
if [ -f pyproject.toml ]; then
  echo "==> Installing project (editable)"
  python -m pip install -e . || true
fi

if [ -f .pre-commit-config.yaml ]; then
  echo "==> Installing pre-commit hooks"
  pre-commit install || true
fi

echo "==> Done. Bring the stack up with: docker compose up -d"
