"""Where trained models live: `<MODELS_DIR>/<name>.joblib` plus `<name>.json`, a manifest
that says what the model was trained on (the start of model lifecycle control, per the
architecture's GMP table; no retraining happens without a new version)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib


def paths(models_dir: str, name: str) -> tuple[Path, Path]:
    root = Path(models_dir)
    return root / f"{name}.joblib", root / f"{name}.json"


def save(models_dir: str, name: str, model: Any, manifest: dict[str, Any]) -> None:
    model_path, manifest_path = paths(models_dir, name)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = model_path.with_suffix(".tmp")
    joblib.dump(model, tmp)
    tmp.replace(model_path)  # never leave a half-written model for a service to load
    manifest = {**manifest, "created": datetime.now(UTC).isoformat()}
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))


def manifest(models_dir: str, name: str) -> dict[str, Any] | None:
    _, manifest_path = paths(models_dir, name)
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text())


def load(models_dir: str, name: str) -> Any | None:
    model_path, _ = paths(models_dir, name)
    return joblib.load(model_path) if model_path.exists() else None
