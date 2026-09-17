"""
JSON serialisation of pipeline results for the Next.js visualiser.

The visualiser reads static JSON from `results/`. This module defines the schema
and writes it, converting numpy types to plain Python so `json.dump` succeeds.

Schema (one file per dataset, e.g. results/synthetic.json):

    {
      "dataset_group": "synthetic",
      "generated_utc": "2026-07-09T21:00:00Z",
      "seed": 42,
      "subsets": {
        "rings": {
          "label": "Concentric rings",
          "k_true": 2,
          "points": [[x, y], ...],          # 2-D coords for the scatter plot
          "ground_truth": [0, 1, ...],
          "sigma": 0.83,                     # RBF bandwidth actually used
          "selection": { ... },              # elbow + eigengap diagnostics
          "algorithms": {
            "kmeans": {
              "label": "K-Means (baseline)",
              "k_used": 2,
              "labels": [0, 1, ...],         # per-point predicted cluster
              "metrics": {"ARI": 0.0, "NMI": 0.0}
            }, ...
          }
        }, ...
      }
    }
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _to_native(obj):
    """Recursively convert numpy scalars/arrays to JSON-serialisable Python."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def export_results(results: dict, path: str | Path) -> Path:
    """Write `results` to `path` as pretty-printed JSON, returning the Path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_native(results)
    payload.setdefault("generated_utc", datetime.now(timezone.utc).isoformat())
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path
