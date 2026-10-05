#!/usr/bin/env python3
"""Create a schema-checked normal-only CSV for one CS5344 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

# Support both `python scripts/generate_submission.py` and `python -m ...`.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.baselines import full_normal_bootstrap
from src.coverage_generator import GenerationConfig, generate_normal_submission, validate_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path, help="Released labeled training CSV.")
    parser.add_argument("--output", required=True, type=Path, help="Output submission CSV.")
    parser.add_argument("--target-n", required=True, type=int, help="Exact official Kaggle submission size.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--method",
        choices=("coverage", "full-bootstrap"),
        default="coverage",
        help="Generation method. Use local evaluation results to choose it.",
    )
    parser.add_argument("--coverage-strength", type=float, default=0.15)
    parser.add_argument("--max-oversample-factor", type=float, default=4.0)
    parser.add_argument("--bootstrap-fraction", type=float, default=0.50)
    parser.add_argument("--interpolation-alpha", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train)
    config = GenerationConfig(
        target_n=args.target_n,
        seed=args.seed,
        coverage_strength=args.coverage_strength,
        max_oversample_factor=args.max_oversample_factor,
        bootstrap_fraction=args.bootstrap_fraction,
        interpolation_alpha=args.interpolation_alpha,
    )
    if args.method == "coverage":
        submission, diagnostics = generate_normal_submission(train, "is_anomaly", config)
    else:
        features = [column for column in train.columns if column != "is_anomaly"]
        synthetic = full_normal_bootstrap(train, "is_anomaly", args.target_n, args.seed)
        submission = synthetic.copy()
        submission.insert(0, "id", range(args.target_n))
        validate_submission(submission, features, args.target_n)
        diagnostics = {
            "normal_rows": int((train["is_anomaly"] == 0).sum()),
            "feature_count": len(features),
            "exact_duplicate_rows": int(synthetic.duplicated().sum()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)

    print(f"Method: {args.method}")
    print(f"Wrote {args.output} with shape {submission.shape}.")
    for name, value in diagnostics.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
