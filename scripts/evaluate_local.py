#!/usr/bin/env python3
"""Local proxy evaluation for CS5344 candidate normal-data generators.

This is not the private Kaggle evaluator. It uses a reproducible stratified holdout
from the released training data to rank methods before a limited public submission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from pyod.models.ecod import ECOD
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.baselines import full_normal_bootstrap, starter_style_jitter
from src.coverage_generator import GenerationConfig, generate_normal_submission, infer_feature_types
from src.coverage_v2 import V2Config, generate_normal_submission_v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--target-n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--iforest-seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def make_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    categorical, _, numeric = infer_feature_types(frame)
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("scale", RobustScaler())]), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
    )


def score_candidate(
    synthetic: pd.DataFrame, audit_x: pd.DataFrame, audit_y: pd.Series, iforest_seeds: list[int]
) -> dict[str, float]:
    preprocessor = make_preprocessor(synthetic)
    synthetic_x = preprocessor.fit_transform(synthetic)
    audit_matrix = preprocessor.transform(audit_x)

    ecod = ECOD()
    ecod.fit(synthetic_x)
    ecod_scores = ecod.decision_function(audit_matrix)

    ecod_auprc = float(average_precision_score(audit_y, ecod_scores))
    iforest_auprcs = []
    for seed in iforest_seeds:
        isolation_forest = IsolationForest(n_estimators=500, random_state=seed, n_jobs=-1)
        isolation_forest.fit(synthetic_x)
        if_scores = -isolation_forest.score_samples(audit_matrix)
        iforest_auprcs.append(float(average_precision_score(audit_y, if_scores)))
    if_auprc = float(np.mean(iforest_auprcs))
    return {
        "ecod_auprc": ecod_auprc,
        "iforest_auprc": if_auprc,
        "iforest_auprc_std": float(np.std(iforest_auprcs)),
        "mean_auprc": (ecod_auprc + if_auprc) / 2,
    }


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train)
    development, audit = train_test_split(
        train,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=train["is_anomaly"],
    )
    development = development.reset_index(drop=True)
    audit = audit.reset_index(drop=True)
    features = [column for column in train.columns if column != "is_anomaly"]

    methods = {
        "starter_20_prototype_jitter": starter_style_jitter(development, "is_anomaly", args.target_n, args.seed),
        "full_normal_bootstrap": full_normal_bootstrap(development, "is_anomaly", args.target_n, args.seed),
        "coverage_aware_v1": generate_normal_submission(
            development,
            "is_anomaly",
            GenerationConfig(target_n=args.target_n, seed=args.seed),
        )[0].drop(columns="id"),
        "coverage_aware_v2": generate_normal_submission_v2(
            development,
            "is_anomaly",
            V2Config(target_n=args.target_n, seed=args.seed),
        )[0].drop(columns="id"),
    }
    results = {
        name: score_candidate(synthetic[features], audit[features], audit["is_anomaly"], args.iforest_seeds)
        for name, synthetic in methods.items()
    }
    report = {
        "warning": "Local holdout proxy only; it is not the official Kaggle score.",
        "train": str(args.train),
        "target_n": args.target_n,
        "seed": args.seed,
        "iforest_seeds": args.iforest_seeds,
        "development_rows": len(development),
        "audit_rows": len(audit),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
