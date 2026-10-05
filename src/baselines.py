"""Reproducible local baselines for CS5344 experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.coverage_generator import infer_feature_types


def full_normal_bootstrap(train: pd.DataFrame, label_column: str, target_n: int, seed: int) -> pd.DataFrame:
    """Sample normal training rows with replacement, preserving the full schema."""

    features = [column for column in train.columns if column != label_column]
    normal = train.loc[train[label_column] == 0, features]
    return normal.sample(n=target_n, replace=True, random_state=seed).reset_index(drop=True)


def starter_style_jitter(train: pd.DataFrame, label_column: str, target_n: int, seed: int) -> pd.DataFrame:
    """Match the supplied starter kit's 20-prototype bootstrap-and-jitter logic."""

    features = [column for column in train.columns if column != label_column]
    normal = train.loc[train[label_column] == 0, features].copy()
    prototypes = normal.sample(n=20, random_state=seed)
    synthetic = prototypes.sample(n=target_n, replace=True, random_state=seed).reset_index(drop=True)
    _, binary, interpolated_numeric = infer_feature_types(normal)
    integer_columns = [
        column
        for column in interpolated_numeric
        if np.all(np.isclose(normal[column].to_numpy(dtype=float), np.round(normal[column].to_numpy(dtype=float))))
    ]
    rng = np.random.default_rng(seed)
    for column in interpolated_numeric:
        std = normal[column].std(ddof=0)
        if not np.isfinite(std) or std == 0:
            continue
        values = synthetic[column].to_numpy(dtype=float) + rng.normal(0, 0.5 * std, target_n)
        values = np.clip(values, normal[column].min(), normal[column].max())
        if column in integer_columns:
            values = np.rint(values).astype(normal[column].dtype)
        synthetic[column] = values
    return synthetic[features]
