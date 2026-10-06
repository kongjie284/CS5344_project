"""Local-neighbour synthetic-normal generator used for coverage v2 experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.coverage_generator import (
    _capped_weights,
    _integer_columns,
    _largest_remainder,
    infer_feature_types,
    validate_submission,
)


@dataclass(frozen=True)
class V2Config:
    target_n: int
    seed: int = 42
    coverage_strength: float = 0.0
    max_oversample_factor: float = 2.0
    bootstrap_fraction: float = 0.20
    interpolation_alpha: float = 0.10
    donor_candidate_pool: int = 64
    log_skew_threshold: float = 2.0

    def __post_init__(self) -> None:
        if self.target_n <= 0:
            raise ValueError("target_n must be positive.")
        if not 0 <= self.coverage_strength <= 1:
            raise ValueError("coverage_strength must be in [0, 1].")
        if self.max_oversample_factor < 1:
            raise ValueError("max_oversample_factor must be at least 1.")
        if not 0 <= self.bootstrap_fraction <= 1:
            raise ValueError("bootstrap_fraction must be in [0, 1].")
        if not 0 <= self.interpolation_alpha <= 1:
            raise ValueError("interpolation_alpha must be in [0, 1].")
        if self.donor_candidate_pool < 2:
            raise ValueError("donor_candidate_pool must be at least 2.")


def _mixed_numeric_representation(normal: pd.DataFrame, numeric: list[str], log_columns: set[str]) -> np.ndarray:
    """Log-transform heavy tails then robust-scale numeric features for neighbour search."""

    values = normal[numeric].to_numpy(dtype=float).copy()
    for index, column in enumerate(numeric):
        if column in log_columns:
            values[:, index] = np.log1p(values[:, index])
    median = np.median(values, axis=0)
    q25, q75 = np.percentile(values, [25, 75], axis=0)
    scale = q75 - q25
    scale[scale <= 1e-12] = 1.0
    return (values - median) / scale


def _choose_local_donors(
    anchors: np.ndarray, group_indices: np.ndarray, representation: np.ndarray, pool_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Choose approximate nearest donors from a random same-signature candidate pool.

    This avoids the quadratic memory/runtime cost of all-pairs nearest-neighbour
    search while ensuring each donor is locally close to its anchor.
    """

    candidates = rng.choice(group_indices, size=(len(anchors), pool_size), replace=True)
    distances = np.sum((representation[candidates] - representation[anchors, None, :]) ** 2, axis=2)
    distances[candidates == anchors[:, None]] = np.inf
    best = np.argmin(distances, axis=1)
    donors = candidates[np.arange(len(anchors)), best]
    return donors


def generate_normal_submission_v2(
    train: pd.DataFrame, label_column: str, config: V2Config
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Generate normal-only synthetic rows with empirical modes and local interpolation."""

    if label_column not in train.columns:
        raise ValueError(f"Missing label column: {label_column}")
    if train.isna().any().any():
        raise ValueError("Training data contains missing values.")

    features = [column for column in train.columns if column != label_column]
    normal = train.loc[train[label_column].eq(0), features].reset_index(drop=True)
    if normal.empty:
        raise ValueError("No normal rows are available.")

    categorical, _, numeric = infer_feature_types(normal)
    integer_columns = _integer_columns(normal, numeric)
    skew = normal[numeric].skew() if numeric else pd.Series(dtype=float)
    log_columns = {
        column
        for column in numeric
        if normal[column].min() >= 0 and abs(float(skew[column])) >= config.log_skew_threshold
    }
    representation = _mixed_numeric_representation(normal, numeric, log_columns)
    rng = np.random.default_rng(config.seed)

    if categorical:
        signatures = normal[categorical].astype(str).agg("\x1f".join, axis=1)
    else:
        signatures = pd.Series("__all_normal_rows__", index=normal.index)
    groups = {key: indices.to_numpy() for key, indices in signatures.groupby(signatures, sort=True).groups.items()}
    keys = list(groups)
    empirical = np.array([len(groups[key]) for key in keys], dtype=float)
    empirical /= empirical.sum()
    weights = _capped_weights(empirical, config.coverage_strength, config.max_oversample_factor)
    quotas = _largest_remainder(weights, config.target_n)

    lower = normal[numeric].min() if numeric else pd.Series(dtype=float)
    upper = normal[numeric].max() if numeric else pd.Series(dtype=float)
    generated_parts: list[pd.DataFrame] = []
    interpolation_rows = 0
    donor_distances: list[float] = []

    for key, quota in zip(keys, quotas):
        if quota == 0:
            continue
        indices = groups[key]
        bootstrap_n = int(round(quota * config.bootstrap_fraction))
        bootstrap = normal.iloc[rng.choice(indices, size=bootstrap_n, replace=True)].copy()
        interpolation_n = quota - bootstrap_n

        if interpolation_n and len(indices) >= 2:
            anchors = rng.choice(indices, size=interpolation_n, replace=True)
            donors = _choose_local_donors(anchors, indices, representation, config.donor_candidate_pool, rng)
            interpolated = normal.iloc[anchors].copy().reset_index(drop=True)
            alpha = rng.uniform(0, config.interpolation_alpha, size=interpolation_n)
            donor_distances.extend(np.sqrt(np.sum((representation[anchors] - representation[donors]) ** 2, axis=1)).tolist())

            for column in numeric:
                anchor_values = normal.iloc[anchors][column].to_numpy(dtype=float)
                donor_values = normal.iloc[donors][column].to_numpy(dtype=float)
                if column in log_columns:
                    values = np.expm1((1 - alpha) * np.log1p(anchor_values) + alpha * np.log1p(donor_values))
                else:
                    values = (1 - alpha) * anchor_values + alpha * donor_values
                interpolated[column] = np.clip(values, lower[column], upper[column])

            # Preserve a documented deterministic UNSW-NB15 feature relationship.
            if {"tcprtt", "synack", "ackdat"}.issubset(interpolated.columns):
                interpolated["tcprtt"] = interpolated["synack"] + interpolated["ackdat"]
            for column in integer_columns:
                interpolated[column] = np.rint(interpolated[column]).clip(lower[column], upper[column]).astype(normal[column].dtype)
            generated_parts.append(pd.concat([bootstrap.reset_index(drop=True), interpolated], ignore_index=True))
            interpolation_rows += interpolation_n
        else:
            fallback = normal.iloc[rng.choice(indices, size=interpolation_n, replace=True)] if interpolation_n else normal.iloc[[]]
            generated_parts.append(pd.concat([bootstrap, fallback], ignore_index=True))

    synthetic = pd.concat(generated_parts, ignore_index=True)
    synthetic = synthetic.sample(frac=1, random_state=config.seed).reset_index(drop=True)
    submission = synthetic[features].copy()
    submission.insert(0, "id", np.arange(config.target_n, dtype=int))
    validate_submission(submission, features, config.target_n)
    diagnostics: dict[str, float | int] = {
        "normal_rows": len(normal),
        "categorical_signatures": len(groups),
        "log_interpolated_columns": len(log_columns),
        "interpolated_rows": interpolation_rows,
        "bootstrap_or_fallback_rows": config.target_n - interpolation_rows,
        "exact_duplicate_rows": int(submission.duplicated(subset=features).sum()),
        "mean_scaled_donor_distance": float(np.mean(donor_distances)) if donor_distances else 0.0,
    }
    return submission, diagnostics
