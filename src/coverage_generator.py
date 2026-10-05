"""Coverage-aware, schema-safe generator for normal-only tabular submissions.

The generator intentionally keeps all categorical values unchanged.  It allocates
the requested output budget across observed categorical signatures, then produces
a mixture of exact normal bootstraps and bounded numeric interpolations within
each signature.  This is a development method, not an implementation of the
course-provided starter-kit baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GenerationConfig:
    """Parameters for one reproducible synthetic-data generation run."""

    target_n: int
    seed: int = 42
    coverage_strength: float = 0.15
    max_oversample_factor: float = 4.0
    bootstrap_fraction: float = 0.50
    interpolation_alpha: float = 0.30

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


def infer_feature_types(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Return categorical, binary numeric, and remaining numeric feature names."""

    categorical = [
        col
        for col in frame.columns
        if pd.api.types.is_object_dtype(frame[col])
        or pd.api.types.is_string_dtype(frame[col])
        or isinstance(frame[col].dtype, pd.CategoricalDtype)
    ]
    numeric = [col for col in frame.columns if col not in categorical]
    binary = []
    continuous_or_integer = []
    for col in numeric:
        values = frame[col].to_numpy()
        if set(np.unique(values).tolist()).issubset({0, 1}):
            binary.append(col)
        else:
            continuous_or_integer.append(col)
    return categorical, binary, continuous_or_integer


def _integer_columns(frame: pd.DataFrame, numeric_columns: list[str]) -> list[str]:
    return [
        col
        for col in numeric_columns
        if np.all(np.isclose(frame[col].to_numpy(dtype=float), np.round(frame[col].to_numpy(dtype=float))))
    ]


def _largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
    raw = weights * total
    quotas = np.floor(raw).astype(int)
    remainder = total - int(quotas.sum())
    if remainder:
        order = np.argsort(-(raw - quotas), kind="stable")
        quotas[order[:remainder]] += 1
    return quotas


def _capped_weights(empirical: np.ndarray, coverage_strength: float, cap: float) -> np.ndarray:
    """Blend empirical and uniform weights, limiting each group's expansion."""

    group_count = len(empirical)
    proposed = (1 - coverage_strength) * empirical + coverage_strength / group_count
    upper = cap * empirical
    active = np.ones(group_count, dtype=bool)
    result = np.zeros(group_count, dtype=float)
    remaining = 1.0

    # Project the proposed distribution onto per-group upper bounds.
    while active.any():
        scaled = proposed[active] / proposed[active].sum() * remaining
        active_indices = np.flatnonzero(active)
        exceeds = scaled > upper[active_indices] + 1e-12
        if not exceeds.any():
            result[active_indices] = scaled
            break
        capped_indices = active_indices[exceeds]
        result[capped_indices] = upper[capped_indices]
        remaining -= float(upper[capped_indices].sum())
        active[capped_indices] = False

    return result / result.sum()


def _signature_key(row: pd.Series, categorical_columns: list[str]) -> tuple[object, ...]:
    return tuple(row[column] for column in categorical_columns)


def generate_normal_submission(
    train: pd.DataFrame, label_column: str, config: GenerationConfig
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Generate an ``id + feature`` submission and compact audit diagnostics."""

    if label_column not in train.columns:
        raise ValueError(f"Missing label column: {label_column}")
    if train.isna().any().any():
        raise ValueError("Training data contains missing values; handle them before generation.")

    features = [column for column in train.columns if column != label_column]
    normal = train.loc[train[label_column] == 0, features].reset_index(drop=True)
    if normal.empty:
        raise ValueError("No normal rows are available.")

    categorical, binary, interpolated_numeric = infer_feature_types(normal)
    integer_columns = _integer_columns(normal, interpolated_numeric)
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

    numeric_lower = normal[interpolated_numeric].min() if interpolated_numeric else pd.Series(dtype=float)
    numeric_upper = normal[interpolated_numeric].max() if interpolated_numeric else pd.Series(dtype=float)
    generated_parts: list[pd.DataFrame] = []
    interpolation_rows = 0

    for key, quota in zip(keys, quotas):
        if quota == 0:
            continue
        indices = groups[key]
        bootstrap_n = int(round(quota * config.bootstrap_fraction))
        bootstrap_positions = rng.choice(indices, size=bootstrap_n, replace=True)
        part = normal.iloc[bootstrap_positions].copy().reset_index(drop=True)

        interpolation_n = quota - bootstrap_n
        if interpolation_n and len(indices) >= 2:
            anchor_positions = rng.choice(indices, size=interpolation_n, replace=True)
            donor_positions = rng.choice(indices, size=interpolation_n, replace=True)
            same = donor_positions == anchor_positions
            while same.any():
                donor_positions[same] = rng.choice(indices, size=int(same.sum()), replace=True)
                same = donor_positions == anchor_positions

            interpolated = normal.iloc[anchor_positions].copy().reset_index(drop=True)
            alpha = rng.uniform(0, config.interpolation_alpha, size=interpolation_n)
            for column in interpolated_numeric:
                anchor_values = normal.iloc[anchor_positions][column].to_numpy(dtype=float)
                donor_values = normal.iloc[donor_positions][column].to_numpy(dtype=float)
                values = (1 - alpha) * anchor_values + alpha * donor_values
                interpolated[column] = np.clip(values, numeric_lower[column], numeric_upper[column])
            for column in integer_columns:
                interpolated[column] = np.rint(interpolated[column]).astype(normal[column].dtype)
            # Categorical and binary columns remain those of the anchor by construction.
            part = pd.concat([part, interpolated], ignore_index=True)
            interpolation_rows += interpolation_n
        elif interpolation_n:
            fallback_positions = rng.choice(indices, size=interpolation_n, replace=True)
            part = pd.concat([part, normal.iloc[fallback_positions]], ignore_index=True)

        generated_parts.append(part)

    synthetic = pd.concat(generated_parts, ignore_index=True)
    synthetic = synthetic.sample(frac=1, random_state=config.seed).reset_index(drop=True)
    submission = synthetic[features].copy()
    submission.insert(0, "id", np.arange(config.target_n, dtype=int))
    validate_submission(submission, features, config.target_n)

    diagnostics: dict[str, float | int] = {
        "normal_rows": len(normal),
        "feature_count": len(features),
        "categorical_columns": len(categorical),
        "binary_columns": len(binary),
        "categorical_signatures": len(groups),
        "interpolated_rows": interpolation_rows,
        "bootstrap_or_fallback_rows": config.target_n - interpolation_rows,
        "exact_duplicate_rows": int(submission.duplicated(subset=features).sum()),
    }
    return submission, diagnostics


def validate_submission(submission: pd.DataFrame, features: list[str], target_n: int) -> None:
    """Raise a clear error when the output violates submission invariants."""

    if len(submission) != target_n:
        raise ValueError(f"Expected {target_n} rows, found {len(submission)}.")
    if submission.columns.tolist() != ["id", *features]:
        raise ValueError("Submission columns do not match id + training feature order.")
    if submission.isna().any().any():
        raise ValueError("Submission contains missing values.")
    numeric = submission.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Submission contains NaN or infinite numeric values.")
    if not np.array_equal(submission["id"].to_numpy(), np.arange(target_n)):
        raise ValueError("id must run consecutively from 0 to target_n - 1.")
