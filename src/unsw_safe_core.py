"""UNSW-NB15 anomaly-aware normal-core selection for detector-oriented augmentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.coverage_generator import GenerationConfig, generate_normal_submission, infer_feature_types


@dataclass(frozen=True)
class SafeCoreConfig:
    """Configuration for retaining normal records farthest from observed anomalies."""

    target_n: int
    seed: int = 42
    # 20% was chosen over more aggressive cuts after cluster-held-out checks.
    boundary_drop_fraction: float = 0.20
    minimum_signature_rows: int = 2
    coverage_strength: float = 0.15
    max_oversample_factor: float = 4.0
    bootstrap_fraction: float = 0.50
    interpolation_alpha: float = 0.30

    def __post_init__(self) -> None:
        if not 0 <= self.boundary_drop_fraction < 1:
            raise ValueError("boundary_drop_fraction must be in [0, 1).")
        if self.minimum_signature_rows < 1:
            raise ValueError("minimum_signature_rows must be positive.")


def select_safe_normal_core(
    train: pd.DataFrame, label_column: str, boundary_drop_fraction: float, minimum_signature_rows: int = 2
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Keep normals far from the nearest known anomaly relative to normal density.

    The distance representation robust-scales numerical columns and one-hot encodes
    categorical columns. For each normal record, ``safety = d(anomaly) / d(normal)``;
    low-safety normals lie close to observed attacks or in sparse normal regions and
    are removed before synthesis. Labels are used only for this selection step and
    never appear in the returned synthetic submission.
    """

    if label_column not in train:
        raise ValueError(f"Missing label column: {label_column}")
    features = [column for column in train if column != label_column]
    normal = train.loc[train[label_column].eq(0), features].reset_index().rename(columns={"index": "_row"})
    anomaly = train.loc[train[label_column].eq(1), features]
    if len(normal) < 2 or anomaly.empty:
        raise ValueError("Safe-core selection requires at least two normal rows and one anomaly row.")

    categorical, _, numeric = infer_feature_types(normal[features])
    transformer = ColumnTransformer(
        [
            ("numeric", RobustScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
    )
    normal_matrix = transformer.fit_transform(normal[features])
    anomaly_matrix = transformer.transform(anomaly[features])
    normal_distance = NearestNeighbors(n_neighbors=2, n_jobs=-1).fit(normal_matrix).kneighbors(
        normal_matrix, return_distance=True
    )[0][:, 1]
    anomaly_distance = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(anomaly_matrix).kneighbors(
        normal_matrix, return_distance=True
    )[0][:, 0]
    safety = anomaly_distance / (normal_distance + 1e-8)
    threshold = float(np.quantile(safety, boundary_drop_fraction))
    keep = safety >= threshold
    # A global boundary cut can erase an entire rare categorical mode. Retain the
    # safest representatives of every observed signature so the generator cannot
    # silently lose a legal normal state.
    if categorical:
        signatures = normal[categorical].astype(str).agg("\x1f".join, axis=1)
        for _, positions in signatures.groupby(signatures, sort=False).groups.items():
            positions = np.asarray(list(positions), dtype=int)
            missing = minimum_signature_rows - int(keep[positions].sum())
            if missing > 0:
                best = positions[np.argsort(safety[positions])[-missing:]]
                keep[best] = True
    kept_rows = normal.loc[keep, "_row"].to_numpy()
    filtered = pd.concat([train.loc[kept_rows], train.loc[train[label_column].eq(1)]], ignore_index=True)
    diagnostics: dict[str, float | int] = {
        "input_normal_rows": len(normal),
        "retained_normal_rows": int(keep.sum()),
        "removed_normal_rows": int((~keep).sum()),
        "retained_signatures": int(signatures[keep].nunique()) if categorical else 1,
        "safety_threshold": threshold,
        "median_safety": float(np.median(safety)),
    }
    return filtered, diagnostics


def generate_unsw_safe_core_submission(
    train: pd.DataFrame, label_column: str, config: SafeCoreConfig
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Select the normal core, then synthesize with the established coverage v1 recipe."""

    filtered, diagnostics = select_safe_normal_core(
        train, label_column, config.boundary_drop_fraction, config.minimum_signature_rows
    )
    submission, generation_diagnostics = generate_normal_submission(
        filtered,
        label_column,
        GenerationConfig(
            target_n=config.target_n,
            seed=config.seed,
            coverage_strength=config.coverage_strength,
            max_oversample_factor=config.max_oversample_factor,
            bootstrap_fraction=config.bootstrap_fraction,
            interpolation_alpha=config.interpolation_alpha,
        ),
    )
    diagnostics.update(generation_diagnostics)
    return submission, diagnostics
