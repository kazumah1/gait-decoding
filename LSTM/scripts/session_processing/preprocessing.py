from __future__ import annotations

"""
Preprocessing guide
===================

- transforms operate on a full session matrix with shape `(time, features)`
- transforms should preserve that shape
- DataFrames are only used at the edges to select feature columns and write the transformed matrix back into the same columns

To add a preprocessing step:

1. Implement a session transform in `signal_transforms.py` with signature `fn(signals: np.ndarray, *, ...) -> np.ndarray`.
2. Add it to `build_default_preprocessing_transforms()` if it belongs in the default pipeline, or pass it directly to `apply_transform()` / `apply_transforms()` for explicit control.
3. Use `functools.partial(...)` to bind parameters like `fs`, `bandpass`, or `clip_uv` when building the transform list.

The end-to-end flow is:

- resolve feature columns from the session DataFrame
- extract `signals = df_session[feature_cols].to_numpy(...)`
- run the ordered transform list over that matrix
- write the result back into the same `feature_cols`
"""

from functools import partial
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd

from . import signal_transforms
from .sessions import iter_subject_sessions, resolve_feature_cols


TransformFn = Callable[..., np.ndarray]
FloatDType = np.dtype[np.floating[Any]] | type[np.floating[Any]]


def _apply_transform_sequence(signals: np.ndarray, transforms: Iterable[TransformFn]) -> np.ndarray:
    current_signals = signals
    for transform in transforms:
        current_signals = np.asarray(transform(current_signals), dtype=np.float64)
        if current_signals.ndim != 2:
            raise ValueError("Transforms must return a 2D array with shape `(time, features)`.")
        if current_signals.shape != signals.shape:
            raise ValueError(
                "Preprocessing transforms must preserve shape. "
                f"Expected {signals.shape}, got {current_signals.shape}."
            )
    return current_signals


def _apply_transforms_to_dataframe(
    df_session: pd.DataFrame,
    transforms: Iterable[TransformFn],
    *,
    feature_cols: list[str],
    output_dtype: FloatDType,
) -> pd.DataFrame:
    signals = df_session[feature_cols].to_numpy(dtype=np.float64)
    output_signals = _apply_transform_sequence(signals, transforms)

    df_out = df_session.copy()
    df_out.loc[:, feature_cols] = output_signals.astype(output_dtype)
    return df_out


def apply_transform(
    df_session: pd.DataFrame,
    transform: TransformFn,
    *,
    feature_cols: Optional[Iterable[str]] = None,
    time_col: str = "Time:512Hz",
    target_col: Optional[str] = None,
    output_dtype: FloatDType = np.float32,
) -> pd.DataFrame:
    """Apply one session-matrix transform to a single session dataframe."""
    return apply_transforms(
        df_session,
        [transform],
        feature_cols=feature_cols,
        time_col=time_col,
        target_col=target_col,
        output_dtype=output_dtype,
    )


def apply_transforms(
    df_session: pd.DataFrame,
    transforms: Iterable[TransformFn],
    *,
    feature_cols: Optional[Iterable[str]] = None,
    time_col: str = "Time:512Hz",
    target_col: Optional[str] = None,
    output_dtype: FloatDType = np.float32,
) -> pd.DataFrame:
    """Apply an ordered session-transform sequence to one session dataframe."""
    if df_session.empty:
        raise ValueError("df_session is empty.")
    if time_col not in df_session.columns:
        raise ValueError(f"time_col '{time_col}' not found in df_session.")

    sorted_df = df_session.sort_values(time_col).reset_index(drop=True).copy()
    resolved_feature_cols = resolve_feature_cols(
        sorted_df, feature_cols, time_col=time_col, target_col=target_col
    )
    return _apply_transforms_to_dataframe(
        sorted_df,
        transforms,
        feature_cols=resolved_feature_cols,
        output_dtype=output_dtype,
    )


def apply_transforms_over_subject_sessions(
    df: pd.DataFrame,
    transforms: Iterable[TransformFn],
    *,
    feature_cols: Optional[Iterable[str]] = None,
    subject_col: str = "Subject",
    session_col: str = "Session",
    time_col: str = "Time:512Hz",
    target_col: Optional[str] = None,
    output_dtype: FloatDType = np.float32,
) -> pd.DataFrame:
    """Apply the same transform sequence to every session in a full dataframe."""
    if df.empty:
        raise ValueError("df is empty.")

    resolved_feature_cols = resolve_feature_cols(df, feature_cols, time_col=time_col, target_col=target_col)
    processed_sessions: list[pd.DataFrame] = []

    for _subject_id, _session_id, df_session in iter_subject_sessions(
        df, subject_col=subject_col, session_col=session_col
    ):
        processed_sessions.append(
            apply_transforms(
                df_session,
                transforms,
                feature_cols=resolved_feature_cols,
                time_col=time_col,
                target_col=target_col,
                output_dtype=output_dtype,
            )
        )

    if not processed_sessions:
        return df.iloc[0:0].copy()
    return pd.concat(processed_sessions, axis=0, ignore_index=True)


def build_default_preprocessing_transforms(
    *,
    fs: float = 512.0,
    bandpass: tuple[float, float] = (0.5, 45.0),
    notch_hz: Optional[float] = 60.0,
    notch_q: float = 30.0,
    apply_car: bool = True,
    apply_zscore: bool = True,
    clip_uv: Optional[float] = None,
) -> list[TransformFn]:
    """Build the default ordered preprocessing transforms for one session.

    Clipping is opt-in because raw signal units vary by dataset.
    """
    transforms: list[TransformFn] = []
    if clip_uv is not None:
        transforms.append(partial(signal_transforms.clip, clip_uv=clip_uv))
    if notch_hz is not None:
        transforms.append(partial(signal_transforms.notch_filter, fs=fs, notch_hz=notch_hz, notch_q=notch_q))
    transforms.append(partial(signal_transforms.bandpass_filter, fs=fs, bandpass=bandpass))
    if apply_car:
        transforms.append(signal_transforms.common_average_reference)
    if apply_zscore:
        transforms.append(signal_transforms.zscore_channels)
    return transforms


def preprocess_session_df(
    df_session: pd.DataFrame,
    *,
    feature_cols: Optional[Iterable[str]] = None,
    time_col: str = "Time:512Hz",
    target_col: Optional[str] = None,
    fs: float = 512.0,
    bandpass: tuple[float, float] = (0.5, 45.0),
    notch_hz: Optional[float] = 60.0,
    notch_q: float = 30.0,
    apply_car: bool = True,
    apply_zscore: bool = True,
    clip_uv: Optional[float] = None,
) -> pd.DataFrame:
    """Run the default preprocessing transform list over one session dataframe."""
    return apply_transforms(
        df_session,
        build_default_preprocessing_transforms(
            fs=fs,
            bandpass=bandpass,
            notch_hz=notch_hz,
            notch_q=notch_q,
            apply_car=apply_car,
            apply_zscore=apply_zscore,
            clip_uv=clip_uv,
        ),
        feature_cols=feature_cols,
        time_col=time_col,
        target_col=target_col,
        output_dtype=np.float32,
    )
