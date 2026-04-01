from __future__ import annotations

from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union, cast, overload

import numpy as np
import pandas as pd

from .preprocessing import preprocess_session_df
from .sessions import iter_subject_sessions, resolve_feature_cols


WindowXY = Tuple[np.ndarray, Optional[np.ndarray]]
WindowXYMeta = Tuple[np.ndarray, Optional[np.ndarray], pd.DataFrame]

@overload
def session_df_to_windows(
    df_session: pd.DataFrame,
    *,
    time_col: str = "Time:512Hz",
    feature_cols: Optional[Iterable[str]] = None,
    target_col: Optional[str] = None,
    fs: float = 512.0,
    window_s: float = 1.0,
    stride_s: float = 0.1,
    lag_s: float = 0.0,
    dropna: bool = True,
    return_meta: Literal[False] = False,
) -> WindowXY: ...


@overload
def session_df_to_windows(
    df_session: pd.DataFrame,
    *,
    time_col: str = "Time:512Hz",
    feature_cols: Optional[Iterable[str]] = None,
    target_col: Optional[str] = None,
    fs: float = 512.0,
    window_s: float = 1.0,
    stride_s: float = 0.1,
    lag_s: float = 0.0,
    dropna: bool = True,
    return_meta: Literal[True],
) -> WindowXYMeta: ...


def session_df_to_windows(
    df_session: pd.DataFrame,
    *,
    time_col: str = "Time:512Hz",
    feature_cols: Optional[Iterable[str]] = None,
    target_col: Optional[str] = None,
    fs: float = 512.0,
    window_s: float = 1.0,
    stride_s: float = 0.1,
    lag_s: float = 0.0,
    dropna: bool = True,
    return_meta: bool = False,
) -> Union[WindowXY, WindowXYMeta]:
    """Convert one session dataframe into sliding windows."""
    if df_session.empty:
        raise ValueError("df_session is empty.")

    session_df_processed = df_session.sort_values(time_col).reset_index(drop=True).copy()
    resolved_feature_cols = resolve_feature_cols(
        session_df_processed, feature_cols, time_col=time_col, target_col=target_col
    )

    required_columns = resolved_feature_cols + ([target_col] if target_col is not None else [])
    required_columns = [column_name for column_name in required_columns if column_name is not None]
    if dropna and required_columns:
        session_df_processed = session_df_processed.dropna(subset=required_columns).reset_index(drop=True)

    window_size_samples = int(round(window_s * fs))
    stride_samples = int(round(stride_s * fs))
    lag_samples = int(round(lag_s * fs))

    if window_size_samples <= 0:
        raise ValueError("window_s must be > 0.")
    if stride_samples <= 0:
        raise ValueError("stride_s must be > 0.")
    if lag_samples < 0:
        raise ValueError("lag_s must be >= 0.")

    feature_matrix = session_df_processed[resolved_feature_cols].to_numpy(dtype=np.float32)
    target_values = None if target_col is None else session_df_processed[target_col].to_numpy()

    num_rows = len(session_df_processed)
    last_valid_window_start = num_rows - window_size_samples - lag_samples
    if last_valid_window_start < 0:
        empty_windows = np.empty((0, window_size_samples, len(resolved_feature_cols)), dtype=np.float32)
        empty_targets = None if target_col is None else np.empty((0,), dtype=float)
        if return_meta:
            return empty_windows, empty_targets, pd.DataFrame()
        return empty_windows, empty_targets

    window_start_indices = np.arange(0, last_valid_window_start + 1, stride_samples, dtype=int)
    window_feature_batches: List[np.ndarray] = []
    window_targets: List[float] = []
    window_metadata_rows: List[Dict[str, Union[int, float]]] = []

    for window_start_idx in window_start_indices:
        start_idx = int(window_start_idx)
        end_idx_exclusive = start_idx + window_size_samples
        target_idx = end_idx_exclusive - 1 + lag_samples

        window_feature_batches.append(feature_matrix[start_idx:end_idx_exclusive])

        if target_values is not None:
            window_targets.append(float(target_values[target_idx]))

        if return_meta:
            window_metadata_rows.append(
                {
                    "start_idx": start_idx,
                    "end_idx_exclusive": end_idx_exclusive,
                    "y_idx": target_idx,
                    "start_time": float(cast(float, session_df_processed.at[start_idx, time_col])),
                    "end_time": float(cast(float, session_df_processed.at[end_idx_exclusive - 1, time_col])),
                    "y_time": float(cast(float, session_df_processed.at[target_idx, time_col])),
                }
            )

    windows_array = (
        np.stack(window_feature_batches, axis=0)
        if window_feature_batches
        else np.empty((0, window_size_samples, len(resolved_feature_cols)), dtype=np.float32)
    )
    target_array = np.array(window_targets, dtype=float) if target_values is not None else None

    if return_meta:
        return windows_array, target_array, pd.DataFrame(window_metadata_rows)
    return windows_array, target_array


@overload
def build_windows_over_subject_session(
    df: pd.DataFrame,
    *,
    subject_col: str = "Subject",
    session_col: str = "Session",
    time_col: str = "Time:512Hz",
    feature_cols: Optional[Iterable[str]] = None,
    target_col: Optional[str] = None,
    fs: float = 512.0,
    window_s: float = 1.0,
    stride_s: float = 0.1,
    lag_s: float = 0.0,
    dropna: bool = True,
    apply_preprocessing: bool = True,
    preprocess_kwargs: Optional[Dict[str, Any]] = None,
    return_meta: Literal[False],
) -> WindowXY: ...


@overload
def build_windows_over_subject_session(
    df: pd.DataFrame,
    *,
    subject_col: str = "Subject",
    session_col: str = "Session",
    time_col: str = "Time:512Hz",
    feature_cols: Optional[Iterable[str]] = None,
    target_col: Optional[str] = None,
    fs: float = 512.0,
    window_s: float = 1.0,
    stride_s: float = 0.1,
    lag_s: float = 0.0,
    dropna: bool = True,
    apply_preprocessing: bool = True,
    preprocess_kwargs: Optional[Dict[str, Any]] = None,
    return_meta: Literal[True] = True,
) -> WindowXYMeta: ...


def build_windows_over_subject_session(
    df: pd.DataFrame,
    *,
    subject_col: str = "Subject",
    session_col: str = "Session",
    time_col: str = "Time:512Hz",
    feature_cols: Optional[Iterable[str]] = None,
    target_col: Optional[str] = None,
    fs: float = 512.0,
    window_s: float = 1.0,
    stride_s: float = 0.1,
    lag_s: float = 0.0,
    dropna: bool = True,
    apply_preprocessing: bool = True,
    preprocess_kwargs: Optional[Dict[str, Any]] = None,
    return_meta: bool = True,
) -> Union[WindowXY, WindowXYMeta]:
    """Build windows from a dataframe containing many sessions."""
    if df.empty:
        raise ValueError("df is empty.")

    resolved_feature_cols = resolve_feature_cols(df, feature_cols, time_col=time_col, target_col=target_col)
    preprocess_options = dict(preprocess_kwargs or {})
    forbidden = {"feature_cols", "time_col", "target_col", "fs"}
    overlap = sorted(forbidden.intersection(preprocess_options.keys()))
    if overlap:
        raise ValueError(
            f"preprocess_kwargs cannot override {overlap}; pass them directly via function arguments."
        )

    window_feature_chunks: List[np.ndarray] = []
    window_target_chunks: List[np.ndarray] = []
    metadata_chunks: List[pd.DataFrame] = []

    for subject_id, session_id, session_df in iter_subject_sessions(
        df, subject_col=subject_col, session_col=session_col
    ):
        if apply_preprocessing:
            session_df = preprocess_session_df(
                session_df,
                feature_cols=resolved_feature_cols,
                time_col=time_col,
                target_col=target_col,
                fs=fs,
                **preprocess_options,
            )

        session_window_result = session_df_to_windows(
            session_df,
            time_col=time_col,
            feature_cols=resolved_feature_cols,
            target_col=target_col,
            fs=fs,
            window_s=window_s,
            stride_s=stride_s,
            lag_s=lag_s,
            dropna=dropna,
            return_meta=return_meta,
        )
        if return_meta:
            session_windows, session_targets, session_meta = cast(WindowXYMeta, session_window_result)
        else:
            session_windows, session_targets = cast(WindowXY, session_window_result)
            session_meta = None

        if session_windows.shape[0] == 0:
            continue

        window_feature_chunks.append(session_windows)
        if session_targets is not None:
            window_target_chunks.append(session_targets)

        if return_meta and session_meta is not None:
            session_meta = session_meta.copy()
            session_meta[subject_col] = cast(Any, subject_id)
            session_meta[session_col] = cast(Any, session_id)
            metadata_chunks.append(session_meta)

    if not window_feature_chunks:
        window_size_samples = int(round(window_s * fs))
        num_features = len(resolved_feature_cols)
        x_all = np.empty((0, window_size_samples, num_features), dtype=np.float32)
        y_all = None if target_col is None else np.empty((0,), dtype=float)
        if return_meta:
            return x_all, y_all, pd.DataFrame()
        return x_all, y_all

    x_all = np.concatenate(window_feature_chunks, axis=0)
    y_all = (
        np.concatenate(window_target_chunks, axis=0)
        if target_col is not None
        else None
    )

    if return_meta:
        meta_all = pd.concat(metadata_chunks, axis=0, ignore_index=True) if metadata_chunks else pd.DataFrame()
        return x_all, y_all, meta_all

    return x_all, y_all
