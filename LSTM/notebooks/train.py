from __future__ import annotations

from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union, cast, overload

import numpy as np
import pandas as pd

#Types
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
    lag_s: float = 0.0, # should do more research on what the best value is here (maybe 120ms?)
    dropna: bool = True,
    return_meta: bool = False,
) -> Union[WindowXY, WindowXYMeta]:
    """
    Convert a single-session dataframe into sliding windows for LSTM-style models.

    Assumptions
    -----------
    df_session contains data for exactly ONE subject and ONE session (already filtered).

    Returns
    -------
    X : np.ndarray
        Shape (num_windows, T, num_features)
    y : np.ndarray or None
        Shape (num_windows,)
        (single-step regression: y taken at window end + lag)
    meta : pd.DataFrame (optional)
        Window bookkeeping (start/end indices and times)
    """
    if df_session.empty:
        raise ValueError("df_session is empty.")

    session_df_processed = df_session.copy()

    # Always sort to guarantee correct temporal ordering
    session_df_processed = session_df_processed.sort_values(time_col).reset_index(drop=True)

    # Choose feature columns if not provided
    if feature_cols is None:
        exclude = {time_col, "ID", "Subject", "Session"}
        if target_col is not None:
            exclude.add(target_col)
        numeric_columns = session_df_processed.select_dtypes(include=[np.number]).columns
        feature_cols = [column_name for column_name in numeric_columns if column_name not in exclude]

    assert feature_cols is not None
    feature_cols = list(feature_cols)

    # Drop NaNs in features/target (optional)
    required_columns = feature_cols + ([target_col] if target_col is not None else [])
    required_columns = [column_name for column_name in required_columns if column_name is not None]
    if dropna and required_columns:
        session_df_processed = session_df_processed.dropna(subset=required_columns).reset_index(drop=True)

    # Convert seconds -> samples
    window_size_samples = int(round(window_s * fs))
    stride_samples = int(round(stride_s * fs))
    lag_samples = int(round(lag_s * fs))

    if window_size_samples <= 0:
        raise ValueError("window_s must be > 0.")
    if stride_samples <= 0:
        raise ValueError("stride_s must be > 0.")
    if lag_samples < 0:
        raise ValueError("lag_s must be >= 0.")

    feature_matrix = session_df_processed[feature_cols].to_numpy(dtype=np.float32) # consider using float64 for more precision, but it will take more memory and may not be necessary

    target_values = None
    if target_col is not None:
        target_values = session_df_processed[target_col].to_numpy()

    num_rows = len(session_df_processed)
    # window is [s, s+T), target at (s+T-1+lag)
    last_valid_window_start = num_rows - window_size_samples - lag_samples
    if last_valid_window_start <= 0:
        # Not enough samples for even one window
        windows_array = np.empty((0, window_size_samples, len(feature_cols)), dtype=np.float32)
        if return_meta:
            return windows_array, None if target_col is None else np.empty((0,), dtype=float), pd.DataFrame()
        return windows_array, None if target_col is None else np.empty((0,), dtype=float)

    window_start_indices = np.arange(0, last_valid_window_start, stride_samples, dtype=int)

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
            # pythons typing system is genuinely a pain
            start_time = float(cast(float, session_df_processed.at[start_idx, time_col]))
            end_time = float(cast(float, session_df_processed.at[end_idx_exclusive - 1, time_col]))
            y_time = float(cast(float, session_df_processed.at[target_idx, time_col]))
            window_metadata_rows.append({
                "start_idx": start_idx,
                "end_idx_exclusive": end_idx_exclusive,
                "y_idx": target_idx,
                "start_time": start_time,
                "end_time": end_time,
                "y_time": y_time,
            })

    windows_array = np.stack(window_feature_batches, axis=0) if window_feature_batches else np.empty((0, window_size_samples, len(feature_cols)), dtype=np.float32)
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
    return_meta: bool = True,
) -> Union[WindowXY, WindowXYMeta]:
    """
    Split a full dataframe into (Subject, Session) groups, sort each by time,
    window each group (via session_df_to_windows), then concatenate everything.

    Returns
    -------
    X_all : np.ndarray
        (total_windows, T, num_features)
    y_all : np.ndarray or None
        (total_windows,)  (targets aligned 1-to-1 with X_all windows)
    meta_all : pd.DataFrame (optional)
        One row per window with Subject/Session plus window indices/times.
    """
    if df.empty:
        raise ValueError("df is empty.")

    window_feature_chunks: List[np.ndarray] = []
    window_target_chunks: List[np.ndarray] = []
    metadata_chunks: List[pd.DataFrame] = []

    # group_keys=False keeps group df clean; we’ll add subject/session to meta ourselves
    for subject_session_key, session_df in df.groupby([subject_col, session_col], sort=True):
        subject_id, session_id = cast(Tuple[object, object], subject_session_key)
        # Skip empty groups (shouldn’t happen, but safe)
        if session_df.empty:
            continue

        # Window this single session
        if return_meta:
            session_windows, session_targets, session_meta = session_df_to_windows(
                session_df,
                time_col=time_col,
                feature_cols=feature_cols,
                target_col=target_col,
                fs=fs,
                window_s=window_s,
                stride_s=stride_s,
                lag_s=lag_s,
                dropna=dropna,
                return_meta=True,
            )
        else:
            session_windows, session_targets = session_df_to_windows(
                session_df,
                time_col=time_col,
                feature_cols=feature_cols,
                target_col=target_col,
                fs=fs,
                window_s=window_s,
                stride_s=stride_s,
                lag_s=lag_s,
                dropna=dropna,
                return_meta=False,
            )
            session_meta = None

        # If this session is too short, it may yield 0 windows
        if session_windows.shape[0] == 0:
            continue

        window_feature_chunks.append(session_windows)
        if session_targets is not None:
            window_target_chunks.append(session_targets)

        if return_meta and session_meta is not None:
            subject_any = cast(Any, subject_id)
            session_any = cast(Any, session_id)
            session_meta = session_meta.copy()
            session_meta[subject_col] = subject_any
            session_meta[session_col] = session_any
            metadata_chunks.append(session_meta)

    # Concatenate
    if not window_feature_chunks:
        # No windows produced at all
        window_size_samples = int(round(window_s * fs))
        num_features = len(list(feature_cols)) if feature_cols is not None else 0
        X_all = np.empty((0, window_size_samples, num_features), dtype=np.float32)
        y_all = None if target_col is None else np.empty((0,), dtype=float)
        if return_meta:
            return X_all, y_all, pd.DataFrame()
        return X_all, y_all

    X_all = np.concatenate(window_feature_chunks, axis=0)

    if target_col is not None:
        y_all = np.concatenate(window_target_chunks, axis=0) if window_target_chunks else np.empty((0,), dtype=float)
    else:
        y_all = None

    if return_meta:
        meta_all = pd.concat(metadata_chunks, axis=0, ignore_index=True) if metadata_chunks else pd.DataFrame()
        return X_all, y_all, meta_all

    return X_all, y_all
