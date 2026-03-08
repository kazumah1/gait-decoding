from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Iterator, List, Literal, Optional, Tuple, Union, cast, overload

import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt, iirnotch


def _resolve_feature_cols(
    df: pd.DataFrame,
    feature_cols: Optional[Iterable[str]],
    *,
    time_col: str,
    target_col: Optional[str] = None,
) -> List[str]:
    """
    Determine which columns should be treated as model features.

    Input
    -----
    df : pd.DataFrame
        Shape `(num_rows, num_columns)`.
    feature_cols : iterable[str] or None
        User-provided feature names. If not `None`, every entry must be a
        column present in `df`.

    Output
    ------
    list[str]
        Length `num_features`. Ordered list of feature column names that will
        be used for preprocessing and window extraction.

    If `feature_cols` is omitted, the function infers numeric columns and
    removes bookkeeping columns such as time, subject/session identifiers,
    and the optional target column. The returned list is validated against
    the dataframe before use.
    """
    if feature_cols is None:
        exclude = {time_col, "ID", "Subject", "Session"}
        if target_col is not None:
            exclude.add(target_col)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        resolved = [column_name for column_name in numeric_columns if column_name not in exclude]
    else:
        resolved = list(feature_cols)

    if not resolved:
        raise ValueError("No feature columns were found for preprocessing/windowing.")

    missing = [column_name for column_name in resolved if column_name not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    return resolved


def _filter_padlen(a: np.ndarray, b: np.ndarray) -> int:
    """
    Return the minimum signal length required by `filtfilt` for a filter.

    Input
    -----
    a, b : np.ndarray
        1D filter coefficient arrays with shapes `(filter_order,)`.

    Output
    ------
    int
        Minimum required number of time samples.
    """
    return 3 * (max(len(a), len(b)) - 1)


def _validate_preprocessing_filters(
    *,
    fs: float,
    bandpass: tuple[float, float],
    notch_hz: Optional[float],
) -> float:
    """
    Validate preprocessing cutoff frequencies for the current sample rate.

    Input
    -----
    fs : float
        Sampling rate in Hz.
    bandpass : tuple[float, float]
        `(low_hz, high_hz)`.
    notch_hz : float or None
        Optional notch center frequency in Hz.

    Output
    ------
    float
        Nyquist frequency `fs / 2`.

    Returns the Nyquist frequency so downstream code can reuse it when
    designing filters.
    """
    nyq = fs / 2.0
    low, high = bandpass
    if not (0 < low < high < nyq):
        raise ValueError(
            f"Invalid bandpass {bandpass} for fs={fs}. "
            f"Must satisfy 0 < low < high < {nyq}."
        )
    if notch_hz is not None and not (0.0 < notch_hz < nyq):
        raise ValueError(
            f"Invalid notch_hz={notch_hz} for fs={fs}. Must satisfy 0 < notch_hz < {nyq}."
        )
    return nyq


def _design_preprocessing_filters(
    *,
    fs: float,
    bandpass: tuple[float, float],
    notch_hz: Optional[float],
    notch_q: float,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], int]:
    """
    Build bandpass and optional notch filters for session preprocessing.

    Input
    -----
    fs : float
        Sampling rate in Hz.
    bandpass : tuple[float, float]
        `(low_hz, high_hz)` bandpass cutoff pair.
    notch_hz : float or None
        Optional notch center frequency in Hz.
    notch_q : float
        Notch quality factor.

    Output
    ------
    tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], int]
        `(b_bp, a_bp, b_n, a_n, required_len)` where:
        `b_bp`, `a_bp` are 1D bandpass coefficients,
        `b_n`, `a_n` are optional 1D notch coefficients,
        `required_len` is the minimum session length needed for filtering.

    Returns the numerator/denominator coefficients for both filters plus the
    largest padding length required by `filtfilt`.
    """
    nyq = _validate_preprocessing_filters(fs=fs, bandpass=bandpass, notch_hz=notch_hz)
    low, high = bandpass
    b_bp, a_bp = butter(4, [low / nyq, high / nyq], btype="band")

    if notch_hz is not None:
        b_n, a_n = iirnotch(notch_hz, notch_q, fs)
        notch_padlen = _filter_padlen(a_n, b_n)
    else:
        b_n, a_n = None, None
        notch_padlen = 0

    bp_padlen = _filter_padlen(a_bp, b_bp)
    required_len = max(notch_padlen, bp_padlen)
    return b_bp, a_bp, b_n, a_n, required_len


def _apply_clipping(eeg: np.ndarray, *, clip_uv: Optional[float]) -> np.ndarray:
    """
    Hard-clip EEG amplitudes to a symmetric microvolt threshold if enabled.

    Input
    -----
    eeg : np.ndarray
        Signal array with shape `(time_samples, num_channels)`.
    clip_uv : float or None
        Clipping threshold in microvolts.

    Output
    ------
    np.ndarray
        Array with the same shape as `eeg`.
    """
    if clip_uv is None:
        return eeg
    return np.clip(eeg, -clip_uv, clip_uv)


def _apply_common_average_reference(eeg: np.ndarray) -> np.ndarray:
    """
    Apply common average referencing across channels at each time step.

    Input
    -----
    eeg : np.ndarray
        Array with shape `(time_samples, num_channels)`.

    Output
    ------
    np.ndarray
        Array with shape `(time_samples, num_channels)`.
    """
    return eeg - eeg.mean(axis=1, keepdims=True)


def _apply_channel_zscore(eeg: np.ndarray) -> np.ndarray:
    """
    Normalize each channel independently using session-level mean and std.

    Input
    -----
    eeg : np.ndarray
        Array with shape `(time_samples, num_channels)`.

    Output
    ------
    np.ndarray
        Z-scored array with shape `(time_samples, num_channels)`.
    """
    mu = eeg.mean(axis=0, keepdims=True)
    sigma = eeg.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)  # guard against flat/dead channels
    return (eeg - mu) / sigma


def _run_preprocessing_pipeline(
    eeg: np.ndarray,
    *,
    b_bp: np.ndarray,
    a_bp: np.ndarray,
    b_n: Optional[np.ndarray],
    a_n: Optional[np.ndarray],
    clip_uv: Optional[float],
    apply_car: bool,
    apply_zscore: bool,
) -> np.ndarray:
    """
    Apply the configured preprocessing steps to an EEG matrix.

    Input
    -----
    eeg : np.ndarray
        Signal matrix with shape `(time_samples, num_channels)`, typically
        `float64` before being written back to a dataframe.
    b_bp, a_bp : np.ndarray
        1D bandpass filter coefficients.
    b_n, a_n : np.ndarray or None
        Optional 1D notch filter coefficients.
    clip_uv : float or None
        Optional clipping threshold.
    apply_car : bool
        Whether to apply common average reference.
    apply_zscore : bool
        Whether to z-score each channel.

    Output
    ------
    np.ndarray
        Processed signal matrix with the same shape as `eeg`:
        `(time_samples, num_channels)`.

    The expected input shape is `(time, channels)`. Steps are applied in this
    order: clipping, notch, bandpass, CAR, then per-channel z-scoring.
    """
    processed = _apply_clipping(eeg, clip_uv=clip_uv)

    if b_n is not None and a_n is not None:
        processed = filtfilt(b_n, a_n, processed, axis=0)

    processed = filtfilt(b_bp, a_bp, processed, axis=0)

    if apply_car:
        processed = _apply_common_average_reference(processed)

    if apply_zscore:
        processed = _apply_channel_zscore(processed)

    return processed


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
    clip_uv: Optional[float] = 150.0,
) -> pd.DataFrame:
    """
    Preprocess one subject/session recording before window extraction.

    Input
    -----
    df_session : pd.DataFrame
        Shape `(num_rows, num_columns)`. Must contain one recording only, with
        one row per time sample.
    feature_cols : iterable[str] or None
        Feature columns to process. Resolved to `num_features` channel columns.
    time_col : str
        Name of the sample-time column.
    target_col : str or None
        Optional supervised target column that is preserved but not filtered.

    Output
    ------
    pd.DataFrame
        Shape `(num_rows, num_columns)`. Same columns as `df_session`, but the
        feature columns are replaced with processed `float32` values.

    This function should be run after splitting the full dataset into
    individual sessions and before any sliding-window extraction. That keeps
    filter edge effects and normalization statistics confined to one recording.

    Processing order:
        1. Optional amplitude clipping
        2. Optional notch filtering
        3. Bandpass filtering
        4. Optional common average reference
        5. Optional per-channel z-scoring

    Parameters
    ----------
    df_session : pd.DataFrame
        Single subject+session dataframe, already sorted by time.
    feature_cols : list[str] or None
        EEG channel column names to process (non-EEG cols are untouched).
        If None, numeric columns are inferred (excluding time/id/subject/session/target).
    fs : float
        Sampling rate in Hz.
    bandpass : (low_hz, high_hz)
        Butterworth bandpass cutoffs. (0.5, 45.0) recommended for gait decoding.
        Narrow to (0.5, 10.0) if focusing purely on kinematics.
    notch_hz : float or None
        Power line frequency to notch out. None to skip.
    notch_q : float
        Quality factor for the notch filter (higher = narrower notch).
    apply_car : bool
        Common Average Reference -- subtracts mean across channels per sample.
        Helps reduce volume conduction artifacts.
    apply_zscore : bool
        Z-score each channel over the session. Important for cross-session
        stability since electrode impedance varies.
    clip_uv : float or None
        Clip raw values to [-clip_uv, +clip_uv] before filtering.
        Typical EEG artifact rejection threshold is 100-150 µV.

    Returns
    -------
    pd.DataFrame
        A time-sorted copy of `df_session` with the feature columns replaced by
        the processed signal values.

    Raises
    ------
    ValueError
        If the session is empty, the time column is missing, the feature set is
        invalid, or the recording is too short for zero-phase filtering.
    """
    if df_session.empty:
        raise ValueError("df_session is empty.")
    if time_col not in df_session.columns:
        raise ValueError(f"time_col '{time_col}' not found in df_session.")

    resolved_feature_cols = _resolve_feature_cols(
        df_session, feature_cols, time_col=time_col, target_col=target_col
    )

    # Ensure filters run in temporal order.
    df_out = df_session.sort_values(time_col).reset_index(drop=True).copy()
    eeg = df_out[resolved_feature_cols].to_numpy(dtype=np.float64)  # (T, C)

    b_bp, a_bp, b_n, a_n, required_len = _design_preprocessing_filters(
        fs=fs,
        bandpass=bandpass,
        notch_hz=notch_hz,
        notch_q=notch_q,
    )
    if len(df_out) <= required_len:
        raise ValueError(
            f"Session is too short for filtfilt preprocessing: got {len(df_out)} samples, "
            f"need > {required_len}."
        )

    eeg = _run_preprocessing_pipeline(
        eeg,
        b_bp=b_bp,
        a_bp=a_bp,
        b_n=b_n,
        a_n=a_n,
        clip_uv=clip_uv,
        apply_car=apply_car,
        apply_zscore=apply_zscore,
    )

    df_out[resolved_feature_cols] = eeg.astype(np.float32)
    return df_out

# WindowXY: `(X, y)` where `X.shape == (num_windows, window_size_samples, num_features)`
# and `y` is `None` or has shape `(num_windows,)`.
WindowXY = Tuple[np.ndarray, Optional[np.ndarray]]
# WindowXYMeta: `(X, y, meta)` where `meta.shape == (num_windows, num_meta_columns)`.
WindowXYMeta = Tuple[np.ndarray, Optional[np.ndarray], pd.DataFrame]
# SessionEffect: `(session_df, subject_id, session_id) -> transformed_session_df`.
SessionEffect = Callable[[pd.DataFrame, object, object], pd.DataFrame]


def iter_subject_sessions(
    df: pd.DataFrame,
    *,
    subject_col: str = "Subject",
    session_col: str = "Session",
) -> Iterator[Tuple[object, object, pd.DataFrame]]:
    """
    Iterate over the dataframe one `(subject, session)` group at a time.

    Input
    -----
    df : pd.DataFrame
        Shape `(num_rows, num_columns)` containing one or more sessions.

    Output
    ------
    iterator[tuple[object, object, pd.DataFrame]]
        Yields `(subject_id, session_id, session_df)`, where `session_df` has
        shape `(session_rows, num_columns)` for one group.

    Yields the subject id, session id, and the corresponding group dataframe in
    sorted group-key order.
    """
    for subject_session_key, session_df in df.groupby([subject_col, session_col], sort=True):
        subject_id, session_id = cast(Tuple[object, object], subject_session_key)
        if session_df.empty:
            continue
        yield subject_id, session_id, session_df


def apply_session_effects(
    df_session: pd.DataFrame,
    *,
    effects: Iterable[SessionEffect],
    subject_id: object,
    session_id: object,
) -> pd.DataFrame:
    """
    Run a sequence of session-level transforms on one session dataframe.

    Input
    -----
    df_session : pd.DataFrame
        Shape `(session_rows, num_columns)`.
    effects : iterable[SessionEffect]
        Each effect has signature `(pd.DataFrame, object, object) -> pd.DataFrame`.
    subject_id, session_id : object
        Group identifiers passed through to each effect.

    Output
    ------
    pd.DataFrame
        Final transformed session dataframe. Shape may change depending on the
        supplied effects.

    Each effect receives the current dataframe plus the session identifiers and
    must return the transformed dataframe for the next effect in the chain.
    """
    session_out = df_session
    for effect in effects:
        session_out = effect(session_out, subject_id, session_id)
    return session_out


def apply_effects_over_subject_sessions(
    df: pd.DataFrame,
    *,
    effects: Iterable[SessionEffect],
    subject_col: str = "Subject",
    session_col: str = "Session",
) -> pd.DataFrame:
    """
    Apply the same ordered list of effects to every subject/session group.

    Input
    -----
    df : pd.DataFrame
        Shape `(num_rows, num_columns)` across all sessions.
    effects : iterable[SessionEffect]
        Session transforms to apply to every group.

    Output
    ------
    pd.DataFrame
        Concatenated transformed dataframe. Column set should usually match the
        per-session outputs; row count depends on the applied effects.

    This is the dataframe-level utility for session customization: split into
    sessions, transform each session independently, then concatenate the
    results back into one dataframe.
    """
    if df.empty:
        raise ValueError("df is empty.")

    processed_sessions: List[pd.DataFrame] = []
    for subject_id, session_id, session_df in iter_subject_sessions(
        df, subject_col=subject_col, session_col=session_col
    ):
        processed_sessions.append(
            apply_session_effects(
                session_df,
                effects=effects,
                subject_id=subject_id,
                session_id=session_id,
            )
        )

    if not processed_sessions:
        return df.iloc[0:0].copy()
    return pd.concat(processed_sessions, axis=0, ignore_index=True)


def _build_default_preprocess_effect(
    *,
    feature_cols: Iterable[str],
    time_col: str,
    target_col: Optional[str],
    fs: float,
    preprocess_kwargs: Optional[Dict[str, Any]] = None,
) -> SessionEffect:
    """
    Wrap `preprocess_session_df` as a reusable session effect.

    Input
    -----
    feature_cols : iterable[str]
        Length `num_features`. Feature columns to preprocess.
    time_col : str
        Time column name.
    target_col : str or None
        Optional target column name.
    fs : float
        Sampling rate in Hz.
    preprocess_kwargs : dict or None
        Extra keyword arguments passed to `preprocess_session_df`.

    Output
    ------
    SessionEffect
        Callable with signature `(pd.DataFrame, object, object) -> pd.DataFrame`.

    This lets the higher-level session pipeline treat preprocessing the same
    way as any other custom per-session transform.
    """
    kwargs = dict(preprocess_kwargs or {})
    forbidden = {"feature_cols", "time_col", "target_col", "fs"}
    overlap = sorted(forbidden.intersection(kwargs.keys()))
    if overlap:
        raise ValueError(
            f"preprocess_kwargs cannot override {overlap}; pass them directly via function arguments."
        )

    def _effect(df_session: pd.DataFrame, _subject_id: object, _session_id: object) -> pd.DataFrame:
        return preprocess_session_df(
            df_session,
            feature_cols=feature_cols,
            time_col=time_col,
            target_col=target_col,
            fs=fs,
            **kwargs,
        )

    return _effect


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
    Convert one pre-split session dataframe into sliding windows.

    Input
    -----
    df_session : pd.DataFrame
        Shape `(num_rows, num_columns)` for exactly one session.
    feature_cols : iterable[str] or None
        Resolved to `num_features` feature columns.
    target_col : str or None
        Optional scalar target column.
    fs : float
        Sampling rate in Hz.
    window_s, stride_s, lag_s : float
        Window length, step size, and target lag in seconds.
    dropna : bool
        If `True`, rows with missing feature/target values are removed first.
    return_meta : bool
        If `True`, also return a metadata dataframe.

    Output
    ------
    If `return_meta=False`:
        tuple[np.ndarray, np.ndarray | None]
        `X` has shape `(num_windows, window_size_samples, num_features)` and
        dtype `float32`. `y` is `None` or has shape `(num_windows,)`.
    If `return_meta=True`:
        tuple[np.ndarray, np.ndarray | None, pd.DataFrame]
        Same `X` and `y` plus `meta` with shape `(num_windows, 6)`, containing
        `start_idx`, `end_idx_exclusive`, `y_idx`, `start_time`, `end_time`,
        and `y_time`.

    Assumptions
    -----------
    `df_session` contains data for exactly one session. It may already be
    preprocessed, but this function does not require it.

    Behavior
    --------
    The dataframe is sorted by time, optional NaN rows are dropped, and then
    overlapping windows of shape `(window_size_samples, num_features)` are
    extracted. If `target_col` is provided, each window is paired with the
    target value taken from the last sample in the window plus the configured
    `lag_s`.

    Returns
    -------
    X : np.ndarray
        Shape (num_windows, T, num_features)
    y : np.ndarray or None
        Shape (num_windows,)
        (single-step regression: y taken at window end + lag)
    meta : pd.DataFrame (optional)
        Window bookkeeping with source indices and aligned timestamps.
    """
    if df_session.empty:
        raise ValueError("df_session is empty.")

    session_df_processed = df_session.copy()

    # Always sort to guarantee correct temporal ordering
    session_df_processed = session_df_processed.sort_values(time_col).reset_index(drop=True)

    # Choose feature columns if not provided
    feature_cols = _resolve_feature_cols(
        session_df_processed, feature_cols, time_col=time_col, target_col=target_col
    )

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
    if last_valid_window_start < 0:
        # Not enough samples for even one window
        windows_array = np.empty((0, window_size_samples, len(feature_cols)), dtype=np.float32)
        if return_meta:
            return windows_array, None if target_col is None else np.empty((0,), dtype=float), pd.DataFrame()
        return windows_array, None if target_col is None else np.empty((0,), dtype=float)

    # valid start indices satisfy: start <= num_rows - window_size - lag
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
    apply_preprocessing: bool = True,
    preprocess_kwargs: Optional[Dict[str, Any]] = None,
    session_effects: Optional[Iterable[SessionEffect]] = None,
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
    session_effects: Optional[Iterable[SessionEffect]] = None,
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
    session_effects: Optional[Iterable[SessionEffect]] = None,
    return_meta: bool = True,
) -> Union[WindowXY, WindowXYMeta]:
    """
    Build windows from a full dataframe containing many sessions.

    Input
    -----
    df : pd.DataFrame
        Shape `(num_rows, num_columns)` containing one or more sessions.
    feature_cols : iterable[str] or None
        Resolved once to `num_features` columns and reused for all sessions.
    target_col : str or None
        Optional scalar target column.
    fs : float
        Sampling rate in Hz.
    window_s, stride_s, lag_s : float
        Window length, step size, and target lag in seconds.
    apply_preprocessing : bool
        Whether to insert `preprocess_session_df` into the session pipeline.
    preprocess_kwargs : dict or None
        Extra keyword arguments forwarded to `preprocess_session_df`.
    session_effects : iterable[SessionEffect] or None
        Additional session-level transforms.
    return_meta : bool
        Whether to return per-window metadata.

    Output
    ------
    If `return_meta=False`:
        tuple[np.ndarray, np.ndarray | None]
        `X_all` has shape `(total_windows, window_size_samples, num_features)`.
        `y_all` is `None` or has shape `(total_windows,)`.
    If `return_meta=True`:
        tuple[np.ndarray, np.ndarray | None, pd.DataFrame]
        Same `X_all` and `y_all` plus `meta_all` with shape
        `(total_windows, 8)`, containing the six window metadata columns plus
        `subject_col` and `session_col`.

    The function resolves feature columns once, loops over each
    `(subject, session)` group, optionally applies preprocessing and any custom
    session effects, converts each session into windows, and concatenates the
    results into one training-ready output.

    Parameters
    ----------
    apply_preprocessing : bool
        If `True`, `preprocess_session_df` is applied to each session before
        window extraction.
    preprocess_kwargs : dict or None
        Extra keyword arguments forwarded to `preprocess_session_df`.
    session_effects : iterable of callables or None
        Additional per-session transforms applied after the default
        preprocessing effect, in the order provided.

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
    resolved_feature_cols = _resolve_feature_cols(df, feature_cols, time_col=time_col, target_col=target_col)

    effects: List[SessionEffect] = []
    if apply_preprocessing:
        effects.append(
            _build_default_preprocess_effect(
                feature_cols=resolved_feature_cols,
                time_col=time_col,
                target_col=target_col,
                fs=fs,
                preprocess_kwargs=preprocess_kwargs,
            )
        )
    if session_effects is not None:
        effects.extend(session_effects)

    for subject_id, session_id, session_df in iter_subject_sessions(
        df, subject_col=subject_col, session_col=session_col
    ):
        if effects:
            session_df = apply_session_effects(
                session_df,
                effects=effects,
                subject_id=subject_id,
                session_id=session_id,
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
        num_features = len(resolved_feature_cols)
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
