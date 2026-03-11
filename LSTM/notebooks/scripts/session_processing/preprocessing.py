from __future__ import annotations

"""
Preprocessing extension guide
=============================

Use this module when you want to define a new preprocessing step and run it
through the dataframe/session tools here.

1. Implement the actual signal operation in `signal_transforms.py`.
    Use one of these function shapes:

    - Per-channel transform:
         `fn(channel: np.ndarray, *, ...) -> np.ndarray`
    - Per-channel-with-context transform:
         `fn(channel: np.ndarray, signals: np.ndarray, *, ...) -> np.ndarray`
    - Full-session transform:
         `fn(signals: np.ndarray, *, ...) -> np.ndarray`

2. Register the transform in this file so the dispatcher knows how to call it.
    The registration helpers are:

    - `register_per_channel_transform(...)`
    - `register_per_channel_with_context_transform(...)`
    - `register_session_transform(...)`

3. If a per-channel transform also has a faster matrix-wide implementation,
    register that implementation with:

    - `register_session_implementation(channel_fn, session_fn)`

    This lets callers keep using one logical transform while the dispatcher uses
    the more efficient session-level version internally.

4. If the transform changes the number of output channels, define output column
    names with one of these approaches:

    - pass `output_feature_cols=...` to `apply_transform(...)`
    - or register a resolver with `register_output_feature_name_resolver(transform, resolver)`

5. Build and run preprocessing pipelines with these functions:

    - `apply_transform(...)`
    - `apply_transforms(...)`
    - `apply_transforms_over_subject_sessions(...)`
    - `build_default_preprocessing_transforms(...)`
    - `preprocess_session_df(...)`

6. Use `functools.partial(...)` to bind parameters like `fs`, `bandpass`,
    `clip_uv`, etc. The helper logic in this module unwraps partials before
    dispatch, so registration still works.

Typical workflow:

- write the transform in `signal_transforms.py`
- register it with one of:
    `register_per_channel_transform`,
    `register_per_channel_with_context_transform`,
    `register_session_transform`
- optionally register `register_session_implementation`
- optionally register `register_output_feature_name_resolver`
- then either add it to `build_default_preprocessing_transforms()` or pass it directly into `apply_transforms()`
"""

from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, cast

import numpy as np
import pandas as pd

from . import signal_transforms
from .sessions import iter_subject_sessions, resolve_feature_cols


TransformFn = Callable[..., np.ndarray]
FeatureNameResolver = Callable[[Sequence[str], np.ndarray], List[str]]
TransformKind = Literal["per_channel", "per_channel_with_context", "session"]
FloatDType = np.dtype[np.floating[Any]] | type[np.floating[Any]]

_TRANSFORM_KINDS: Dict[TransformFn, TransformKind] = {}
_SESSION_IMPLEMENTATIONS: Dict[TransformFn, TransformFn] = {}
_OUTPUT_FEATURE_NAME_RESOLVERS: Dict[TransformFn, FeatureNameResolver] = {}


def _unwrap_transform(transform: TransformFn) -> TransformFn:
    current = transform
    while isinstance(current, partial):
        current = current.func
    return cast(TransformFn, current)


def _collect_partial_bindings(transform: TransformFn) -> tuple[tuple[Any, ...], Dict[str, Any]]:
    current: TransformFn = transform
    partial_chain: List[partial[Any]] = []
    while isinstance(current, partial):
        partial_chain.append(current)
        current = current.func

    args: List[Any] = []
    kwargs: Dict[str, Any] = {}
    for partial_transform in reversed(partial_chain):
        args.extend(partial_transform.args)
        if partial_transform.keywords:
            kwargs.update(partial_transform.keywords)
    return tuple(args), kwargs


def register_per_channel_transform(transform: TransformFn) -> TransformFn:
    """Register a transform with input/output shape `(time,) -> (time,)`."""
    _TRANSFORM_KINDS[_unwrap_transform(transform)] = "per_channel"
    return transform


def register_per_channel_with_context_transform(transform: TransformFn) -> TransformFn:
    """Register a transform with shape `(time,), (time, channels) -> (time,)`."""
    _TRANSFORM_KINDS[_unwrap_transform(transform)] = "per_channel_with_context"
    return transform


def register_session_transform(transform: TransformFn) -> TransformFn:
    """Register a transform with shape `(time, channels) -> (time, new_channels)`."""
    _TRANSFORM_KINDS[_unwrap_transform(transform)] = "session"
    return transform


def register_session_implementation(transform: TransformFn, session_transform: TransformFn) -> TransformFn:
    """Register the session-matrix implementation for a per-channel transform."""
    _SESSION_IMPLEMENTATIONS[_unwrap_transform(transform)] = _unwrap_transform(session_transform)
    return session_transform


def register_output_feature_name_resolver(
    transform: TransformFn,
    resolver: FeatureNameResolver,
) -> FeatureNameResolver:
    """Register how a shape-changing transform should name its output columns."""
    _OUTPUT_FEATURE_NAME_RESOLVERS[_unwrap_transform(transform)] = resolver
    return resolver


def _resolve_transform_kind(transform: TransformFn) -> Optional[TransformKind]:
    return _TRANSFORM_KINDS.get(_unwrap_transform(transform))


def _dispatch_transform(transform: TransformFn, signals: np.ndarray) -> np.ndarray:
    """Run one registered transform over a `(time, channels)` session matrix.

    Input
    -----
    transform : callable
        Registered transform or partial thereof.
    signals : np.ndarray
        Session matrix with shape `(time_samples, num_channels)`.

    Output
    ------
    np.ndarray
        Output matrix with shape `(time_samples, output_channels)`.
    """
    base_transform = _unwrap_transform(transform)
    transform_kind = _resolve_transform_kind(transform)
    bound_args, bound_kwargs = _collect_partial_bindings(transform)

    if transform_kind == "session":
        return np.asarray(base_transform(signals, *bound_args, **bound_kwargs), dtype=np.float64)

    if transform_kind == "per_channel":
        session_impl = _SESSION_IMPLEMENTATIONS.get(base_transform)
        if session_impl is not None:
            return np.asarray(session_impl(signals, *bound_args, **bound_kwargs), dtype=np.float64)
        return np.column_stack([transform(signals[:, idx]) for idx in range(signals.shape[1])]).astype(np.float64)

    if transform_kind == "per_channel_with_context":
        return np.column_stack(
            [transform(signals[:, idx], signals) for idx in range(signals.shape[1])]
        ).astype(np.float64)

    transform_name = getattr(base_transform, "__name__", repr(base_transform))
    raise ValueError(
        f"Transform '{transform_name}' is not registered. "
        "Use register_per_channel_transform, register_per_channel_with_context_transform, "
        "or register_session_transform."
    )


def _resolve_output_feature_cols(
    transform: TransformFn,
    input_feature_cols: Sequence[str],
    output_signals: np.ndarray,
    output_feature_cols: Optional[Sequence[str]],
) -> List[str]:
    if output_feature_cols is not None:
        resolved = list(output_feature_cols)
    elif output_signals.shape[1] == len(input_feature_cols):
        resolved = list(input_feature_cols)
    else:
        resolver = _OUTPUT_FEATURE_NAME_RESOLVERS.get(_unwrap_transform(transform))
        if resolver is not None:
            resolved = list(resolver(input_feature_cols, output_signals))
        else:
            base_name = getattr(_unwrap_transform(transform), "__name__", "transform")
            resolved = [f"{base_name}_{idx}" for idx in range(output_signals.shape[1])]

    if len(resolved) != output_signals.shape[1]:
        raise ValueError(
            f"Output feature name count ({len(resolved)}) does not match "
            f"transform output width ({output_signals.shape[1]})."
        )
    return resolved


def _replace_feature_block(
    df_session: pd.DataFrame,
    old_feature_cols: Sequence[str],
    output_signals: np.ndarray,
    output_feature_cols: Sequence[str],
    *,
    output_dtype: FloatDType,
) -> pd.DataFrame:
    column_positions: List[int] = []
    for column_name in old_feature_cols:
        column_loc = df_session.columns.get_loc(column_name)
        if not isinstance(column_loc, int):
            raise ValueError(f"Feature column '{column_name}' must map to a single column.")
        column_positions.append(column_loc)
    first_feature_idx = min(column_positions)
    non_feature_df = df_session.drop(columns=list(old_feature_cols))
    feature_df = pd.DataFrame(
        output_signals.astype(output_dtype),
        index=df_session.index,
        columns=list(output_feature_cols),
    )
    left = non_feature_df.iloc[:, :first_feature_idx]
    right = non_feature_df.iloc[:, first_feature_idx:]
    return pd.concat([left, feature_df, right], axis=1)


def _apply_transform_impl(
    df_session: pd.DataFrame,
    transform: TransformFn,
    *,
    feature_cols: Sequence[str],
    output_feature_cols: Optional[Sequence[str]],
    output_dtype: FloatDType,
) -> tuple[pd.DataFrame, List[str]]:
    signals = df_session[list(feature_cols)].to_numpy(dtype=np.float64)
    output_signals = _dispatch_transform(transform, signals)

    if output_signals.ndim != 2:
        raise ValueError("Transforms must return a 2D array with shape `(time, features)`.")
    if output_signals.shape[0] != signals.shape[0]:
        raise ValueError(
            f"Transforms must preserve the number of rows. Got {output_signals.shape[0]} "
            f"rows from input with {signals.shape[0]} rows."
        )

    resolved_output_feature_cols = _resolve_output_feature_cols(
        transform,
        input_feature_cols=feature_cols,
        output_signals=output_signals,
        output_feature_cols=output_feature_cols,
    )
    df_out = _replace_feature_block(
        df_session,
        old_feature_cols=feature_cols,
        output_signals=output_signals,
        output_feature_cols=resolved_output_feature_cols,
        output_dtype=output_dtype,
    )
    return df_out, resolved_output_feature_cols


def apply_transform(
    df_session: pd.DataFrame,
    transform: TransformFn,
    *,
    feature_cols: Optional[Iterable[str]] = None,
    time_col: str = "Time:512Hz",
    target_col: Optional[str] = None,
    output_feature_cols: Optional[Sequence[str]] = None,
    output_dtype: FloatDType = np.float32,
) -> pd.DataFrame:
    """Apply one registered transform to a single session dataframe.

    Input
    -----
    df_session : pd.DataFrame
        One session dataframe with shape `(time_samples, num_columns)`.
    transform : callable
        Registered signal transform. The callable may represent a per-channel,
        per-channel-with-context, or full-session transform.
    feature_cols : iterable[str] or None
        Feature columns that become the signal matrix with shape
        `(time_samples, num_features)`.
    time_col : str
        Time column used to sort the session before processing.
    target_col : str or None
        Optional target column excluded from inferred feature columns.
    output_feature_cols : sequence[str] or None
        Optional output column names for shape-changing transforms.

    Output
    ------
    pd.DataFrame
        A sorted copy of `df_session` with the transformed feature block
        written back into the dataframe.
    """
    if df_session.empty:
        raise ValueError("df_session is empty.")
    if time_col not in df_session.columns:
        raise ValueError(f"time_col '{time_col}' not found in df_session.")

    resolved_feature_cols = resolve_feature_cols(
        df_session, feature_cols, time_col=time_col, target_col=target_col
    )
    sorted_df = df_session.sort_values(time_col).reset_index(drop=True).copy()
    df_out, _ = _apply_transform_impl(
        sorted_df,
        transform,
        feature_cols=resolved_feature_cols,
        output_feature_cols=output_feature_cols,
        output_dtype=output_dtype,
    )
    return df_out


def apply_transforms(
    df_session: pd.DataFrame,
    transforms: Iterable[TransformFn],
    *,
    feature_cols: Optional[Iterable[str]] = None,
    time_col: str = "Time:512Hz",
    target_col: Optional[str] = None,
    output_dtype: FloatDType = np.float32,
) -> pd.DataFrame:
    """Apply an ordered transform sequence to one session dataframe.

    The underlying session matrix always has shape `(time_samples, current_features)`.
    A transform may preserve the feature width or project it into a new one.
    """
    if df_session.empty:
        raise ValueError("df_session is empty.")
    if time_col not in df_session.columns:
        raise ValueError(f"time_col '{time_col}' not found in df_session.")

    current_df = df_session.sort_values(time_col).reset_index(drop=True).copy()
    current_feature_cols = resolve_feature_cols(
        current_df, feature_cols, time_col=time_col, target_col=target_col
    )

    for transform in transforms:
        current_df, current_feature_cols = _apply_transform_impl(
            current_df,
            transform,
            feature_cols=current_feature_cols,
            output_feature_cols=None,
            output_dtype=output_dtype,
        )

    return current_df


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
    processed_sessions: List[pd.DataFrame] = []

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
    clip_uv: Optional[float] = 150.0,
) -> List[TransformFn]:
    """Build the default ordered preprocessing transforms for one session."""
    transforms: List[TransformFn] = []
    if clip_uv is not None:
        transforms.append(partial(signal_transforms.clip_channel, clip_uv=clip_uv))
    if notch_hz is not None:
        transforms.append(
            partial(signal_transforms.notch_filter_channel, fs=fs, notch_hz=notch_hz, notch_q=notch_q)
        )
    transforms.append(partial(signal_transforms.bandpass_channel, fs=fs, bandpass=bandpass))
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
    clip_uv: Optional[float] = 150.0,
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


register_per_channel_transform(signal_transforms.clip_channel)
register_session_implementation(signal_transforms.clip_channel, signal_transforms.clip_session)

register_per_channel_transform(signal_transforms.notch_filter_channel)
register_session_implementation(
    signal_transforms.notch_filter_channel,
    signal_transforms.notch_filter_session,
)

register_per_channel_transform(signal_transforms.bandpass_channel)
register_session_implementation(
    signal_transforms.bandpass_channel,
    signal_transforms.bandpass_session,
)

register_session_transform(signal_transforms.common_average_reference)
register_session_transform(signal_transforms.zscore_channels)
