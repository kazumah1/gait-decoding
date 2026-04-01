"""Compatibility wrapper for session preprocessing and windowing helpers."""

from session_processing.preprocessing import (
    apply_transform,
    apply_transforms,
    apply_transforms_over_subject_sessions,
    build_default_preprocessing_transforms,
    preprocess_session_df,
)
from session_processing.sessions import (
    iter_subject_sessions,
    resolve_feature_cols,
)
from session_processing.signal_transforms import (
    bandpass_filter,
    clip,
    common_average_reference,
    notch_filter,
    zscore_channels,
)
from session_processing.windowing import WindowXY, WindowXYMeta, build_windows_over_subject_session, session_df_to_windows
