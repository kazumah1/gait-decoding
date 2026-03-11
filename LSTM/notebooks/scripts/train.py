"""Compatibility wrapper for session preprocessing and windowing helpers."""

from session_processing.preprocessing import (
    apply_transform,
    apply_transforms,
    apply_transforms_over_subject_sessions,
    build_default_preprocessing_transforms,
    preprocess_session_df,
    register_output_feature_name_resolver,
    register_per_channel_transform,
    register_per_channel_with_context_transform,
    register_session_implementation,
    register_session_transform,
)
from session_processing.sessions import (
    SessionEffect,
    apply_effects_over_subject_sessions,
    apply_session_effects,
    iter_subject_sessions,
    resolve_feature_cols,
)
from session_processing.signal_transforms import (
    bandpass_channel,
    bandpass_session,
    clip_channel,
    clip_session,
    common_average_reference,
    notch_filter_channel,
    notch_filter_session,
    zscore_channels,
)
from session_processing.windowing import WindowXY, WindowXYMeta, build_windows_over_subject_session, session_df_to_windows
