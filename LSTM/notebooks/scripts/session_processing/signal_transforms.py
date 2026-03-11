from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def _filter_padlen(a: np.ndarray, b: np.ndarray) -> int:
    return 3 * (max(len(a), len(b)) - 1)


def _validate_bandpass(*, fs: float, bandpass: tuple[float, float]) -> float:
    nyq = fs / 2.0
    low, high = bandpass
    if not (0.0 < low < high < nyq):
        raise ValueError(
            f"Invalid bandpass {bandpass} for fs={fs}. Must satisfy 0 < low < high < {nyq}."
        )
    return nyq


def _validate_notch(*, fs: float, notch_hz: float) -> None:
    nyq = fs / 2.0
    if not (0.0 < notch_hz < nyq):
        raise ValueError(
            f"Invalid notch_hz={notch_hz} for fs={fs}. Must satisfy 0 < notch_hz < {nyq}."
        )


def _validate_signal_length(num_samples: int, required_len: int, *, transform_name: str) -> None:
    if num_samples <= required_len:
        raise ValueError(
            f"Session is too short for {transform_name}: got {num_samples} samples, "
            f"need > {required_len}."
        )


def clip_channel(channel: np.ndarray, *, clip_uv: float) -> np.ndarray:
    """Hard clip one channel to a symmetric amplitude range.

    Input
    -----
    channel : np.ndarray
        One-dimensional signal with shape `(time_samples,)`.
    clip_uv : float
        Positive clipping threshold in microvolts. Values are clipped to
        `[-clip_uv, +clip_uv]`.

    Output
    ------
    np.ndarray
        Clipped channel with shape `(time_samples,)`.
    """
    return np.clip(channel, -clip_uv, clip_uv)


def clip_session(signals: np.ndarray, *, clip_uv: float) -> np.ndarray:
    """Hard clip every channel in one session at once.

    Input
    -----
    signals : np.ndarray
        Session matrix with shape `(time_samples, num_channels)`.
    clip_uv : float
        Positive clipping threshold in microvolts applied to every entry.

    Output
    ------
    np.ndarray
        Clipped session matrix with shape `(time_samples, num_channels)`.
    """
    return np.clip(signals, -clip_uv, clip_uv)


def notch_filter_channel(
    channel: np.ndarray,
    *,
    fs: float,
    notch_hz: float = 60.0,
    notch_q: float = 30.0,
) -> np.ndarray:
    """Apply a notch filter to one channel.

    Input
    -----
    channel : np.ndarray
        One-dimensional signal with shape `(time_samples,)`.
    fs : float
        Sampling rate in Hz.
    notch_hz : float
        Center frequency of the notch in Hz.
    notch_q : float
        Quality factor controlling notch width.

    Output
    ------
    np.ndarray
        Filtered channel with shape `(time_samples,)`.
    """
    _validate_notch(fs=fs, notch_hz=notch_hz)
    b_n, a_n = iirnotch(notch_hz, notch_q, fs)
    _validate_signal_length(len(channel), _filter_padlen(a_n, b_n), transform_name="notch filtering")
    return filtfilt(b_n, a_n, channel)


def notch_filter_session(
    signals: np.ndarray,
    *,
    fs: float,
    notch_hz: float = 60.0,
    notch_q: float = 30.0,
) -> np.ndarray:
    """Apply the same notch filter to every channel in a session matrix.

    Input
    -----
    signals : np.ndarray
        Session matrix with shape `(time_samples, num_channels)`.
    fs : float
        Sampling rate in Hz.
    notch_hz : float
        Center frequency of the notch in Hz.
    notch_q : float
        Quality factor controlling notch width.

    Output
    ------
    np.ndarray
        Filtered session matrix with shape `(time_samples, num_channels)`.
    """
    _validate_notch(fs=fs, notch_hz=notch_hz)
    b_n, a_n = iirnotch(notch_hz, notch_q, fs)
    _validate_signal_length(
        signals.shape[0], _filter_padlen(a_n, b_n), transform_name="notch filtering"
    )
    return filtfilt(b_n, a_n, signals, axis=0)


def bandpass_channel(
    channel: np.ndarray,
    *,
    fs: float,
    bandpass: tuple[float, float] = (0.5, 45.0),
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter to one channel.

    Input
    -----
    channel : np.ndarray
        One-dimensional signal with shape `(time_samples,)`.
    fs : float
        Sampling rate in Hz.
    bandpass : tuple[float, float]
        `(low_hz, high_hz)` cutoff pair.
    order : int
        Butterworth filter order.

    Output
    ------
    np.ndarray
        Filtered channel with shape `(time_samples,)`.
    """
    nyq = _validate_bandpass(fs=fs, bandpass=bandpass)
    low, high = bandpass
    b_bp, a_bp = butter(order, [low / nyq, high / nyq], btype="band")
    _validate_signal_length(len(channel), _filter_padlen(a_bp, b_bp), transform_name="bandpass filtering")
    return filtfilt(b_bp, a_bp, channel)


def bandpass_session(
    signals: np.ndarray,
    *,
    fs: float,
    bandpass: tuple[float, float] = (0.5, 45.0),
    order: int = 4,
) -> np.ndarray:
    """Apply the same Butterworth bandpass filter to every channel in a session.

    Input
    -----
    signals : np.ndarray
        Session matrix with shape `(time_samples, num_channels)`.
    fs : float
        Sampling rate in Hz.
    bandpass : tuple[float, float]
        `(low_hz, high_hz)` cutoff pair.
    order : int
        Butterworth filter order.

    Output
    ------
    np.ndarray
        Filtered session matrix with shape `(time_samples, num_channels)`.
    """
    nyq = _validate_bandpass(fs=fs, bandpass=bandpass)
    low, high = bandpass
    b_bp, a_bp = butter(order, [low / nyq, high / nyq], btype="band")
    _validate_signal_length(
        signals.shape[0], _filter_padlen(a_bp, b_bp), transform_name="bandpass filtering"
    )
    return filtfilt(b_bp, a_bp, signals, axis=0)


def common_average_reference(signals: np.ndarray) -> np.ndarray:
    """Apply common average reference across channels.

    Input
    -----
    signals : np.ndarray
        Session matrix with shape `(time_samples, num_channels)`.

    Output
    ------
    np.ndarray
        Re-referenced session matrix with the same shape
        `(time_samples, num_channels)`.

    For each time sample, the mean across channels is subtracted from every
    channel at that time step.
    """
    return signals - signals.mean(axis=1, keepdims=True)


def zscore_channels(signals: np.ndarray) -> np.ndarray:
    """Z-score each channel using session-level statistics.

    Input
    -----
    signals : np.ndarray
        Session matrix with shape `(time_samples, num_channels)`.

    Output
    ------
    np.ndarray
        Z-scored session matrix with shape `(time_samples, num_channels)`.

    Each channel is normalized independently using its own mean and standard
    deviation computed over the full session.
    """
    mu = signals.mean(axis=0, keepdims=True)
    sigma = signals.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return (signals - mu) / sigma
