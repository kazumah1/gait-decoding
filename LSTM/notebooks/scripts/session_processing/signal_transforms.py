from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def _filter_padlen(a: np.ndarray, b: np.ndarray) -> int:
    return 3 * (max(len(a), len(b)) - 1)


def _coerce_filter_coefficients(coefficients: object, *, filter_name: str) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(coefficients, tuple) or len(coefficients) < 2:
        raise TypeError(f"{filter_name} did not return valid filter coefficients.")
    b_coeffs = np.asarray(coefficients[0], dtype=np.float64)
    a_coeffs = np.asarray(coefficients[1], dtype=np.float64)
    return b_coeffs, a_coeffs


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


def clip(signals: np.ndarray, *, clip_uv: float) -> np.ndarray:
    """Hard clip every feature channel in a session matrix."""
    return np.clip(signals, -clip_uv, clip_uv)


def notch_filter(
    signals: np.ndarray,
    *,
    fs: float,
    notch_hz: float = 60.0,
    notch_q: float = 30.0,
) -> np.ndarray:
    """Apply the same notch filter to every feature channel in a session matrix."""
    _validate_notch(fs=fs, notch_hz=notch_hz)
    b_n, a_n = _coerce_filter_coefficients(
        iirnotch(notch_hz, notch_q, fs),
        filter_name="iirnotch",
    )
    _validate_signal_length(
        signals.shape[0], _filter_padlen(a_n, b_n), transform_name="notch filtering"
    )
    return filtfilt(b_n, a_n, signals, axis=0)


def bandpass_filter(
    signals: np.ndarray,
    *,
    fs: float,
    bandpass: tuple[float, float] = (0.5, 45.0),
    order: int = 4,
) -> np.ndarray:
    """Apply the same Butterworth bandpass filter to every feature channel in a session."""
    nyq = _validate_bandpass(fs=fs, bandpass=bandpass)
    low, high = bandpass
    b_bp, a_bp = _coerce_filter_coefficients(
        butter(order, [low / nyq, high / nyq], btype="band"),
        filter_name="butter",
    )
    _validate_signal_length(
        signals.shape[0], _filter_padlen(a_bp, b_bp), transform_name="bandpass filtering"
    )
    return filtfilt(b_bp, a_bp, signals, axis=0)


def common_average_reference(signals: np.ndarray) -> np.ndarray:
    """Apply common average reference across channels."""
    return signals - signals.mean(axis=1, keepdims=True)


def zscore_channels(signals: np.ndarray) -> np.ndarray:
    """Z-score each channel using session-level statistics."""
    mu = signals.mean(axis=0, keepdims=True)
    sigma = signals.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return (signals - mu) / sigma
