from __future__ import annotations

from typing import Iterable, Iterator, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

def resolve_feature_cols(
    df: pd.DataFrame,
    feature_cols: Optional[Iterable[str]],
    *,
    time_col: str,
    target_col: Optional[str] = None,
) -> List[str]:
    """Resolve the ordered feature columns used for signal processing.

    Input
    -----
    df : pd.DataFrame
        Full dataframe or one session dataframe with shape `(num_rows, num_columns)`.
    feature_cols : iterable[str] or None
        Explicit feature columns to use. If `None`, numeric columns are inferred.
    time_col : str
        Name of the time column excluded from inferred feature columns.
    target_col : str or None
        Optional target column excluded from inferred feature columns.

    Output
    ------
    list[str]
        Ordered feature column names with length `num_features`.
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


def iter_subject_sessions(
    df: pd.DataFrame,
    *,
    subject_col: str = "Subject",
    session_col: str = "Session",
) -> Iterator[Tuple[object, object, pd.DataFrame]]:
    """Yield one `(subject_id, session_id, session_df)` group at a time.

    Input
    -----
    df : pd.DataFrame
        Dataframe containing one or more `(subject, session)` groups.
    subject_col, session_col : str
        Column names used to define session boundaries.

    Output
    ------
    iterator[tuple[object, object, pd.DataFrame]]
        Each yielded dataframe has shape `(session_rows, num_columns)` and
        contains exactly one subject/session recording.
    """
    for subject_session_key, session_df in df.groupby([subject_col, session_col], sort=True):
        subject_id, session_id = cast(Tuple[object, object], subject_session_key)
        if session_df.empty:
            continue
        yield subject_id, session_id, session_df
