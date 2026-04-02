import numpy as np
import pandas as pd
import math


def time2samples(signal_df: pd.DataFrame, window_size: str):

    # check that the index is a time stamp
    assert isinstance(signal_df.index.dtype, np.dtypes.DateTime64DType), \
        f"Index of data frame is [{signal_df.index.dtype}] not np.dtypes.DateTime64DType."
    # check that the time stamps are regular
    diffs = signal_df.index.to_series().diff().iloc[1:].unique()
    assert len(diffs) == 1, f"Time is irregular."

    # compute the window size from the time into samples given the sampling rate (round up)
    window_size = math.ceil(pd.to_timedelta(window_size).total_seconds()/(diffs[0].total_seconds()))
    return window_size


def find_score(signal_df: pd.DataFrame, prefix: str = None, suffix: str = None):

    if prefix is None and suffix is None:
        prefix = "score"
        suffix = ""
    elif prefix is None:
        prefix = ""
    elif suffix is None:
        suffix = ""

    return [col for col in signal_df.columns if col.startswith(prefix) and col.endswith(suffix)]


def get_anomaly_stamps(annotations: pd.DataFrame, name: str) -> (np.dtypes.DateTime64DType, np.dtypes.DateTime64DType):
    return annotations.loc[name]["Anomalie von"], annotations.loc[name]["Anomalie bis"]
