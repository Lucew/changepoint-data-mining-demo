import logging

import pandas as pd
import numpy as np

# get the logger
logger = logging.getLogger("frontend-logger")


def normalization(df: pd.DataFrame, window_length: int = None) -> pd.DataFrame:

    # make a min max scaling function
    if window_length is None:
        mini = df.min()
        maxi = df.max()
        df -= mini

        # make the normalizer
        norma = maxi - mini
        if isinstance(norma, np.float64):
            if norma == 0:
                norma = 1
        else:
            norma[norma == 0] = 1

        # make the normalization
        df /= norma
    else:
        for col in df.columns:
            col_mean = df[col].rolling(window=window_length, min_periods=1).mean()
            col_std = df[col].rolling(window=window_length, min_periods=1).std()

            df.loc[:, col] = (df[col] - col_mean) / col_std
    return df
