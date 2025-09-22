import logging
import time

import pandas as pd
import pandas.api.typing as pdtypes
import numpy as np

import util.cache_registry as ucache
import util.load_data as utl
from GLOBALS import *

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


@ucache.lru_cache(maxsize=1)
def preprocess_regression_results(session_id: str, folder_name: str) -> (pd.DataFrame, pdtypes.DataFrameGroupBy,
                                                                         pd.Series):
    start = time.perf_counter()
    # load the regression results from the raw files
    _, _, _, _, regression_results, _, _ = utl.load_data(os.path.join(DATA_FOLDER, session_id, folder_name))

    # extend the regression results so every sensor is in x and y
    complete_regression_results = pd.concat((regression_results,
                                             regression_results.rename(columns={"x": "y", "y": "x"})),
                                            ignore_index=True)

    # group the flattened regression results by the tag
    complete_regression_results_grouped = complete_regression_results.groupby("x")

    # find the maximum correlation per signal tag
    complete_max_correlation = complete_regression_results_grouped.max()
    logger.info(f"[{__name__}]  Preprocessed regression results in {time.perf_counter() - start:0.2f} s.")
    return complete_regression_results, complete_regression_results_grouped, complete_max_correlation
