import weakref
import typing
import logging
import time

import pandas as pd
from tqdm import tqdm
import numpy as np

_RESIDUAL_REGISTER = weakref.WeakValueDictionary()
# get the logger
logger = logging.getLogger("frontend-logger")


def compute_weighted_residual_norm(regression_results: pd.DataFrame, signal_list: list[str], scores: dict[str: pd.DataFrame],
                                   min_correlation: float = 0.05, coming_from: str = None, target: str = None):
    start = time.perf_counter()

    # make a dict to save the resulting weighted residuals
    residual_results = dict()

    # get all the rows of the regression result, which applies to the signal pairs
    signal_set = set(signal_list)
    regression_results = regression_results[regression_results["x"].isin(signal_set) &
                                            regression_results["y"].isin(signal_set)]

    # go through each pair in the regression results and compute the residuals and add them to the overall score
    # by accounting for a weight
    # results = {"x": [], "y": [], "correlation": [], "correlation delay": [],
    # "window size": [], "alpha": [], "beta": []}
    # for _, row in tqdm(regression_results.iterrows(), desc="Compute weighted Scoring", total=regression_results.shape[0]):
    _iterations = 0
    for _, row in regression_results.iterrows():

        # check whether we have a target
        if target is not None and row["x"] != target:
            continue

        # extract all the necessary information from the row
        x = row['x']
        y = row['y']
        corr_coeff = row['correlation']
        delay = row['correlation delay']
        window_size = row['window size']
        alpha = row['alpha']
        beta = row['beta']

        # check whether the correlation coefficient is too small
        if corr_coeff < min_correlation:
            continue
        assert beta >= 0, f'Something is fishy. A positive correlation {corr_coeff} but a negative beta {beta}.'

        # compute the inverse alpha and beta for the weights of the other signals
        # y = alpha + beta*x
        # x = -alpha/beta + (1/beta)*y
        inverse_alpha = -alpha/beta
        inverse_beta = 1/beta

        # get the signals from the loaded signals
        signal1_df = scores[x]
        signal2_df = scores[y]

        # get the signals for the specified window length
        signal1 = signal1_df.get_group(window_size)[["value"]]
        signal2 = signal2_df.get_group(window_size)[["value"]]

        # check that the time stamps are regularly sampled (as we dropped the nan values)
        assert signal1.index.to_series().diff().iloc[1:].nunique() == 1, "Signal1 is irregular."
        assert signal2.index.to_series().diff().iloc[1:].nunique() == 1, "Signal2 is irregular."

        # merge the signals
        signals = pd.merge(signal1, signal2, how="inner", on="timestamp")
        index = signals.index.to_numpy()

        # get the numpy values from the signals
        signals = signals.to_numpy()
        signal1 = signals[:, 0]
        signal2 = signals[:, 1]

        # shift the signals for the maximum correlation
        if delay > 0:
            signal1 = signal1[delay:]
            signal2 = signal2[:-delay]
            index = index[delay:]
        elif delay < 0:
            signal1 = signal1[:delay]
            signal2 = signal2[-delay:]
            index = index[-delay:]

        # compute the squared and weighted residuals
        signal1_residuals = np.square(signal1-(signal2*inverse_beta + inverse_alpha)) * (corr_coeff**2)
        signal2_residuals = np.square(signal2-(signal1*beta + alpha)) * (corr_coeff**2)

        # create a new series to save it into the dataframe
        # TODO: What about ending and starting timestamps, where we have fewer values per timestamp?
        sum_scores(x, signal1_residuals, index, residual_results)
        sum_scores(y, signal2_residuals, index, residual_results)

        # update the iterations
        _iterations += 1

    # make the overall dataframe
    result_df = pd.concat(residual_results, axis=1)
    # logger.info(f"[{__name__}] Computing the residuals took {time.perf_counter() - start:0.2f} s and {_iterations} iters. {coming_from=}.")
    return result_df


def register_residual_fn(fn: typing.Callable):
    _RESIDUAL_REGISTER[fn.__name__] = fn
    return fn


def sum_scores(name, residuals: np.ndarray, index: np.ndarray, saving_dict: dict[str: pd.Series]):
    if name in saving_dict:
        saving_dict[name] = saving_dict[name].add(pd.Series(data=residuals, index=index), fill_value=0)
    else:
        saving_dict[name] = pd.Series(data=residuals, index=index)


@register_residual_fn
def standard_anomaly_scoring(residuals: pd.Series):
    """
    The standard anomaly scoring function also used in the project.
    :param residuals:
    :return:
    """
    # residuals = (residuals - residuals.min()) / (residuals.max() - residuals.min())
    residuals = (residuals - residuals.median()) / (residuals.quantile(0.75) - residuals.quantile(0.25))
    return residuals.max()


@register_residual_fn
def integral_anomaly_scoring(residuals: pd.Series):
    """
    The standard anomaly scoring function also used in the project.
    :param residuals:
    :return:
    """
    # residuals = (residuals - residuals.min()) / (residuals.max() - residuals.min())
    residuals = (residuals - residuals.median()) / (residuals.quantile(0.75) - residuals.quantile(0.25))
    return residuals.sum()


def get_registered_scoring_functions():
    return _RESIDUAL_REGISTER

def get_standard_scoring_functions():
    assert standard_anomaly_scoring.__name__ in _RESIDUAL_REGISTER
    return standard_anomaly_scoring.__name__, standard_anomaly_scoring