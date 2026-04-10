import time
import logging
import inspect
import os

import numpy as np
import pandas as pd

import sklearn.manifold as skmani

import util.cache_registry as ucache
import util.prepocessing as prep
import util.load_data as utl
import util.process_kks as ukks
from GLOBALS import *


@ucache.cache
def get_random_seed():
    return 3


def get_default_corr_threshold():
    return 0.6


@ucache.lru_cache(maxsize=CACHE_SIZE)
def filter_regression_results(session_id: str, folder_name: str, correlation_threshold: float = None) -> (pd.DataFrame, pd.DataFrame, float):
    start = time.perf_counter()
    logger = logging.getLogger("frontend-logger")

    # get the preprocessed regression results
    regression_results, _, complete_max_correlation = prep.preprocess_regression_results(session_id, folder_name)

    # create default correlation threshold if not given
    if correlation_threshold is None:
        correlation_threshold = get_default_corr_threshold()

    # find all the signal names that have larger correlation than the threshold
    mask = complete_max_correlation["correlation"] > correlation_threshold
    max_correlated_signal_tags = set(complete_max_correlation[mask].index)

    # only keep the signals that have maximum correlation
    filtered_regression_results = regression_results.loc[regression_results["x"].isin(max_correlated_signal_tags) &
                                                         regression_results["y"].isin(max_correlated_signal_tags)]

    # fill the distance matrix for the signals using pivoting of the regression results
    # https://stackoverflow.com/questions/71671054/convert-pandas-dataframe-to-distance-matrix
    filtered_distance_matrix = filtered_regression_results.pivot(index='x', columns='y', values='distance').fillna(0)

    # return the filtered regression results
    logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Filtered regression results in {time.perf_counter() - start:0.2f} s.")
    return filtered_regression_results, filtered_distance_matrix, correlation_threshold


@ucache.lru_cache(maxsize=CACHE_SIZE)
def window_size_correlation(session_id: str, folder_name: str, window_size: int) -> (pd.DataFrame, pd.DataFrame, float):
    start = time.perf_counter()
    logger = logging.getLogger("frontend-logger")

    # load the raw data
    scores, _, _, _, _, _, _ = utl.load_data(os.path.join(DATA_FOLDER, session_id, folder_name))

    # put the scores into a dataframe
    # get the score for the defined window size
    scores = {name: df.get_group(window_size)[["value"]].rename(columns={'value': name}) for name, df in scores.items()}
    score_df = pd.concat(scores.values(), axis=1)
    score_corr = score_df.corr()
    score_corr = score_corr.abs()

    # get the columns where there is nan
    not_null_columns = score_corr.columns[~score_corr.isnull().any(axis=0)]
    score_corr[score_corr.isnull()] = 0
    # score_corr = score_corr.loc[not_null_columns, not_null_columns]

    # transform correlation into distance
    score_corr = 10_000 - score_corr * 1000

    # return the filtered regression results
    logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Correlated scores in {time.perf_counter() - start:0.2f} s.")
    return score_corr


def get_default_perplexity(sensor_number: int) -> int:
    return int(np.sqrt(sensor_number))


# we have to do twice the cache size as this is used for two pages within our application
@ucache.lru_cache(maxsize=CACHE_SIZE*2)
def create_tsne(session_id: str, folder_name: str, perplexity: int = None, correlation_threshold: float = None, window_size: int = None) -> (pd.DataFrame, int, float):

    # get the logger and log the attempt
    logger = logging.getLogger("frontend-logger")
    logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Requested TSNE ({perplexity=}, {correlation_threshold=}, {window_size=}).")

    # make the timing
    started = time.perf_counter()

    # get the filtered regression results
    if window_size is None:
        _, distance_matrix, correlation_threshold = filter_regression_results(session_id, folder_name, correlation_threshold)
    else:
        distance_matrix = window_size_correlation(session_id, folder_name, window_size)

    # load the anomaly scores
    _, _, _, anomaly_scores, _, _, _ = utl.load_data(os.path.join(DATA_FOLDER, session_id, folder_name))

    # load the maximum threshold information and signal names
    _, _, max_correlation = prep.preprocess_regression_results(session_id, folder_name)

    # compute default perplexity
    # print("Parameters", session_id, folder_name, perplexity, correlation_threshold)
    if perplexity is None:
        perplexity = get_default_perplexity(distance_matrix.shape[0])
    if perplexity >= distance_matrix.shape[0]:
        logger.warning(f"[{__name__}][{inspect.stack()[0][3]}] Perplexity larger than distance matrix. Set to maximum number: {distance_matrix.shape[0]=}.")
        perplexity = distance_matrix.shape[0]

    # compute the TSNE representation of the data
    tsne = skmani.TSNE(metric='precomputed', init='random', perplexity=perplexity, random_state=get_random_seed())
    transf = tsne.fit_transform(distance_matrix)

    # compute the TSNE representation of the data
    tsne3d = skmani.TSNE(n_components=3, metric='precomputed', init='random', perplexity=perplexity, random_state=get_random_seed())
    transf3d = tsne3d.fit_transform(distance_matrix)

    # update the kks tag processing
    info_dict = ukks.get_info_from_list(distance_matrix.columns, unique=False)
    blocks = info_dict["block"]
    turbines = info_dict["turbine"]
    components = info_dict["component"]
    measurements = info_dict["measurement"]
    types = info_dict["type"]

    # create the dataframe from the information
    # make the dataframe that we want to have
    bokeh_df = pd.DataFrame({'x': transf[:, 0],
                             'y': transf[:, 1],
                             'x3d': transf3d[:, 0],
                             'y3d': transf3d[:, 1],
                             'z3d': transf3d[:, 2],
                             'original': list(distance_matrix.columns),
                             'Max. Corr.': [f"{ele:0.3f} " for ele in
                                            list(max_correlation.loc[distance_matrix.columns]["correlation"])],
                             'Anomaly Score': anomaly_scores.loc[distance_matrix.columns, 'score'] if anomaly_scores is not None else [None]*transf.shape[0],
                             'block': blocks,
                             'block_turbine': [f"{block}{turbine}" for block, turbine in zip(blocks, turbines)],
                             'turbine': [f'Steam [{turbine}]' if turbine == '0'
                                         else f'Gas [{turbine}]'
                                         for turbine in turbines],
                             'component': components,
                             'measurement': measurements
                             })

    # TODO TSNE perplexity as component group size?
    # print((bokeh_df.groupby("component")["x"].count()).median())

    # recompute the anomaly scores
    rec_anomaly = [0]*transf.shape[0]
    if anomaly_scores is not None:
        rec_anomaly = bokeh_df.loc[:, 'Anomaly Score'].to_numpy().copy()
        rec_anomaly[np.isnan(rec_anomaly)] = 0
        zeroless_mask = rec_anomaly != 0
        zero_mask = ~zeroless_mask
        rec_zeroless = np.log(rec_anomaly[zeroless_mask])
        rec_anomaly[zero_mask] = rec_zeroless.min()
        rec_anomaly[zeroless_mask] = rec_zeroless
        rec_anomaly = (rec_anomaly - rec_anomaly.min()) / (rec_anomaly.max() - rec_anomaly.min())
    bokeh_df['opac'] = rec_anomaly

    logger.info(f"[{__name__}][{inspect.stack()[0][3]}] Created TSNE ({time.perf_counter()-started:0.2f} s).")
    return bokeh_df, perplexity, correlation_threshold