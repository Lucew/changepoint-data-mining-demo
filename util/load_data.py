import os
import typing
import logging
import shutil
import time
import json

import pandas as pd
import pandas.api.typing as pdtypes
from tqdm import tqdm

import util.cache_registry as ucache
from GLOBALS import *


# get the logger
logger = logging.getLogger("frontend-logger")


@ucache.lru_cache(1)
def load_data(folder_path: str, mock_signals: bool = False, reduce_count: typing.Optional[int] = None) -> tuple[dict[str: pdtypes.DataFrameGroupBy], dict[str: pd.DataFrame], tuple[int], typing.Optional[pd.DataFrame], typing.Optional[pd.DataFrame], pdtypes.DataFrameGroupBy, pd.DataFrame]:
    start = time.perf_counter()

    # check whether we wanted to reduce the count
    if reduce_count is None:
        reduce_count = MAX_SIGNALS

    # read the configuration
    config = json.load(open(os.path.join(folder_path, "params.json")))
    config_window_sizes = tuple(sorted(config['window_sizes']))
    raw_signals_path = os.path.join(folder_path, config["input"]["filename"])

    # get all the different files from the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    score_files = [filename for filename in files if filename.endswith('.parquet') and (filename.startswith('@') or (filename[:2].isnumeric() and filename[5:7].isnumeric()))]
    signal_files = [filename for filename in files if filename.endswith('.parquet') and filename.startswith('resamp')]

    score_files_set = None
    if reduce_count is not None and 0 < reduce_count < len(score_files):

        # get the score files we want to use
        score_files = score_files[:reduce_count]

        # get the names of the signals we have
        score_files_set = set(os.path.splitext(os.path.split(file)[-1])[0] for file in score_files)

        # file the signal files
        signal_files = [file for file in signal_files if os.path.splitext(os.path.split(file)[-1])[0].split('_')[1] in score_files_set]

    # load the scoring dataframes into dictionary
    scores = {os.path.splitext(filename)[0]: pd.read_parquet(os.path.join(folder_path, filename)).drop(columns="signal") for filename in tqdm(score_files, desc='Loading the Scores')}

    # go through the scores signals and restrict to maximum starting point and minimum ending point
    max_start = max(score["timestamp"].min() for score in scores.values())
    min_end = min(score["timestamp"].max() for score in scores.values())

    # get the window sizes
    window_sizes: dict[str: pd.DataFrame] = {tuple(sorted(int(ele) for ele in df["window"].unique())) for df in scores.values()}
    assert len(window_sizes) == 1, "Some Signals have different window sizes."
    window_sizes = window_sizes.pop()
    assert window_sizes == config_window_sizes, "Some Signals have different window sizes than specified in config."

    # restrict the scores and check that they have the same starting and end points
    # scores = {name: score.loc[score["timestamp"].between(max_start, min_end)] for name, score in scores.items()}

    # group the scores by the window sizes and set the timestamp as index
    scores = {name: score.set_index("timestamp").sort_index().groupby("window", sort=False) for name, score in tqdm(scores.items(), desc='Grouping Scores')}

    # load the anomaly scores if they are available
    anomaly_score_path = os.path.join(folder_path, 'anomaly_scores.parquet')
    anomaly_scores = None
    if os.path.exists(anomaly_score_path):
        anomaly_scores = pd.read_parquet(anomaly_score_path)

    # load the distances if they are available
    distances_path = os.path.join(folder_path, 'distances.csv')
    distances = None
    if os.path.exists(distances_path):
        distances = pd.read_csv(distances_path)

    # sort out the distances
    if score_files_set is not None:
        distances = distances[distances['x'].isin(score_files_set) & distances['y'].isin(score_files_set)]

    # load the signal correlations if they are available
    raw_signal_correlations_path = os.path.join(folder_path, 'signal_correlation.parquet')
    raw_signal_correlations = None
    if os.path.exists(raw_signal_correlations_path):
        raw_signal_correlations = pd.read_parquet(raw_signal_correlations_path)

    # get the raw signals (currently just mock data from the scores)
    if mock_signals:
        signals = {}
        indexer = None
        for name, score in scores.items():

            # get the window sizes
            print(score["window"].unique())
            ws = min(score["window"].unique())

            # get only the values where the minimum window size is there and make it as the signal
            restricted_df = score.get_group(ws)["value"]
            indexer = score.get_group(ws)["timestamp"]
            signals[name] = restricted_df.to_numpy()
        signals = pd.DataFrame(signals)
        signals.index = indexer

        # get the raw signals (currently just mock data from the scores)
        raw_signals = signals.melt(value_name="value", var_name="sensor", ignore_index=False)
    else:

        # load the raw signals into memory
        raw_signals = pd.read_parquet(raw_signals_path)

        # drop the signals we do not want to have
        if score_files_set is not None:
            raw_signals = raw_signals[raw_signals['sensor'].isin(score_files_set)]

        # load the resampled data
        signals: dict[str: pd.DataFrame]
        signals = {os.path.splitext(filename)[0].split("_")[1]: pd.read_parquet(os.path.join(folder_path, filename))
                   for filename in tqdm(signal_files, desc='Loading the Signals')}

    # make the zscore normalization by group
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#transformation
    logger.info(f"[{__name__}] Normalizing the signals (@{time.perf_counter() - start:0.2f} s).")
    raw_signals.loc[:, "normalized value"] = raw_signals.groupby("sensor", sort=False)[["value"]].transform(lambda x: (x - x.min()) / (xminmax if (xminmax := x.max()-x.min()) != 0 else 1))

    # sort the signals and create the groups
    raw_signals = raw_signals.sort_index(ascending=True)
    raw_signals_grouped = raw_signals.groupby("sensor", sort=False)

    logger.info(f"[{__name__}] Loaded data files into cache from disk {time.perf_counter() - start:0.2f} s.")
    return scores, signals, window_sizes, anomaly_scores, distances, raw_signals_grouped, raw_signal_correlations


def folder_size_bytes(root_path: str) -> int:
    """
    Walks the directory and sums file sizes. Follows regular files,
    skips broken links; ignores permission errors.
    """
    if not os.path.isdir(root_path):
        return 0
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root_path):
        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            try:
                if os.path.islink(fp):
                    # count the link file itself, not the target
                    total += os.lstat(fp).st_size
                else:
                    total += os.path.getsize(fp)
            except Exception:
                # best-effort; skip unreadables
                continue
    return total

def format_bytes(num: int) -> str:
    # human-friendly bytes
    for unit in ["bytes", "KB", "MB", "GB", "TB"]:
        if num < 1024 or unit == "TB":
            return f"{num:.0f} {unit}" if unit == "bytes" else f"{num:.2f} {unit}"
        num /= 1024.0


def delete_all_files_in_root(root_path: str) -> int:
    if not os.path.isdir(root_path):
        return 0
    removed = 0
    for name in os.listdir(root_path):
        p = os.path.join(root_path, name)
        try:
            if os.path.isdir(p) and not os.path.islink(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
            removed += 1
        except Exception as e:
            logger.error(f"Failed removing {p}: {e}")
    return removed


if __name__ == '__main__':
    # load_data(r"C:\Users\lucas\Downloads\2025-07-23T095510Z_Keadby")
    logging.basicConfig(level=logging.DEBUG)
    load_data(r"C:\Users\lucas\PycharmProjects\changepoint-analysis-frontends\tmp-data-folder\download_zip\2025-09-10T143441Z_df_raw_lucas")