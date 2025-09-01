import os
import typing
import functools
import logging
import shutil

import pandas as pd
import pandas.api.typing as pdtypes

# get the logger
logger = logging.getLogger("frontend-logger")


@functools.lru_cache(1)
def load_data(folder_path: str) -> tuple[dict[str: pd.DataFrame], pd.DataFrame, tuple[int], typing.Optional[pd.DataFrame], typing.Optional[pd.DataFrame], pdtypes.DataFrameGroupBy, pd.DataFrame]:

    # get all the different files from the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # load the scoring dataframes into dictionary
    scores = {os.path.splitext(filename)[0]: pd.read_parquet(os.path.join(folder_path, filename)) for filename in files
              if filename.endswith('.parquet') and filename.startswith('@')}

    # go through the scores signals and restrict to maximum starting point and minimum ending point
    max_start = max(score["timestamp"].min() for score in scores.values())
    min_end = min(score["timestamp"].max() for score in scores.values())

    # get the window sizes
    window_sizes = {tuple(df["window"].unique()) for df in scores.values()}
    assert len(window_sizes) == 1, "Some Signals have different window sizes."
    window_sizes = window_sizes.pop()


    # restrict the scores and check that they have the same starting and end points
    scores = {name: score.loc[score["timestamp"].between(max_start, min_end)] for name, score in scores.items()}

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

    # load the signal correlations if they are available
    raw_signal_correlations_path = os.path.join(folder_path, 'signal_correlation.parquet')
    raw_signal_correlations = None
    if os.path.exists(raw_signal_correlations_path):
        raw_signal_correlations = pd.read_parquet(raw_signal_correlations_path)

    # get the raw signals (currently just mock data from the scores)
    signals = {}
    indexer = None
    for name, score in scores.items():

        # get the window sizes
        ws = min(score["window"].unique())

        # get only the values where the minimum window size is there and make it as the signal
        restricted_df = score.loc[score["window"]==ws, "value"]
        indexer = score.loc[score["window"]==ws, "timestamp"]
        signals[name] = restricted_df.to_numpy()
    signals = pd.DataFrame(signals)
    signals.index = indexer

    # get the raw signals (currently just mock data from the scores)
    raw_signals = signals.melt(value_name="value", var_name="sensor", ignore_index=False)

    # make the zscore normalization by group
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#transformation
    raw_signals.loc[:, "normalized value"] = raw_signals.groupby("sensor").transform(lambda x: (x - x.min()) / (xminmax if (xminmax := x.max()-x.min()) != 0 else 1))["value"]

    # sort the signals and create the groups
    raw_signals = raw_signals.sort_index(ascending=True)
    raw_signals_grouped = raw_signals.groupby("sensor")

    logger.info("Loaded data files into cache.")
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
    load_data(r"C:\Users\lucas\Downloads\2025-07-23T095510Z_Keadby")