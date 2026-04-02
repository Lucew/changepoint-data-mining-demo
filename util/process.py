import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util.signal_utils import get_anomaly_stamps
import warnings


METHODS = {'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'barycentric', 'polynomial', 'ffill', 'bfill', 'mean',
           'median'}


def normalization(signal: np.ndarray):
    min_val = np.min(signal)
    max_val = np.max(signal)
    signal -= min_val
    if min_val != max_val:
        signal /= (max_val-min_val)
    return signal


def aggregate_ts(df: pd.DataFrame, sample_rate: str = "1min", method: str = "linear",
                 emulate_dead_band: bool = True) -> pd.DataFrame:
    """
    This function computes a time weighted average for resampling of a signal.

    :param df: the time series data with a time index
    :param sample_rate: the target sampling rate
    :param method: the aggregation method to compute the weighted mean
    :param emulate_dead_band: whether to create synthetic time stamps to emulate the dead banding
    :return: the resampled dataframe
    """

    # check whether we have a named index, otherwise name it
    if df.index.name is None:
        index_name = "datetime"
        df.index.name = index_name
    else:
        index_name = df.index.name

    # sort the signal timewise (for our application, we have no guarantee that the samples are sorted)
    df.sort_index(inplace=True)

    # get all the signals we have at the same time stamps
    signals = df.columns

    # check that there is only one signals
    assert len(signals) == 1, f'There are more than one value column in the dataframe: {signals}.'
    signal = signals[0]

    # check if we want to introduce synthetic time stamps to emulate the dead band
    if emulate_dead_band:
        # make an educated guess about the dead band
        df.loc[:, "dx"] = df[signal].diff()
        db = df["dx"].abs().min()
        if db <= 0:
            warnings.warn(f'The deadband for {signal} is not as expected: {db}.')

        # get the minimum time difference from the data
        min_time_diff = df.index.diff().min()-pd.Timedelta('1ms')

        # insert synthetic time stamp shortly before the real one and insert the one changed by the dead band
        df = pd.concat([df, pd.DataFrame(index=df.index - min_time_diff, data=None, columns=[signal, "dx"],
                                         dtype=df[signal].dtype)],
                       axis=0).sort_index()
        df["synthetic"] = df[signal].shift(1) + ((df["dx"].shift(-1) > 0).astype('int') - 0.5) * 2 * db
        df[signal] = df[signal].fillna(df["synthetic"])
        df = df[[signal]].dropna()

    # create the new sampling frequency time stamps
    sampling_frequency_index = pd.date_range(
        start=df.index.min().ceil(freq=sample_rate),
        end=df.index.max().ceil(freq=sample_rate),
        freq=sample_rate,
        name=index_name,
    )

    # insert the time stamps into the other dataframe and fill with NaN
    df = df.combine_first(pd.DataFrame(index=sampling_frequency_index)).sort_index()

    # fill the now missing time stamps using the chosen method of interpolation (this should also make sure that
    # there always is a value per time stamp)
    if method == "ffill":
        df = df.ffill()
    elif method == "linear":
        df = df.interpolate("time")
    else:
        raise ValueError(f"Method {method} not available for aggregation.")

    # compute the delta time stamps (can never be zero as the time is an index and therefore unique)
    df["dt"] = (pd.to_numeric(df.index.to_series()).diff()).shift(-1)

    # go over all the signals and make the aggregation
    if method == "ffill":
        df[signal] = df[signal] * df["dt"]  # area of rectangle
    elif method == "linear":
        df[signal] = df["dt"] * (df[signal] + df[signal].shift(-1)) / 2  # area of trapezoid
    else:
        raise ValueError(f"Method {method} not available for aggregation.")

    # resample the signal to only keep the values that we need and sum up the areas
    df = df.resample(sample_rate, label="left").sum()

    # go over the signals and divide the area by the summed time, so we have a good
    df[signal] = df[signal] / df["dt"]

    # get rid of the delta time column
    df = df.drop(["dt"], axis=1)

    # find the indices of all NaN values and check they are only at the beginning, and at the end, so we do not end
    # up with non-regular time series if we drop NaN values
    nandx = np.where(df.isna())[0]
    assert len(nandx) < 3 or np.all(np.diff(nandx) < 2), f"There are some values missing in between: {nandx[1:]}."

    # drop the NaN values
    df = df.dropna()
    return df


def check_aggregation(raw_df: pd.DataFrame, aggregated_df: pd.DataFrame, name: str,
                      annotations: pd.DataFrame = None) -> None:

    # get the annotations if necessary
    annotation_start = None
    annotation_end = None
    if annotations is not None:
        annotation_start, annotation_end = get_anomaly_stamps(annotations, name)

    # plot the raw data as dots
    plt.plot(raw_df.index, raw_df["value"], "kx")
    plt.plot(aggregated_df.index, aggregated_df["value"], "o-")
    plt.title(f"Plot for {name}")
    if annotation_start is not None and annotation_end is not None:

        # check whether the time series interval and annotation interval overlap
        overlaps = annotation_start <= raw_df.index[-1] and raw_df.index[0] <= annotation_end

        # plot the annotation interval if it overlaps
        if overlaps:
            plt.axvspan(max(annotation_start, raw_df.index[0]), min(annotation_end, raw_df.index[-1]), facecolor='b',
                        alpha=0.2)
    plt.show()


def main():

    # read annotation and data
    from load import read_annotation, read_data

    annotations = read_annotation()
    data = read_data()

    # get example data from the loaded data
    name = list(data.keys())[2]
    data = data[name]

    # try different aggregations
    aggregated_df = aggregate_ts(data, method="linear", sample_rate="20min")

    # limit the output of the time series
    check_aggregation(data, aggregated_df.loc[:data.index[-1]], name, annotations)


if __name__ == '__main__':
    main()
