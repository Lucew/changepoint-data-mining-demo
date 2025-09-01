import pandas as pd
from tqdm import tqdm
import numpy as np


def compute_weighted_residual_norm(regression_results: pd.DataFrame, signal_list: list[str], scores: dict[str: pd.DataFrame],
                                   min_correlation: float = 0.05):

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
    for _, row in tqdm(regression_results.iterrows(), desc="Compute weighted Scoring", total=regression_results.shape[0]):

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
        signal1 = signal1_df[signal1_df["window"] == window_size][["timestamp", "value"]]
        signal1.set_index("timestamp", inplace=True)
        signal2 = signal2_df[signal2_df["window"] == window_size][["timestamp", "value"]]
        signal2.set_index("timestamp", inplace=True)

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

    # make the overall dataframe
    result_df = pd.concat(residual_results, axis=1)
    return result_df


def sum_scores(name, residuals: np.ndarray, index: np.ndarray, saving_dict: dict[str: pd.Series]):
    if name in saving_dict:
        saving_dict[name] = saving_dict[name].add(pd.Series(data=residuals, index=index), fill_value=0)
    else:
        saving_dict[name] = pd.Series(data=residuals, index=index)