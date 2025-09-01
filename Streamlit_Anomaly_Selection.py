import os

import matplotlib.pyplot as plt
from tqdm import tqdm

import streamlit as st
import pandas as pd
import processing.anomaly_score as pas
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import seaborn as sns
import plotly.express as px


@st.cache_data
def get_anomaly_data(score_path: str) -> (pd.DataFrame, dict[str: pd.DataFrame]):

    # get the regression results
    # regression_path = os.path.join(score_path, 'results_pairwise_regression_spearman_ws-10_90_1.csv')
    regression_path = os.path.join(score_path, 'results_pairwise_regression.csv')
    unfolded_regression_results = pas.unfold_regression_results(pd.read_csv(regression_path, header=0, index_col=0))

    # get the top_n regression results
    top_n_regression_results = pas.find_top_n(unfolded_regression_results, 3)
    top_n_grouped = top_n_regression_results.groupby('y')

    # load the anomaly score
    anomaly_scores, residual_scores = pas.compute_anomaly_scores(top_n_regression_results, score_path,
                                                                 False, False)
    anomaly_scores = anomaly_scores.sort_values('score', ascending=False)

    # build the dataframe we want to create
    data_collection = {'Rank': [],
                       'Anomaly Score': [],
                       'Signal Name': [],
                       'Block': [],
                       'Turbine': [],
                       'Component': [],
                       'Measurement': [],
                       'Neighbors': [],
                       'Neighbor Names': [],
                       'Largest Correlation': [],
                       'Smallest Correlation': []
                       }
    iterator = tqdm(anomaly_scores.iterrows(), desc='Make Dataframe', total=anomaly_scores.shape[0])
    for idx, (name, anomaly_score) in enumerate(iterator, 1):

        # append the position
        data_collection['Rank'].append(idx)

        # append the name and the corresponding anomaly score of the signal
        data_collection['Anomaly Score'].append(anomaly_score['score'])
        data_collection['Signal Name'].append(name)

        # collect the information from the KKS name
        data_collection['Block'].append(name[1:2])
        data_collection['Turbine'].append(name[2:3])
        data_collection['Component'].append(name[3:8])
        data_collection['Measurement'].append(name[8:10])

        # go through all neighbors and keep the correlation information
        neighbors = [f'{row["x"]}: {row["correlation"]:0.3f}' for _, row in top_n_grouped.get_group(name).iterrows()]
        neighbors_names = [row["x"] for _, row in top_n_grouped.get_group(name).iterrows()]
        data_collection['Neighbors'].append(neighbors)
        data_collection['Neighbor Names'].append(neighbors_names)
        data_collection['Largest Correlation'].append(top_n_grouped.get_group(name)['correlation'].max())
        data_collection['Smallest Correlation'].append(top_n_grouped.get_group(name)['correlation'].min())
    data_collection = pd.DataFrame(data_collection)
    return data_collection, top_n_grouped, residual_scores


@st.cache_data
def get_time_series_data(score_path: str):
    # load the normalized signals
    signal_path = os.path.join(*os.path.split(score_path)[:-1], 'df_raw_normalized.parquet')
    normalized_signals = pd.read_parquet(signal_path)
    normalized_signals = normalized_signals.sort_index(ascending=True)
    normalized_signals = {name: data for name, data in normalized_signals.groupby('sensor', sort=False)}

    # go through the dict ones more and attach the original name
    for name, data in normalized_signals.items():
        data['sensor'] = name
    return normalized_signals


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns
    https://blog.streamlit.io/auto-generate-a-dataframe-filtering-ui-in-streamlit-with-filter_dataframe/

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if isinstance(df[column], pd.CategoricalDtype) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}, e.g. (?:CT|CP) for pressure and temperature",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df


column_configuration = {
    "Signal Name": st.column_config.TextColumn(
        "Name", help="The name of the user", max_chars=100, width=None
    ),
    "Anomaly Score": st.column_config.NumberColumn(format='%.4e'),
    "Neighbor Names": None,
    "Neighbors": st.column_config.ListColumn(width=None),
}

st.set_page_config(layout="wide")
st.header("All members")
# __score_path = r"C:\Users\Lucas\Data\CP_Anomaly\Albur11_Heatmap_Regression\scores_linear_ESST_synth_10-90_1"
# __score_path = r"C:\Users\Lucas\Data\CP_Anomaly\Albur11_Heatmap_Regression\scores_linear_ESST_synth"
# __score_path = r"C:\Users\Lucas\Data\CP_Anomaly\Kitby_Gehäusetemperature\scores_linear_ESST_synth"
__score_path = r"C:\Users\Lucas\Data\CP_Anomaly\KEAD_Unknown\scores_linear_ESST_synth"
__df, _, __residual_scores = get_anomaly_data(score_path=__score_path)
__df = filter_dataframe(__df)


event = st.dataframe(
    __df,
    column_config=column_configuration,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
)

# check whether we selected something
selected_signal = event.selection.rows
if selected_signal:

    # get the info from the selection
    selected_signal = selected_signal.pop()
    __neighbors = __df.iloc[selected_signal]['Neighbor Names'][::-1]
    signal_name = __df.iloc[selected_signal]['Signal Name']

    # attach the sensor itself so it gets also plotted
    __neighbors.append(signal_name)

    # get the position of the maximum score
    posi = __residual_scores[signal_name].idxmax()

    # make some writes
    st.header("Currently Selected data")
    st.write(signal_name)

    # create the score plot
    fig = px.line(__residual_scores[signal_name])
    fig.add_vline(posi, layer='below')
    fig.update_layout(xaxis_title="Date", yaxis_title="Residual Value (Unexpected Change)")
    st.plotly_chart(fig)

    # make a slider that selects how many samples to skip
    skip_samples = st.slider('How many samples you want to skip (higher ist faster)?', min_value=1, max_value=100, value=10)

    # get the signals for the raw signal plot
    __normalized_signals = get_time_series_data(score_path=__score_path)
    __new_df = pd.concat((__normalized_signals[neighbor][['sensor', 'normalized value']].iloc[::skip_samples] for neighbor in __neighbors))

    # make the raw signal plot
    st.write(signal_name)
    fig = px.line(__new_df, color='sensor')
    fig.add_vline(posi, layer='below')
    st.plotly_chart(fig)
