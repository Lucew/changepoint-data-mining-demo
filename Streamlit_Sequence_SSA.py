import pandas as pd
import numpy as np
from Dash_Sequence_Heatmap import load_data
import streamlit as st
import altair as alt
from glob import glob
from pathlib import Path
from sklearn.manifold import TSNE


def normalization(df: pd.DataFrame) -> pd.DataFrame:
    mini = df['value'].min()
    maxi = df['value'].max()
    if maxi > mini:
        df['value'] = (df["value"] - mini) / (maxi - mini)
    return df


def create_comparison_vector(left_eigenvectors, eigenvalues, right_eigenvectors, ranged, eigenidx):
    # make the index array to get the correct eigenvectors
    idx = list(range(int(ranged[0])-1, int(ranged[1])))

    # make the indices from the input
    eigenidx = [int(ele.split(" ")[0]) for ele in eigenidx]

    # compute the sequences from the information
    sequences = (left_eigenvectors[:, eigenidx] @ np.diag(eigenvalues[eigenidx])
                 @ right_eigenvectors[eigenidx, :][:, idx])
    sequences = np.mean(sequences, axis=1)
    return sequences


def get_folders():

    # find all the folders in the repository
    folders = [Path(ele).parts[-1] for ele in glob('data/*/')]
    return folders


if __name__ == '__main__':

    @st.cache_data
    def process_data(folder_name: str, number_eigenvectors: int = 6) \
            -> (pd.DataFrame, np.ndarray, np.ndarray, np.ndarray):
        sequence_matrices, sequence_indices, eigensequences, timeframe = load_data(number_eigenvectors, folder_name)
        return sequence_matrices, sequence_indices, eigensequences, timeframe

    @st.cache_data
    def get_names(folder_name: str):
        sequence_matrices, _, _, _ = process_data(folder_name=folder_name)
        return list(sequence_matrices.keys())

    def select_data(signal_name: str, folder_name: str):
        sequence_matrices, sequence_indices, eigensequences, timeframe = process_data(folder_name=folder_name)
        sequence_matrices = sequence_matrices[signal_name]
        sequence_indices = sequence_indices[signal_name]
        left_eigenvectors, eigenvalues, right_eigenvectors = eigensequences[signal_name]
        sequence_dataframe = pd.DataFrame(sequence_matrices,
                                          columns=[str(ele) for ele in range(sequence_matrices.shape[1])])
        return sequence_dataframe, sequence_indices, left_eigenvectors, eigenvalues, right_eigenvectors, timeframe

    # do some streamlit related elements and code ----------------------------------------------------------------------
    st.set_page_config(layout="wide")

    # make some columns
    col1, col2 = st.columns(2)

    # create the sidebar
    st.sidebar.title('Singular Spectrum Analyzer')

    # get the folders that are available and select one
    fs = get_folders()
    fs_name = st.sidebar.selectbox("Select a data folder", options=fs)

    # load the data accordingly
    get_folders()
    signal_names = get_names(folder_name=fs_name)

    # select the startups or shutdowns
    source = st.sidebar.selectbox("Select an input", options=signal_names)

    # sort out the data we need
    dataframe, seq_idces, left, values, right, time = select_data(source, fs_name)

    # selection for eigenvectors
    eigenvector_choices = [f"{idx} [{values[idx]/sum(values):.2%}]" for idx in range(right.shape[0])]
    eigenvector2num = {f"{idx} [{values[idx]/sum(values):.2%}]": idx for idx in range(right.shape[0])}
    block_selector = st.sidebar.multiselect(
        "Which eigenvectors do you want to plot?",
        options=eigenvector_choices,
        default=eigenvector_choices[0]
    )

    sliding_mean = st.sidebar.slider("Sliding Mean Size", min_value=1, max_value=right.shape[1]//2, value=1, step=2)
    right_smoothed = np.apply_along_axis(lambda x: np.convolve(x, np.ones(sliding_mean)/sliding_mean, mode='valid'), axis=1, arr=right)

    # make a dataframe from the eigenvectors
    left_df = pd.DataFrame(data=left, index=[ele + sliding_mean//2 for ele in range(left.shape[0])], columns=eigenvector_choices)
    left_df = left_df[block_selector]
    left_df["x"] = left_df.index
    right_df = pd.DataFrame(data=right_smoothed.T, index=[ele + sliding_mean//2 for ele in range(right_smoothed.shape[1])], columns=eigenvector_choices)
    right_df = right_df[block_selector]
    right_df["x"] = right_df.index

    # check whether we want to plot single signals or a range of signals
    plot_sequence = st.sidebar.selectbox("What do you want to plot?", options=("Single Real Sequences",
                                                                               "Merged Real Sequences"), index=1)

    # make the plots of the eigenvector and participation
    melted_left = left_df.melt('x')
    melted_right = right_df.melt('x')
    melted_right['Real Sequence ID'] = [seq_idces[ele] for ele in melted_right['x']]
    melted_right['Date'] = [(f'From {time.loc[ele, "from"]: %d-%m-%Y %H:%M:%S} to '
                             f'{time.loc[ele, "to"]: %d-%m-%Y %H:%M:%S}') for ele in melted_right['Real Sequence ID']]
    melted_left.rename(columns={"variable": "Eigenvector", "x": "Timestep"}, inplace=True)
    melted_right.rename(columns={"variable": "Participation", "x": "Sequence ID"}, inplace=True)
    with col1:
        title = "Eigensequences (Weighted Sum creates Sequences)"
        st.altair_chart(alt.Chart(melted_left, title=title).mark_line().encode(x="Timestep", y="value",
                                                                               color="Eigenvector").interactive(),
                        use_container_width=True)
    with col2:
        title = 'Participation of Eigensequences in real Sequences (individual weights per Sequences)'
        ttp = ['Participation', 'value', 'Sequence ID', 'Real Sequence ID', 'Date']
        st.altair_chart(alt.Chart(melted_right, title=title).mark_line().encode(x="Sequence ID", y="value",
                                                                                color="Participation",
                                                                                tooltip=ttp).interactive(),
                        use_container_width=True)

    if plot_sequence == "Merged Real Sequences":

        # the list of possible colors
        color_list = [
            "blue",
            "orange",
            "green",
            "red",
            "purple",
            "brown",
            "pink",
            "gray",
            "black",
            "yellow"
        ]

        # create a color selection
        numbers = list(range(1, 11))
        number_ranges = st.sidebar.selectbox("How many ranges do you want to specify?", options=numbers, index=1)

        # create the sliders
        sequence_selectors = [st.sidebar.select_slider(f"Which range do you want to plot for line {idx}?",
                                                       options=dataframe.columns,
                                                       value=(dataframe.columns[0], dataframe.columns[5]))
                              for idx in range(number_ranges)]

        # make a function that outputs the column names for a selected range
        def column_names(ranged: tuple[str, str]):
            return [str(ele) for ele in range(int(ranged[0]), int(ranged[1]))]

        # compute the mean vector for each of the ranges and melt it into the right format
        new_df = pd.DataFrame.from_dict({f"{idx}: {ranged[0]}-{ranged[1]}": dataframe[column_names(ranged)].mean(axis=1)
                                         for idx, ranged in enumerate(sequence_selectors)})
        new_df["x"] = new_df.index
        melted_df = new_df.melt("x")
        melted_df.rename(columns={"variable": "Sequences", "x": "Timestep"}, inplace=True)

        # create the chart of the mean sequences over the given area
        title = 'Real Sequences (mean over time window)'
        real_sequence_chart = alt.Chart(melted_df, title=title).mark_line().encode(x="Timestep", y="value",
                                                                                   color=alt.Color("Sequences", sort=None)
                                                                                   ).interactive()

        # compute the modeled sequences from the eigenvectors
        modeled_df = pd.DataFrame.from_dict({f"{idx}: {ranged[0]}-{ranged[1]}": create_comparison_vector(left, values,
                                                                                                         right, ranged,
                                                                                                         block_selector)
                                             for idx, ranged in enumerate(sequence_selectors)})
        modeled_df["x"] = modeled_df.index
        modeled_df = modeled_df.melt("x")
        modeled_df.rename(columns={"variable": "Sequences", "x": "Timestep"}, inplace=True)
        modeled_df = normalization(modeled_df)
        title = (f'Reconstructed Sequences (reconstruction of signals by the selected'
                 f' and weighted eigenvectors: [{", ".join(ele.split(" ")[0] for ele in block_selector)}])')
        modeled_sequence_chart = alt.Chart(modeled_df, title=title).mark_line().encode(x="Timestep", y="value",
                                                                                       color=alt.Color("Sequences",
                                                                                                       sort=None)
                                                                                       ).interactive()

        # combine both plots
        with col1:
            st.altair_chart(real_sequence_chart, use_container_width=True)
        with col1:
            st.altair_chart(modeled_sequence_chart, use_container_width=True)
    else:

        # selector which lines to plot
        sequence_selector = st.sidebar.multiselect(
            "Which sequence do you want to plot?",
            options=dataframe.columns,
            default=dataframe.columns[0]
        )
        if sequence_selector:
            # get the real sequences
            dataframe = dataframe[sequence_selector]
            dataframe["x"] = dataframe.index
            melted_data = dataframe.melt('x')
            melted_data.rename(columns={"variable": "Sequences", "x": "Timestep"}, inplace=True)
            title = 'Real Sequences'
            real_sequence_chart = alt.Chart(melted_data, title=title).mark_line().encode(x="Timestep", y="value",
                                                                                         color=alt.Color("Sequences",
                                                                                                         sort=None)
                                                                                         ).interactive()

            # compute the modeled sequences from the eigenvectors
            modeled_df = pd.DataFrame.from_dict({str(ranged): create_comparison_vector(left, values, right,
                                                                                       (ranged, ranged),
                                                                                       block_selector)
                                                 for ranged in sequence_selector})
            modeled_df["x"] = modeled_df.index
            modeled_df = modeled_df.melt("x")
            modeled_df.rename(columns={"variable": "Sequences", "x": "Timestep"}, inplace=True)
            title = (f'Reconstructed Sequences (reconstruction of signals by the selected'
                     f' and weighted eigenvectors: [{", ".join(ele.split(" ")[0] for ele in block_selector)}])')
            modeled_sequence_chart = alt.Chart(modeled_df, title=title).mark_line().encode(x="Timestep", y="value",
                                                                                           color=alt.Color("Sequences",
                                                                                                           sort=None)
                                                                                           ).interactive()
            with col1:
                st.altair_chart(real_sequence_chart, use_container_width=True)
            with col1:
                st.altair_chart(modeled_sequence_chart, use_container_width=True)

    # check whether we have selected more than one singular vector
    if len(block_selector) >= 2:
        with col2:

            # get the selected eigenvectors
            coordinate_df = right_df[block_selector].copy()
            # coordinate_df = coordinate_df * values[[eigenvector2num[ele] for ele in block_selector]]

            if len(block_selector) == 2:
                # make a scatter plot with coordinates from the singular vectors weighted by the singular values
                coordinate_df.rename(columns={ele: str(idx) for idx, ele in enumerate(block_selector)}, inplace=True)

            else:

                # make tsne on the data
                perplexity = int(np.sqrt(coordinate_df.shape[0]))

                # compute the TSNE representation of the data
                tsne = TSNE(metric='euclidean', init='random', perplexity=perplexity, random_state=42)
                transf = tsne.fit_transform(coordinate_df)

                # make the coordinates
                coordinate_df = pd.DataFrame(transf, columns=['0', '1'])

            # make a scatter plot from the coordinates
            coordinate_df["idx"] = coordinate_df.index
            st.altair_chart(alt.Chart(coordinate_df, title="Sequence Scatter").mark_circle().encode(x="0", y="1", color='idx',
                                                                                                    tooltip=["idx", "0", "1"])
                            .interactive())
