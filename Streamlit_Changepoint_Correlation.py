import numpy as np
import streamlit as st
import pandas as pd
import altair as alt
from sklearn.manifold import TSNE
import os
import functools

# do some streamlit related elements and code --------------------------------------------------------------------------
st.sidebar.title('Change Point Correlation with TSNE')

# Using object notation
color_selector = st.sidebar.selectbox(
    "What should be coded by color/symbols?",
    ('Block/Turbine', 'Turbine/Block')
)
coding_selection = color_selector.split('/')
coding_selection = list(map(str.lower, coding_selection))

# change position of the searchbar
# https://stackoverflow.com/questions/62103632/altair-change-the-position-of-a-slider
# https://github.com/BugzTheBunny/streamlit_custom_gui/blob/main/utils.py
st.markdown("""
<style>
form.vega-bindings {
  position: absolute;
  left: -420px;
  top: -15px;
}
</style>
""", unsafe_allow_html=True)


# load the data and preprocess it --------------------------------------------------------------------------------------
@st.cache_data()
def get_data(rstate=2):

    # get the data for the evaluation
    df = pd.read_csv('data/Changepoint_Correlation_5_samples_shifted.csv', index_col=0, header=0)
    arr = df.to_numpy()
    arr = arr.astype(float)

    # make the tsne embedding
    tsne = TSNE(metric='precomputed', init='random', perplexity=int(np.sqrt(len(df.columns))), random_state=rstate)
    transf = tsne.fit_transform(arr)

    # read the descriptions
    descriptions = pd.read_excel(os.path.join('data', 'Mappe1.xlsx'), index_col=[0])

    # copy the descriptions
    descriptions = [descriptions.loc[f'#{col.split("_")[0][2:]}.M']['Description']
                    if f'#{col.split("_")[0][2:]}.M' in descriptions.index
                    else "NOT FOUND"
                    for col in list(df.columns)]

    # make the dataframe that we want to have
    bokeh_df = pd.DataFrame({'x': transf[:, 0],
                             'y': transf[:, 1],
                             'original': list(df.columns),
                             'annotation': [f'{col} | {description}'
                                            for col, description in zip(list(df.columns), descriptions)],
                             'block': [col[1:2] for col in list(df.columns)],
                             'block_turbine': [col[1:3] for col in list(df.columns)],
                             'turbine': [f'Steam [{col[2:3]}]' if col[2:3] == '0'
                                         else f'Gas [{col[2:3]}]'
                                         for col in list(df.columns)],
                             'component': [col[3:6] for col in list(df.columns)],
                             'measurement': [col[8:10] for col in list(df.columns)]
                             })
    return bokeh_df


# get the data for the plot
datadf = get_data()


# Select Blocks to show
possible_blocks = list(datadf["block"].unique())
block_selector = st.sidebar.multiselect(
    "Which block do you want to plot",
    possible_blocks,
    default=possible_blocks
)

# make the block selections on the dataframe directly
datadf = datadf[datadf['block'].isin(set(block_selector))]

# select the turbines to show
possible_turbines = list(datadf["turbine"].unique())
turbine_selector = st.sidebar.multiselect(
    "Which turbine do you want to plot?",
    possible_turbines,
    default=possible_turbines
)

# make the turbine selections on the dataframe directly
datadf = datadf[datadf['turbine'].isin(set(turbine_selector))]


# select the turbines to show
possible_components = list(datadf["component"].unique())
component_selector = st.sidebar.multiselect(
    "Which component do you want to plot?",
    possible_components,
    default=possible_components
)

# make the turbine selections on the dataframe directly
datadf = datadf[datadf['component'].isin(set(component_selector))]


# select the turbines to show
possible_measurement = list(datadf["measurement"].unique())
measurement_selector = st.sidebar.multiselect(
    "Which measurement do you want to plot?",
    possible_measurement,
    default=possible_measurement
)

# make the turbine selections on the dataframe directly
datadf = datadf[datadf['measurement'].isin(set(measurement_selector))]


# make altair related stuff --------------------------------------------------------------------------------------------

# check whether there are any values to plot
if datadf.shape[0]:

    # create a brush selector
    brush = alt.selection_interval()

    # create a search input for KKS and labels
    search_input = alt.param(
        value='',
        bind=alt.binding(
            input='search',
            placeholder="MKB",
            name='Search partial labels: ',
        )
    )

    # enable panning and zooming while pressing alt key
    interaction1 = alt.selection_interval(bind='scales',
                                          on="[mousedown[event.altKey], mouseup] > mousemove",
                                          translate="[mousedown[event.altKey], mouseup] > mousemove!",
                                          zoom="wheel![event.altKey]")

    # make the scatter plot of the tsne visualization. Also enable the selectors for color etc.
    points = alt.Chart(datadf, title=f"Changepoint Correlation Visualization. Press [ALT] to zoom and pan.")\
        .mark_point().encode(
        x='x',
        y='y',
        color=alt.condition(alt.expr.test(alt.expr.regexp(search_input, 'i'), alt.datum['annotation']),
                            coding_selection[0],
                            alt.value('lightgrey')),
        shape=alt.Shape(coding_selection[1]),
        opacity=alt.condition(brush,
                              alt.value(1),
                              alt.value(0.01)),
        size=alt.condition(~alt.expr.test(alt.expr.regexp(search_input, 'i'), alt.datum['annotation']),
                           alt.value(5),
                           alt.value(50)),
        tooltip=alt.condition(alt.expr.test(alt.expr.regexp(search_input, 'i'), alt.datum['annotation']),
                              'annotation',
                              alt.value(None))
    ).add_params(brush, search_input, interaction1)

    # create all the histograms for the things we want to visualize
    def create_histograms():
        hists = []
        for col in datadf.columns[7:9]:
            # make the histogram
            block_bars = alt.Chart(datadf).mark_bar().encode(
                x='count()',
                y=col,
                tooltip=[col, 'count()']
            ).transform_filter(
                brush & interaction1
            ).properties(
                width=200
            )
            hists.append(block_bars)
        return functools.reduce(lambda x, y: x & y, hists)


    # calls the function that creates the histograms
    hists = create_histograms()

    # include the plots in streamlit
    st.altair_chart(points | hists, use_container_width=True)

