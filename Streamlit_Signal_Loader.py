from util.mtss_data_loader import get_all_tags, load_timeseries, get_datasources
import streamlit as st
import pandas as pd
import os


start_criteria = None
stop_criteria = None
sequences = []
score_window = None
selected_tags = []
window_size = None

# select unit
datasource = st.selectbox(
    "Datasource",
    get_datasources(),
    key="datasource",
    index=None,
)
if datasource is not None:
    tags = get_all_tags(datasource)
    ts_from = pd.to_datetime(st.text_input("Analyse Start", "2018-01-18 06:00:00.000"))
    ts_to = pd.to_datetime(st.text_input("Analyse Ende", "2018-01-18 12:00:00.000"))

    sample_rate = st.selectbox(
        "Sample Rate?", ("1s", "10s", "1min", "5min"), index=None, key="sample_rate"
    )

    match sample_rate:
        case "1s":
            score_window = st.selectbox(
                "Score Time Window?",
                ("1min", "5min", "15min"),
                index=None,
                key="score_window",
            )
        case "10s":
            score_window = st.selectbox(
                "Score Time Window?",
                ("5min", "15min", "30min", "1h"),
                index=None,
                key="score_window",
            )
        case "1min":
            score_window = st.selectbox(
                "Score Time Window?",
                ("15min", "30min", "1h", "2h"),
                index=None,
                key="score_window",
            )
        case "5min":
            score_window = st.selectbox(
                "Score Time Window?",
                ("1h", "3h", "6h"),
                index=None,
                key="score_window",
            )

    if score_window is not None:
        window_size = int(pd.Timedelta(score_window) / pd.Timedelta(sample_rate))
        st.write(f"Anzahl Datenpunkte zur Berechnung des Scores: {window_size}")

    st.divider()
    if len(tags) > 0:
        tags_df = pd.DataFrame()
        tags_df["tagname"] = tags
        tags_df["favorite"] = False
        tags_df = tags_df.sort_values("tagname")

        st.write("G42MBD11CY101_1X, MBY10CE901_XQ01, MBY10CS901_ZQ11")
        tags_df = st.data_editor(
            tags_df,
            column_config={
                "favorite": st.column_config.CheckboxColumn(
                    "Your favorite?",
                    help="Select your **favorite** widgets",
                    default=False,
                )
            },
            disabled=["widgets"],
            hide_index=True,
        )
        st.write("selected_tags:")
        selected_tags = list(tags_df.loc[tags_df["favorite"] == True, "tagname"])
        st.write(selected_tags)

        if (len(selected_tags) > 0) & (window_size is not None):
            load_data = st.toggle("Daten laden")
            if load_data:
                df_score = load_timeseries(
                    ts_from, ts_to, datasource, selected_tags, sample_rate, window_size
                )

                if len(df_score) > 0:
                    tagname_to_plot = st.selectbox(
                        "Wähle ein Signal aus zum Plotten",
                        selected_tags,
                        key="tagname_to_plot",
                        index=None,
                    )
                    if tagname_to_plot != None:
                        st.scatter_chart(
                            df_score,
                            y=tagname_to_plot,
                            color="changescore " + tagname_to_plot,
                        )
                        st.line_chart(df_score[tagname_to_plot])
                        st.line_chart(df_score["changescore " + tagname_to_plot])

                    if st.button("Speichere Daten"):
                        df_score.to_parquet(os.path.join("data", "data_scored.parquet"))
                        st.write("Daten wurden gespeichert")
