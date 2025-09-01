import streamlit as st

# st.set_page_config(layout="wide")
import pandas as pd
from util.mtss_data_loader import (
    get_sequences,
    get_sequence_data,
    get_all_tags,
)
from util.mtss_data_loader import Event, Threshold, Timedelta, get_datasources
import os
import shutil
from zipfile import ZipFile


start_criteria = None
stop_criteria = None
sequences = []

# select unit
datasource = st.selectbox(
    "Datasource",
    get_datasources(),
    key="datasource",
    index=None,
)
if datasource != None:
    tags = pd.DataFrame()
    tags["Name"] = get_all_tags(datasource)
    ts_from = pd.to_datetime(
        st.date_input("Analyse Startdatum", pd.Timestamp("2023-11-01").date())
    )
    ts_to = pd.to_datetime(
        st.date_input("Analyse Enddatum", pd.Timestamp("2023-12-31").date())
    )
    st.divider()
    # select timerange, start signal, stop signal and maximal duration
    if len(tags) > 0:
        colA, colB = st.columns(2)
        with colA:
            st.header("Start Kriterium")
            start_criteria_type = st.radio(
                "typ",
                options=["Event", "Threshold", "Timedelta"],
                horizontal=True,
                key="start_criteria_type",
                index=None,
            )

            match start_criteria_type:
                case "Event":
                    start_tagname = st.selectbox(
                        "Wähle ein Signal aus",
                        tags["Name"],
                        index=None,
                        key="start_tagname_event",
                    )
                    st.write("Beispiel: MBM13EU010_ZV01")
                    start_value = st.radio(
                        "Welchen Wert soll das Signal haben",
                        options=[1, 0],
                        horizontal=True,
                        key="start_value_event",
                        index=None,
                    )
                    if (start_tagname != None) & (start_value != None):
                        start_criteria = Event(start_tagname, start_value)

                case "Threshold":
                    start_tagname = st.selectbox(
                        "Wähle ein Signal aus",
                        tags["Name"],
                        index=None,
                        key="start_tagname_threshold",
                    )
                    start_value = st.number_input(
                        "Welchen Wert soll das Signal haben",
                        value=None,
                        key="start_value_threshold",
                    )
                    start_edge = st.radio(
                        "steigen oder fallend?",
                        options=["rise", "fall"],
                        horizontal=True,
                        index=None,
                        key="start_edge_threshold",
                    )
                    if (
                        (start_tagname != None)
                        & (start_value != None)
                        & (start_edge != None)
                    ):
                        start_criteria = Threshold(
                            start_tagname, start_value, start_edge
                        )
                case "Timedelta":
                    start_value = st.number_input(
                        label="Wert",
                        value=None,
                        min_value=1,
                        step=1,
                        key="start_value_timedelta",
                    )
                    start_unit = st.radio(
                        "Einheit",
                        ["min", "s"],
                        index=None,
                        key="start_unit_timedelta",
                        horizontal=True,
                    )
                    if (start_value != None) & (start_unit != None):
                        start_criteria = Timedelta(start_value, start_unit)

        with colB:
            st.header("Stop Kriterium")
            stop_criteria_type = st.radio(
                "typ",
                options=["Event", "Threshold", "Timedelta"],
                horizontal=True,
                key="stop_criteria_type",
                index=None,
            )
            match stop_criteria_type:
                case "Event":
                    stop_tagname = st.selectbox(
                        "Wähle ein Signal aus",
                        tags["Name"],
                        key="stop_tagname_event",
                        index=None,
                    )
                    st.write("Beispiel: MKY01EU010_ZV01")
                    stop_value = st.radio(
                        "Welchen Wert soll das Signal haben",
                        options=[1, 0],
                        horizontal=True,
                        key="stop_value_event",
                        index=None,
                    )
                    if (stop_tagname != None) & (stop_value != None):
                        stop_criteria = Event(stop_tagname, stop_value)

                case "Threshold":
                    stop_tagname = st.selectbox(
                        "Wähle ein Signal aus",
                        tags["Name"],
                        index=None,
                        key="stop_tagname_threshold",
                    )
                    stop_value = st.number_input(
                        "Welchen Wert soll das Signal haben",
                        value=None,
                        key="stop_value_threshold",
                    )
                    stop_edge = st.radio(
                        "steigen oder fallend?",
                        options=["rise", "fall"],
                        horizontal=True,
                        index=None,
                        key="stop_edge_threshold",
                    )
                    if (
                        (stop_tagname != None)
                        & (stop_value != None)
                        & (stop_edge != None)
                    ):
                        stop_criteria = Threshold(stop_tagname, stop_value, stop_edge)
                case "Timedelta":
                    stop_value = st.number_input(
                        label="Wert",
                        value=None,
                        min_value=1,
                        step=1,
                        key="stop_value_timedelta",
                    )
                    stop_unit = st.radio(
                        "Einheit",
                        ["min", "s"],
                        index=None,
                        key="stop_unit_timedelta",
                        horizontal=True,
                    )
                    if (stop_value != None) & (stop_unit != None):
                        stop_criteria = Timedelta(stop_value, stop_unit)

        if (start_criteria != None) & (stop_criteria != None):
            sequences = get_sequences(
                ts_from,
                ts_to,
                datasource,
                start_criteria,
                stop_criteria,
                max_sequence_duration=60 * 60 * 6,
            )
            max_sequence_duration = st.slider(
                "maximale Sequenzlänge in Sekunden", 0, 1800, value=1800
            )
            sequences = sequences[
                sequences["duration"] <= max_sequence_duration
            ].reset_index(drop=True)
            st.divider()
            st.header("Sequenzen")
            with st.popover("Zeige Details"):
                st.write(sequences)
            st.line_chart(sequences[["duration"]])

        if len(sequences) > 0:
            st.divider()
            st.header("Signalauswahl")
            selected_values = st.multiselect(
                "Welche Signalverläufe willst du haben?",
                options=tags,
                key="selected tags",
            )
            if len(selected_values) > 0:
                load_data = st.toggle("Daten laden")
                if load_data:
                    df = get_sequence_data(
                        datasource, tuple(selected_values), sequences
                    )
                    st.write(df)
                    if len(df) > 0:
                        tagname_to_plot = st.selectbox(
                            "Wähle ein Signal aus zum Plotten",
                            df.columns,
                            key="tagname_to_plot",
                            index=None,
                        )
                        if tagname_to_plot != None:
                            x = df[[tagname_to_plot]].reset_index()
                            st.line_chart(
                                x,
                                x="dt",
                                y=tagname_to_plot,
                                color="sequence id",
                                width=10,
                            )

                        if st.button("Speichere Sequenzdaten"):
                            path = os.path.join("data", "sequences")
                            if os.path.exists(path):
                                shutil.rmtree(path)
                            os.mkdir(path)
                            sequences_to_store = sequences.copy()
                            sequences_to_store.index.name = "sequence id"
                            sequences_to_store.columns = ["from", "to", "duration"]
                            sequences_to_store.to_csv(
                                os.path.join(path, "sequences.csv"), index=True, header=True
                            )
                            for seq_id in df.index.get_level_values(0).unique():
                                df.loc[(seq_id,), :].to_csv(
                                    os.path.join(path, f"{seq_id}.csv"), index=True, header=True
                                )
                            st.write("Daten wurden gespeichert")
