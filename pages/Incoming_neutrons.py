import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import date, datetime, timedelta
import pytz

from lib.functions import capture_stdout, show_footer_nmdb

try:
    from neptoon.external.nmdb_data_collection import (
        NMDBConfig,
        NMDBDataHandler,
        CacheHandler,
    )
except ImportError as e:
    st.error(f"Missing or problematic packages: {e}")

st.subheader(":material/trending_down: Incoming cosmic-ray reference")

st.markdown(
    "The cosmic-ray reference signal is used to correct the CRNS data for incoming variation of particles, e.g. due to solar events or the solar cycle. This reference signal is measured independently by neutron monitors. The signal of a neutron monitor at a similar geomagnetic cutoff-rigidity and altitude compared to the CRNS location represents the incoming flux at the CRNS site in the best way. Please visit the [NMDB station map](https://www.nmdb.eu/nest/help.php#helpstations) for more information. Select a nearby neutron monitor, so that its data can be downloaded from [NMDB](http://www.nmdb.eu/nest). You can then inspect the data and download it as CSV."
)

stations = dict(
    JUNG=(46.55, 7.98),
    SOPO=(-90, 0),
    OULU=(65.0544, 25.4681),
    PSNM=(18.59, 98.49),
    MXCO=(19.8, -99.1781),
    HRMS=(-34.43, 19.23),
)

stations_acknowledgements = dict(
    JUNG="IGY Jungfraujoch (Physikalisches Institut, University of Bern, Switzerland)",
    SOPO="South Pole (University of Wisconsin, River Falls, USA)",
    OULU="Oulu (Sodankyla Geophysical Observatory of the University of Oulu, Finland)",
    PSNM="Doi Inthanon (Princess Sirindhorn NM) (Mahidol University, Chulalongkorn University, and Ubon Rajathanee University, Thailand)",
    MXCO="Mexico (Cosmic Ray Group, Geophysical Institute, National Autonomous University of Mexico, UNAM, Mexico)",
    HRMS="Hermanus (Centre for Space Research, North-West University, Potchefstroom, South Africa)",
)

@st.fragment
def select_nm(nmdbstation_is="JUNG"):

    c1, c2 = st.columns(2, vertical_alignment="top")

    with c1:
        nmdbstation = st.pills(
            "Select a nearby high-energy neutron monitor",
            options=stations.keys(),
            default=nmdbstation_is,
            selection_mode="multi",
            key="nm_station",
        )

        c11, c12, c13 = st.columns(3)
        date_start = c11.date_input(
            "Period from",
            date(2025, 1, 1),
            max_value="today",
            min_value=date(1951, 1, 1),
            format="YYYY-MM-DD",
            key="date_start",
        )
        date_end = c12.date_input(
            "to",
            date(2025, 2, 1),
            max_value="today",
            min_value=date(1951, 1, 1),
            format="YYYY-MM-DD",
            key="date_end",
        )
        resolution = c13.number_input(
            "Resolution (Min)",
            min_value=1,
            value=60,
            step=1,
            key="resolution",
            help="Examples: 1 (1 Min), 60 (1 hour), 720 (12 hours), 1440 (1 day)",
        )

        st.checkbox(
            "Use cache",
            True,
            help="By default, Neptoon uses cached files of already downloaded data. Deactivate cache if you want to run a fresh download of the latest data.",
            key="use_cache",
        )

    with c2:
        fig = go.Figure(
            go.Scattergeo(
                lat=[ll[0] for ll in stations.values()],
                lon=[ll[1] for ll in stations.values()],
                marker=dict(color="blue"),
                name="Available stations",
                hoverinfo="text",
                text=list(stations.keys()),
            )
        )

        if len(nmdbstation) > 0:
            last_picked_station = nmdbstation[-1]
            selected_stations = {
                key: stations[key] for key in nmdbstation if key in stations
            }
            fig.add_trace(
                go.Scattergeo(
                    lat=[ll[0] for ll in selected_stations.values()],
                    lon=[ll[1] for ll in selected_stations.values()],
                    marker=dict(color="orange"),
                    hoverinfo="text",
                    text=list(selected_stations.keys()),
                    name="Selected station",
                )
            )
        else:
            last_picked_station = "JUNG"

        # Show own position
        # fig.add_trace(
        #     go.Scattergeo(
        #         lat=[
        #             st.session_state["yaml"].sensor_config.sensor_info.latitude
        #         ],
        #         lon=[
        #             st.session_state[
        #                 "yaml"
        #             ].sensor_config.sensor_info.longitude
        #         ],
        #         marker=dict(color="red", symbol="star"),
        #         name="CRNS location",
        #     )
        # )

        # editing the marker
        fig.update_traces(marker_size=10)

        # this projection_type = 'orthographic is the projection which return 3d globe map'
        fig.update_geos(
            projection=dict(
                type="orthographic",
                rotation=dict(
                    lat=stations[last_picked_station][0],
                    lon=stations[last_picked_station][1],
                ),  # , roll=15),
            )
        )

        # layout, exporting html and showing the plot
        fig.update_layout(
            height=200,
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            legend=dict(
                yanchor="bottom",
                y=0.0,
            ),
        )

        st.plotly_chart(fig)


# @st.cache_data(show_spinner="Downloading from NMDB...")
def attach_nmdb(
    start_date_wanted="2023-01-01",
    end_date_wanted="2023-01-31",
    station="JUNG",
    resolution=60,
    use_cache=True,
    nmdb_table="revori",
):
    config = NMDBConfig(
        start_date_wanted=start_date_wanted,
        end_date_wanted=end_date_wanted,
        station=station,
        resolution=resolution,
        nmdb_table=nmdb_table,
    )
    nmdb_handler = NMDBDataHandler(config)
    if not use_cache:
        nmdb_handler.cache_handler.delete_cache()

    nmdb_data = nmdb_handler.collect_nmdb_data()

    return nmdb_data


select_nm("JUNG")

c1, c2 = st.columns([1, 3])
process_button = c1.button(
    ":material/download: Get Neutron Monitor data", type="primary"
)

if process_button:

    number_of_stations = len(st.session_state["nm_station"])
    if number_of_stations == 0:
        st.warning(":material/warning: You need to select a station first.")
    elif st.session_state["date_start"] >= st.session_state["date_end"]:
        st.warning(":material/warning: Start date cannot be after end date")
    else:

        progress_bar = c2.progress(0)

        i = 0
        data_dict = {}
        for nm_station in st.session_state["nm_station"]:
            i += 1
            progress_bar.progress(
                (i - 1) / number_of_stations * 0.7,
                text="Downloading %s..." % nm_station,
            )

            data_nm_days = []
            if st.session_state["resolution"] == 1:
                start = datetime.combine(
                    st.session_state["date_start"], datetime.min.time()
                )
                end = datetime.combine(
                    st.session_state["date_end"], datetime.min.time()
                )
                delta = end - start
                days = delta.days
                j = 0
                for d in [start + timedelta(days=d) for d in range(days + 1)]:
                    j += 1
                    progress_bar.progress(
                        ((i - 1) + (j - 1) / (days + 1)) / number_of_stations * 0.7,
                        text="Downloading %s (%s)..."
                        % (nm_station, d.strftime("%Y-%m-%d")),
                    )
                    d_start = d.replace(hour=0, minute=0, second=0, tzinfo=pytz.UTC)
                    d_end = d.replace(hour=23, minute=59, second=0, tzinfo=pytz.UTC)
                    if d_end > datetime.now(pytz.UTC):
                        d_end = min(end, datetime.now(pytz.UTC))

                    data_nm_d = attach_nmdb(
                        start_date_wanted=str(d_start),
                        end_date_wanted=str(d_end),
                        station=nm_station,
                        resolution=st.session_state["resolution"],
                        use_cache=st.session_state["use_cache"],
                    )

                    data_nm_days.append(data_nm_d)

                data_nm = pd.concat([d for d in data_nm_days if d is not None], axis=0)

                if not data_nm.empty:
                    # Clean data
                    data_nm = data_nm[~data_nm.index.duplicated(keep="first")]
                    # data_nm.index = data_nm.index.tz_localize(None).tz_localize('UTC')
                    data_nm = data_nm.sort_index()
                    # mean = data_nm.mean()
                    # self.data.loc[self.data['NM'] < mean / 1.2, 'NM'] = np.nan
                    # self.data.loc[self.data['NM'] > mean * 1.2, 'NM'] = np.nan
                    data_nm = data_nm.interpolate()
            else:
                data_nm = attach_nmdb(
                    start_date_wanted=str(st.session_state["date_start"]),
                    end_date_wanted=str(st.session_state["date_end"]),
                    station=nm_station,
                    resolution=st.session_state["resolution"],
                    use_cache=st.session_state["use_cache"],
                )
            data_dict[nm_station] = data_nm

        progress_bar.progress(0.7, text="Merging data...")
        data = pd.concat(
            {key: series["count"] for key, series in data_dict.items()}, axis=1
        )

        tab1, tab2 = st.tabs([":material/show_chart: Plots", ":material/table: Data"])

        progress_bar.progress(0.8, text="Plotting data...")
        fig = go.Figure()

        for column in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[column] / data[column].mean(),
                    mode="lines",
                    name=column,
                    line=dict(width=2),
                )
            )

        fig.update_layout(
            yaxis_title="Normalized counts",
            hovermode="x unified",
            margin=dict(t=20),
        )

        tab1.plotly_chart(fig, use_container_width=True)

        with tab2:
            progress_bar.progress(0.9, text="Generating data table...")
            st.dataframe(data)

        progress_bar.progress(1.0, text=":material/check_circle: Completed.")

        file_name = "nmdb-{:}-{:}Min-{:}-{:}.csv".format(
            "_".join(st.session_state["nm_station"]),
            st.session_state["resolution"],
            st.session_state["date_start"].strftime("%Y%m%d"),
            st.session_state["date_end"].strftime("%Y%m%d"),
        )

        st.download_button(
            label="Download CSV",
            data=data.to_csv().encode("utf-8"),
            file_name=file_name,
            mime="text/csv",
            icon=":material/download:",
            on_click="ignore",
            help=file_name,
        )
        station_acknowledgement_text = ", ".join([stations_acknowledgements[nm_station] for nm_station in st.session_state["nm_station"]])
        st.info(
            f":material/info: Data retrieved via NMDB are the property of the individual data providers. These data are free for non commercial use to within the restriction imposed by the providers. If you use such data for your research or applications, please acknowledge the origin by a sentence like: 'We acknowledge the NMDB database (www.nmdb.eu) founded under the European Union's FP7 programme (contract no. 213 007), and the PIs of individual neutron monitors at: {station_acknowledgement_text}.'."
        )

show_footer_nmdb()
