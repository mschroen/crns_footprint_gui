import streamlit as st
import numpy as np
import pandas as pd
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, datetime, timedelta
import pytz
from io import StringIO
from scipy import interpolate

from lib.functions import capture_stdout, show_footer

try:
    from neptoon.corrections.theory.calibration_functions import Schroen2017
except ImportError as e:
    st.error(f"Missing or problematic packages: {e}")

# Streamlit GUI

st.subheader(":material/line_style: Vertical averaging of soil moisture")

st.markdown(
    "Upload a CSV file with datetime and TDR-like soil moisture measurements as a column for each measurement depth. This script will calculate the weighted-average soil moisture time series as seen by a Cosmic-Ray Neutron Sensor."
)


def guess_depth_from_column_name(column_name):
    """Extract depth value from column name using regex"""
    # Look for numbers in the column name
    numbers = re.findall(r"\d+", str(column_name))
    if numbers:
        # Return the first number found, assuming it's the depth
        return int(numbers[0])
    return 10  # Default value if no number found


def generate_colors_by_depth(depths):
    """Generate earth-tone colors from green (shallow) to brown (deep) with lower saturation"""
    colors = []
    n_colors = len(depths)

    # Sort depths to assign colors properly
    sorted_depths = sorted(depths)
    depth_to_color = {}

    for i, depth in enumerate(sorted_depths):
        # Create earth-tone colors from green to brown
        # Hue goes from 120° (green) to 30° (brown/orange)
        hue = 120 - (i / max(1, n_colors - 1)) * 90  # 120° to 30° (green to brown)

        # Lower saturation for more muted, natural colors
        saturation = 25 + (i / max(1, n_colors - 1)) * 15  # 25% to 40%

        # Higher luminosity for better visibility
        lightness = 50 + (i / max(1, n_colors - 1)) * 25  # 50% to 75%

        color = f"hsl({hue}, {saturation}%, {lightness}%)"
        depth_to_color[depth] = color
        colors.append(color)

    return depth_to_color


def process_data(df, bulk_density, depth_mapping):
    """
    Process the dataframe by adding D86 and Avg columns
    df: Dataframe with time as index, and one column for each depth
    bulk_density: single value
    depth_mapping: dict of column names with values of depth in cm
    """
    df_processed = df.copy()

    moisture_cols = df.columns
    # For estimateing D86, take the upper 15 cm as initial values
    relevant_moisture_cols = [col for col in df.columns if depth_mapping[col] <= 15]
    depths = [depth_mapping[col] / 100 for col in df.columns]

    if len(moisture_cols) > 0:
        df_processed["measurement_depth"] = Schroen2017.calculate_measurement_depth(
            distance=1.0,
            bulk_density=bulk_density,
            volumetric_soil_moisture=df_processed[relevant_moisture_cols].mean(axis=1),
        )
    else:
        df_processed["measurement_depth"] = np.nan

    if len(moisture_cols) > 0:
        vertical_layers = np.linspace(0.001, 1.01, 1000)
        listofweights = []
        sm_wavg = []
        for i, row in df_processed.iterrows():
            vertical_weighs = Schroen2017.vertical_weighting(
                depth=vertical_layers * 100,
                distance=1.0,
                bulk_density=bulk_density,
                volumetric_soil_moisture=row[relevant_moisture_cols].mean(),
            )
            vertical_weighs /= vertical_weighs.sum()
            listofweights.append(vertical_weighs)
            f = interpolate.interp1d(
                depths,
                row[moisture_cols],
                kind="linear",  # nearest can also be interesting
                fill_value=(row[moisture_cols[0]] * 1.2, row[moisture_cols[-1]]),
                bounds_error=False,
            )
            row_avg = np.nansum(vertical_weighs * f(vertical_layers))
            if row_avg == 0:
                row_avg = np.nan
            sm_wavg.append(row_avg)

        df_processed["SM_vw_avg"] = sm_wavg

    else:
        df_processed["SM_vw_avg"] = np.nan

    return df_processed


def create_plot(df, depth_mapping):
    """Create the time series plot with multiple y-axes"""

    # Create subplots with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Generate colors based on depths
    depth_to_color = generate_colors_by_depth(list(depth_mapping.values()))

    # Add soil moisture lines
    for col, depth in depth_mapping.items():
        color = depth_to_color[depth]
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines",
                name=f"{col} ({depth} cm)",
                line=dict(color=color),
                yaxis="y",
            ),
            secondary_y=False,
        )

    # Add D86 line with fill
    if "measurement_depth" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["measurement_depth"],
                mode="lines",
                name="Measurement Depth",
                line=dict(color="brown", width=1, dash="dash"),
                yaxis="y2",
            ),
            secondary_y=True,
        )

    # Add Avg line
    if "SM_vw_avg" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SM_vw_avg"],
                mode="lines",
                name="Weighted Average",
                line=dict(color="black", width=3),
                yaxis="y",
            ),
            secondary_y=False,
        )

    # Update layout
    fig.update_layout(
        title="Soil Moisture Time Series",
        xaxis_title="Date/Time",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=20),
    )

    # Set y-axes titles
    fig.update_yaxes(title_text="Soil Moisture / Average", secondary_y=False)
    fig.update_yaxes(title_text="Measurement Depth", secondary_y=True)
    fig.update_yaxes(showgrid=False, zeroline=False, range=[0, None], secondary_y=True)
    return fig


# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    try:
        df = pd.read_csv(uploaded_file)
        # Convert first column to datetime
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df = df.set_index(df.iloc[:, 0])
        # Get column names (excluding the first datetime column)
        moisture_columns = df.columns[1:].tolist()
        if len(moisture_columns) == 0:
            st.error(
                "No soil moisture columns found. Please ensure your CSV has at least two columns."
            )
        else:
            st.subheader("Define Soil Depths")
            st.write(
                "Set the depth (in cm) for each soil moisture column. I have tried to make a guess based on the given name."
            )

            # Create depth mapping inputs
            depth_mapping = {}

            depth_columns = st.columns(len(moisture_columns))

            for i, column in enumerate(moisture_columns):
                # Guess initial depth from column name
                guessed_depth = guess_depth_from_column_name(column)

                # Alternate between columns for better layout
                with depth_columns[i]:
                    depth = st.number_input(
                        f"Depth for column '{column}' (in cm)",
                        min_value=0,
                        max_value=100,
                        value=guessed_depth,
                        step=5,
                        key=f"depth_{column}",
                    )
                    depth_mapping[column] = depth

            # Bulk density slider
            bulk_density = st.slider(
                "Average soil bulk density (g/cm³)",
                min_value=0.0,
                max_value=2.0,
                value=1.40,
                step=0.05,
            )

            # Process data button
            if st.button("Process Data and Generate Analysis", type="primary"):
                # Process the data

                processed_df = process_data(
                    df[moisture_columns], bulk_density, depth_mapping
                )

                # Store processed data in session state
                st.session_state.processed_df = processed_df
                st.session_state.depth_mapping = depth_mapping
                st.session_state.data_processed = True

    except Exception as e:
        st.error(f"Problems to parse CSV file: {str(e)}")

# Show results if data has been processed
if hasattr(st.session_state, "data_processed") and st.session_state.data_processed:
    processed_df = st.session_state.processed_df
    depth_mapping = st.session_state.depth_mapping
    # Create tabs
    tab1, tab2 = st.tabs([":material/line_axis: Plot", ":material/table: Data"])
    with tab1:
        # Create and display plot
        fig = create_plot(processed_df, depth_mapping)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.dataframe(processed_df)

    # Download button
    csv_buffer = StringIO()
    processed_df.to_csv(csv_buffer, index=False)
    csv_string = csv_buffer.getvalue()

    st.download_button(
        label="Download CSV",
        data=csv_string,
        file_name="vert_weight_avg_soil_moisture_data.csv",
        mime="text/csv",
        icon=":material/download:",
        on_click="ignore",
        help="vert_weight_avg_soil_moisture_data.csv",
    )


else:
    if uploaded_file is None:
        st.info("👆 Please upload a CSV file to get started")

show_footer()
