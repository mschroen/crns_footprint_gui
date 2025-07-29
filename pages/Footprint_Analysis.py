import streamlit as st

import os
import numpy as np
from glob import glob
from scipy import optimize
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from lib.functions import capture_stdout, show_footer

try:
    from figurex import Figure

    from neptoon.corrections import Schroen2017

    from lib.Schroen2022hess import (
        Field_at_Distance,
        Generate_Splitfields,
        Plot_Field,
        Plot_Ntheta,
        Plot_Sensibility,
        optimize_R,
    )
except ImportError as e:
    st.error(f"Missing or problematic packages: {e}")


def create_stacked_horizontal_bar(R_50, R_33, R_66, R_20, R_40, R_60, R_80):
    # Define the data
    data = {
        "50 %": [0, R_50, 300],  # y=2: 0-28, 28-300
        "33 %": [0, R_33, R_66, 300],  # y=3: 0-5, 5-67, 67-300
        "20 %": [
            0,
            R_20,
            R_40,
            R_60,
            R_80,
            300,
        ],  # y=5: 0-2, 2-12, 12-49, 49-118, 118-300
    }

    # Color palettes for each y value (decreasing saturation, increasing luminosity)
    color_palettes = {
        "50 %": ["#1f77b4", "#87ceeb"],  # Blue shades
        "33 %": ["#2ca02c", "#90ee90", "#d3f5d3"],  # Green shades
        "20 %": [
            "#ff7f0e",
            "#ffb366",
            "#ffd4b3",
            "#ffe6d9",
            "#fff2e6",
        ],  # Orange shades
    }

    # Create the figure
    fig = go.Figure()

    # Annotation percentages for each y value
    # annotations_text = {"2": "50%", "3": "33%", "5": "20%"}

    # Process each y value
    for y_val in ["50 %", "33 %", "20 %"]:
        values = data[y_val]
        colors = color_palettes[y_val]

        # Calculate bar widths (differences between consecutive values)
        bar_widths = [values[i + 1] - values[i] for i in range(len(values) - 1)]

        # Add bars for this y value
        for i, (width, color) in enumerate(zip(bar_widths, colors)):
            # Calculate the starting position (x0) for this bar segment
            x0 = values[i]

            # Add horizontal bar
            fig.add_trace(
                go.Bar(
                    y=[y_val],
                    x=[width],
                    orientation="h",
                    name=f"Annuli {y_val} - Segment {i+1}",
                    marker_color=color,
                    base=x0,
                    showlegend=False,  # Hide legend for cleaner look
                    width=0.6,  # Make bars thinner
                    hovertemplate=f"<b>Annulus with {y_val} contribution</b><br>"
                    + f"Distance: {x0:.0f}-{values[i+1]:.0f} m<br>"
                    + "<extra></extra>",
                )
            )

    # Update layout
    fig.update_layout(
        title={
            "text": "Circular areas of equal contribution",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16},
            "y": 0.95,  # Position title closer to top
            "yanchor": "top",
        },
        xaxis_title="Distance from sensor (m)",
        yaxis_title="Contributions",
        barmode="relative",  # This creates the stacked effect
        height=250,
        # width=800,
        template="plotly_white",
        yaxis=dict(
            categoryorder="array",
            categoryarray=["20 %", "33 %", "50 %"],  # Top to bottom: 2, 3, 5
            tickmode="array",
            tickvals=["50 %", "33 %", "20 %"],
            ticktext=["50 %", "33 %", "20 %"],
        ),
        xaxis=dict(
            range=[0, 300],
            dtick=50,
            showgrid=True,  # Enable vertical grid lines
            gridwidth=1,  # Grid line width
            gridcolor="lightgray",
        ),  # Show ticks every 50 units
        margin=dict(l=80, r=20, t=40, b=60),  # Increased top margin for annotations
        bargap=0.5,  # Very small gap between bars
        bargroupgap=0.05,  # Small gap between bar groups
    )

    return fig


# def Footprint_content():
st.header(":material/barefoot: Footprint Analysis")

with st.container(border=True):
    st.subheader(":material/settings: General settings")

    c1, c2 = st.columns(2)
    bulk_density = c1.slider(
        "Soil bulk density ρ_bulk (kg/m³)",
        min_value=0.5,
        max_value=2.5,
        value=1.43,
        step=0.01,
    )

    N0 = c2.slider(
        "Neutron calibration parameter N₀ (cph)",
        min_value=100,
        max_value=10000,
        value=1500,
        step=10,
    )

    c1, c2 = st.columns(2)
    air_humidity = c1.slider(
        "Air humidity h (g/m³)",
        min_value=0,
        max_value=25,
        value=5,
        step=1,
        format="%.0f",
    )
    air_pressure = c2.slider(
        "Air pressure P (hPa)", min_value=500, max_value=1100, value=1013, step=1
    )

with st.container(border=True):
    st.subheader(":material/adjust: Radial footprint")
    st.markdown(
        "In a homogeneous environment, the radial footprint is a good estimate for the area that mainly influences the detector. Near fields have a stronger influence than far fields. By deviding the footprint area into annuli of equal contribution, one could estimate how much of the signal comes from which annulus. The plot shows three examples of 2, 3 or 5 equally contributing annuli. This information could be used for soil sampling campaigns to take equal number of samples in each annulus."
    )

    @st.fragment
    def radial_footprint():
        c1, c2 = st.columns([1, 3])

        with c1:
            sm = st.slider(
                "Soil moisture (m³/m³)",
                min_value=0.0,
                max_value=0.80,
                value=0.20,
                step=0.01,
            )

            R86 = Schroen2017.calculate_footprint_radius(
                volumetric_soil_moisture=sm,
                abs_air_humidity=air_humidity,
                atmospheric_pressure=air_pressure,
            )

            st.metric(
                "Footprint radius (R86)",
                f"{R86:.1f} m",
                help="Defined as the annulus with 86.5% contribution.",
            )

        with c2:
            r = pd.Series(np.arange(0.01, 500, 0.01), name="r")
            W = pd.DataFrame(index=r)
            W["r_rescaled"] = Schroen2017.rescale_distance(
                distance_from_sensor=W.index,
                atmospheric_pressure=air_pressure,
            )
            # print(W.index)
            # print(W.r_rescaled)
            # W["w"] = Wr(W.index, sm=sm, hum=air_humidity, p=air_pressure)
            W["w"] = Schroen2017.horizontal_weighting(W.index, sm, air_humidity)
            W["wn"] = W.w / W.w.sum()
            W["wn_cum"] = W["wn"].cumsum()
            R_20 = np.max(W.loc[W["wn_cum"] <= 0.20].r_rescaled)
            R_40 = np.max(W.loc[W["wn_cum"] <= 0.40].r_rescaled)
            R_60 = np.max(W.loc[W["wn_cum"] <= 0.60].r_rescaled)
            R_80 = np.max(W.loc[W["wn_cum"] <= 0.80].r_rescaled)
            R_33 = np.max(W.loc[W["wn_cum"] <= 0.34].r_rescaled)
            R_66 = np.max(W.loc[W["wn_cum"] <= 0.67].r_rescaled)
            R_50 = np.max(W.loc[W["wn_cum"] <= 0.50].r_rescaled)
            # print(R_50, R_33, R_66, R_20, R_40, R_60, R_80)

            fig = create_stacked_horizontal_bar(
                R_50, R_33, R_66, R_20, R_40, R_60, R_80
            )
            st.plotly_chart(fig, use_container_width=True)

    radial_footprint()

with st.container(border=True):
    st.subheader(":material/arrow_range: Field at a Distance")
    st.markdown(
        """
    Consider a main field with θ₁ and a remote field with θ₂ at distance R.
    - What is the remote field's relative contribution to the detector signal?
    - What is the effective neutron counts and effective soil moisture measured by the detector?
    - Is the change in detected neutrons significant with regards to sensor precision?
    """
    )

    @st.fragment
    def update_plot():
        col1, col2, col3 = st.columns(3)

        with col1:
            R = st.slider("Distance R (m)", 0, 400, 100, 1, key="tab1_R")
        with col2:
            theta1 = st.slider(
                "Main field θ₁ (m³/m³)", 0.01, 0.8, 0.1, 0.01, key="tab1_theta1"
            )
        with col3:
            theta2 = st.slider(
                "Remote field θ₂ (m³/m³)", 0.01, 0.8, 0.3, 0.01, key="tab1_theta2"
            )

        try:
            size = 499
            M = Generate_Splitfields(size, R)
            N1, N2, Neff, thetaeff, c1, c2 = Field_at_Distance(
                R,
                theta1=theta1,
                theta2=theta2,
                hum=air_humidity,
                N0=N0,
                air_pressure=air_pressure,
                bd=bulk_density,
                verbose=False,
                max_radius=500,
            )

            # Create plots
            fig, axes = plt.subplots(
                1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [2, 2, 1]}
            )
            # fig.subplots_adjust()

            Plot_Field(
                ax=axes[0],
                Var=M,
                R=R,
                extent=size,
                x_marker=True,
                annotate=(c1, c2),
            )
            Plot_Ntheta(
                axes[1],
                R,
                theta1,
                theta2,
                thetaeff,
                N1,
                N2,
                Neff,
                hum=air_humidity,
                off=0,
                bd=bulk_density,
                N0=N0,
            )
            Plot_Sensibility(axes[2], N1, Neff)

            st.pyplot(fig)

            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                c1_100 = c1 * 100
                st.metric("Main field contribution", f"{c1_100:.1f} %")
            with col2:
                c2_100 = c2 * 100
                st.metric("Remote field contribution", f"{c2_100:.1f} %")
            with col3:
                st.metric("Effective soil moisture", f"{thetaeff:.3f} m³/m³")

        except Exception as e:
            st.error(f"Error in calculations: {str(e)}")

    update_plot()

with st.container(border=True):

    st.subheader(":material/barefoot: The Practical Footprint")
    st.markdown(
        """
    Consider a main field with θ₁ and a remote field that changes from θ₁ to θ₂.
    What is the remote field's maximal distance R, such that the change is still sensible 
    for a given detector sensitivity σ_N?
    """
    )

    @st.fragment
    def update_plot2():
        col1, col2, col3 = st.columns(3)

        with col1:
            dN = st.slider(
                "Detector sensitivity σ_N (-)",
                0.001,
                0.1,
                0.01,
                0.001,
                format="%0.3f",
                key="tab2_dN",
            )
        with col2:
            theta1_fp = st.slider(
                "Main field θ₁ (m³/m³)", 0.01, 0.8, 0.1, 0.01, key="tab2_theta1"
            )
        with col3:
            theta2_fp = st.slider(
                "Remote field θ₂ (m³/m³)", 0.01, 0.8, 0.3, 0.01, key="tab2_theta2"
            )

        try:
            R_optimal = optimize.newton(
                optimize_R,
                1,
                args=(
                    theta1_fp,
                    theta2_fp,
                    air_pressure,
                    dN,
                    N0,
                    bulk_density,
                    False,
                    500,
                ),
            )

            size = 499
            N1, N2, Neff, thetaeff, c1, c2 = Field_at_Distance(
                R_optimal,
                theta1=theta1_fp,
                theta2=theta2_fp,
                N0=N0,
                air_pressure=air_pressure,
                bd=bulk_density,
                verbose=False,
                max_radius=500,
            )

            # R86 = get_footprint(thetaeff, air_humidity, 1013)
            R86 = Schroen2017.calculate_footprint_radius(
                volumetric_soil_moisture=thetaeff,
                abs_air_humidity=air_humidity,
                atmospheric_pressure=air_pressure,
            )

            M = Generate_Splitfields(size, int(R_optimal))

            fig, axes = plt.subplots(
                1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [2, 2, 1]}
            )
            # fig.subplots_adjust(gridspec_kw={"width_ratios": [2, 2, 1]})

            Plot_Field(
                ax=axes[0],
                Var=M,
                R=R_optimal,
                extent=size,
                x_marker=True,
                annotate=(c1, c2),
            )
            Plot_Ntheta(
                axes[1], R_optimal, theta1_fp, theta2_fp, thetaeff, N1, N2, Neff
            )
            Plot_Sensibility(axes[2], N1, Neff)

            st.pyplot(fig)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Max. distance R", f"{R_optimal:.1f} m")
            with col2:
                st.metric("R86 footprint radius", f"{R86:.1f} m")
            with col3:
                st.metric("Effective soil moisture", f"{thetaeff:.3f} m³/m³")

        except Exception as e:
            st.error(f"Error in calculations: {str(e)}")

    update_plot2()

show_footer()
