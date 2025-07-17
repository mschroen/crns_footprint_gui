import streamlit as st
import sys
import os
import io
import numpy as np
from glob import glob
from scipy import optimize
import pandas as pd
import matplotlib.pyplot as plt
import contextlib
import plotly.graph_objects as go


@contextlib.contextmanager
def capture_stdout():
    old_stdout = sys.stdout
    stdout_capture = io.StringIO()
    try:
        sys.stdout = stdout_capture
        yield stdout_capture
    finally:
        sys.stdout = old_stdout


# You'll need to install these packages or replace with equivalent functionality
try:
    from figurex import Figure
    from .lib.uranos import URANOS

    from neptoon.corrections import Schroen2017

    from .lib.Schroen2022hess import (
        Field_at_Distance,
        Generate_Splitfields,
        Plot_Field,
        Plot_Ntheta,
        Plot_Sensibility,
        optimize_R,
    )
except ImportError as e:
    st.error(f"Missing or problematic packages: {e}")

# Streamlit App Configuration
st.set_page_config(
    # page_title="CRNS Signal Contributions Analysis",
    # page_icon="🌍",
    layout="wide",
    # initial_sidebar_state="expanded",
)

# st.title("🌍 CRNS Signal Contributions Analysis")
# st.markdown(
#     """
# This app helps calculate signal contributions of various areas in the footprint of a CRNS detector,
# and determine the practical footprint distance interactively.
# """
# )

# Sidebar for main properties
# st.sidebar.header("Domain parameters:")


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


##########################
##########################
##########################
# Tab 1: Field at Distance
def Footprint_content():
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

    ##########################
    ##########################
    ##########################
    # Tab 2: Practical Footprint
    # def Practical_content():
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


@st.fragment
def change_sm(X, field_mean, sm_weighted_field_mean, N_weighted_field_mean):
    st.subheader("What happens if Soil Moisture changes?")

    if len(X.region_data) > 0:
        col1, col2 = st.columns(2)

        with col1:
            region_options = list(X.region_data["Regions"].values)
            selected_region = st.selectbox(
                "Select region to modify",
                region_options,
                help="Use the region map above to find the ID",
            )

        with col2:
            new_sm = st.slider(
                "New soil moisture value (m³/m³)", 0.01, 0.80, 0.20, 0.01
            )

        c1, c2 = st.columns([1, 3])
        process_button = c1.button(
            ":material/wand_stars: Apply Changes",
            type="primary",
        )

        if process_button:
            progress_bar = c2.progress(0)
            #
            progress_bar.progress(0.25, text="Modifying soil moisture...")
            X = X.modify(Region=selected_region, SM=new_sm)

            progress_bar.progress(0.50, text="Estimating neutrons...")
            X = X.estimate_neutrons()

            progress_bar.progress(0.75, text="Generating plots...")
            # Show modified results
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            X.plot(
                axes[0],
                extent=250,
                image="SM",
                annotate="SM_diff",
                fontsize=12,
                title="",
            )
            axes[0].set_title("a) Soil moisture change")
            X.plot(
                axes[1],
                extent=250,
                image="SM",
                annotate="Contributions_diff",
                fontsize=12,
                title="",
            )
            axes[1].set_title("b) Contributions change")
            X.plot(
                axes[2],
                extent=250,
                image="SM",
                annotate="Contributions",
                fontsize=12,
                title="",
            )
            axes[2].set_title("c) Estimated signal contributions")

            st.pyplot(fig)

            st.write("Updated average soil moisture:")
            c1, c2, c3 = st.columns(3)
            (
                field_mean_2,
                sm_weighted_field_mean_2,
                N_weighted_field_mean_2,
            ) = X.average_sm()

            field_mean_diff = field_mean_2 - field_mean
            sm_weighted_field_mean_diff = (
                sm_weighted_field_mean_2 - sm_weighted_field_mean
            )
            N_weighted_field_mean_diff = N_weighted_field_mean_2 - N_weighted_field_mean

            c1.metric(
                "Field mean (naive approach)",
                f"{field_mean_2:.3f} m³/m³",
                delta=f"{field_mean_diff:.3f} m³/m³",
            )
            c2.metric(
                "SM-weighted field mean (lazy approach)",
                f"{sm_weighted_field_mean_2:.3f} m³/m³",
                delta=f"{sm_weighted_field_mean_diff:.3f} m³/m³",
            )
            c3.metric(
                "N-weighted field mean (correct approach)",
                f"{N_weighted_field_mean_2:.3f} m³/m³",
                delta=f"{N_weighted_field_mean_diff:.3f} m³/m³",
            )

            progress_bar.progress(1.0, text=":material/check_circle: Completed.")


@st.fragment
def compare_uranos(X):
    st.subheader(":material/blur_on: Compare with URANOS simulations")

    if len(X.region_data) > 0:
        col1, col2 = st.columns(2)

        disable_process = False
        with col1:
            path_origins = st.text_input(
                "Path pattern for URANOS detector origin files",
                "examples/detectorOrigins*",
            )
            if len(glob(path_origins)) == 0:
                st.warning(
                    ":material/warning: No files found matching %s" % path_origins
                )
                disable_process = True

        with col2:
            path_neutrons = st.text_input(
                "Path pattern for URANOS epithermal neutron files",
                "examples/densityMapEpithermal*",
            )
            if len(glob(path_neutrons)) == 0:
                st.warning(
                    ":material/warning: No files found matching %s" % path_neutrons
                )
                disable_process = True

        c1, c2 = st.columns([1, 3])
        process_button = c1.button(
            ":material/wand_stars: Compare", type="primary", disabled=disable_process
        )

        if process_button:
            progress_bar = c2.progress(0)
            with capture_stdout() as output:
                # Try to read additional URANOS files
                try:
                    tab1, tab2, tab3, tab4 = st.tabs(
                        [
                            ":material/globe: Map",
                            ":material/table: Data",
                            ":material/sort: Distribution",
                            ":material/more_horiz: Details",
                        ]
                    )
                    tab2.dataframe(X.region_data.head())

                    progress_bar.progress(
                        0.25, text="Reading URANOS detector origins..."
                    )
                    # print(path_origins)
                    X = X.read_origins(path_origins)
                    progress_bar.progress(0.50, text="Reading URANOS neutron counts...")
                    X = X.read_density(path_neutrons)

                    progress_bar.progress(0.50, text="Plotting distributions...")
                    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
                    X.histogram(axes[0], "area")
                    X.histogram(axes[1], "SM")
                    X.histogram(axes[2], "Density")
                    X.histogram(axes[3], "Origins")
                    tab3.pyplot(fig)

                    progress_bar.progress(0.75, text="Plotting Maps...")
                    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
                    X.plot(axes[0], extent=250, image="SM", annotate="SM")
                    X.plot(axes[1], extent=250, image="SM", annotate="Contributions")
                    X.plot(
                        axes[2],
                        extent=250,
                        image="SM",
                        annotate="Origins",
                        overlay="Origins",
                    )
                    X.plot(
                        axes[3],
                        extent=250,
                        image="Density",
                        annotate="Density",
                        cmap="Spectral_r",
                        vmax_factor=1,
                    )
                    tab1.pyplot(fig)

                except Exception as e:
                    st.warning(f"Could not load URANOS simulation files: {str(e)}")

                tab4.code(output.getvalue())
                progress_bar.progress(1.0, text=":material/check_circle: Completed.")


##########################
##########################
##########################
# Tab 3: Signal Contributions
def Signal_content():
    st.header(":material/globe: Signal Contributions of User-Defined Pattern")
    st.markdown(
        "Calculate the signal contribution for a central sensor in a soil moisture domain given by an image file."
    )
    if not "selected_image" in st.session_state:
        st.session_state.selected_image = None

    @st.dialog("Select scenario image")
    def select_image():
        example_files = glob("examples/*.png") if os.path.exists("examples") else []

        if example_files:
            c1, c2 = st.columns([3, 1])
            selected_image = c1.selectbox(
                "Select an image:",
                example_files,
                format_func=lambda x: os.path.basename(x),
                label_visibility="collapsed",
                help="Select an image from the examples folder, a PNG file with RGB-coded soil moisture pattern. Use SM = RGB/2 (e.g., RGB=(10,10,10) for SM=5%).",
            )
            if c2.button("Select", icon=":material/select_check_box:"):
                st.session_state.selected_image = selected_image
                print(st.session_state.selected_image)
                st.rerun()

            # Display selected image
            if selected_image:
                st.image(
                    selected_image,
                    # caption=os.path.basename(selected_image),
                    use_container_width=True,
                )
                selected_image_path = selected_image
        else:
            st.warning("No PNG images found in folder `examples/`")

    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Select image", icon=":material/gallery_thumbnail:"):
            select_image()
        if st.session_state.selected_image:
            st.image(st.session_state.selected_image, width=150)
    with c2:
        if st.session_state.selected_image:
            st.markdown("**%s**" % os.path.basename(st.session_state.selected_image))
        scaling = st.slider("Scaling (meters per pixel)", 0.1, 10.0, 2.0, 0.1)
        air_humidity_img = st.slider("Air humidity (g/m³)", 1, 20, 5, 1)

    # Process button - only enabled when image is selected/uploaded
    image_available = st.session_state.selected_image is not None

    c1, c2 = st.columns([1, 3])
    process_button = c1.button(
        ":material/wand_stars: Analyze",
        disabled=not image_available,
        type="primary",
    )

    if not image_available:
        st.info("Please select an image from the gallery to enable processing.")

    if process_button and image_available:

        if st.session_state.selected_image:
            try:
                progress_bar = c2.progress(0)

                with capture_stdout() as output:
                    try:
                        image_path = st.session_state.selected_image

                        progress_bar.progress(0.1, text="Initializing URANOS...")
                        X = URANOS(folder="", scaling=scaling, hum=air_humidity_img)

                        progress_bar.progress(0.2, text="Reading image...")
                        X = X.read_materials(image_path)

                        progress_bar.progress(
                            0.3, text="Generating soil moisture matrix..."
                        )
                        X = X.material2sm()

                        progress_bar.progress(0.4, text="Generating distance matrix...")
                        X = X.generate_distance()

                        progress_bar.progress(0.5, text="Generating weights...")
                        X = X.genereate_weights()

                        progress_bar.progress(0.6, text="Finding regions...")
                        X = X.find_regions()

                        progress_bar.progress(0.7, text="Estimating neutrons...")
                        X = X.estimate_neutrons()

                        if hasattr(X, "region_data"):
                            progress_bar.progress(0.8, text="Show regional data...")

                            X.region_data.drop(
                                X.region_data[X.region_data.Materials == 218.0].index,
                                inplace=True,
                            )

                            with st.container(border=True):
                                tab1, tab2, tab3, tab4 = st.tabs(
                                    [
                                        ":material/globe: Map",
                                        ":material/table: Data",
                                        ":material/sort: Distribution",
                                        ":material/more_horiz: Details",
                                    ]
                                )
                                tab2.dataframe(X.region_data.head())

                                # Create histograms
                                fig, axes = plt.subplots(1, 3, figsize=(12, 3))
                                X.histogram(axes[0], "area")
                                X.histogram(axes[1], "SM")
                                X.histogram(axes[2], "Contributions")
                                tab3.pyplot(fig)

                                progress_bar.progress(
                                    0.9, text="Visualizing soil moisture..."
                                )

                                st.markdown(
                                    "Average soil moisture seen by the CRNS detector:"
                                )
                                c1, c2, c3 = st.columns(3)
                                (
                                    field_mean,
                                    sm_weighted_field_mean,
                                    N_weighted_field_mean,
                                ) = X.average_sm()
                                c1.metric(
                                    "Field mean (naive approach)",
                                    f"{field_mean:.3f} m³/m³",
                                )
                                c2.metric(
                                    "SM-weighted field mean (lazy approach)",
                                    f"{sm_weighted_field_mean:.3f} m³/m³",
                                )
                                c3.metric(
                                    "N-weighted field mean (correct approach)",
                                    f"{N_weighted_field_mean:.3f} m³/m³",
                                )

                            # Create visualization
                            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
                            X.plot(
                                axes[0],
                                extent=250,
                                image="SM",
                                annotate="Regions",
                                fontsize=12,
                                title="",
                            )
                            axes[0].set_title("a) Region numbers")
                            X.plot(
                                axes[1],
                                extent=250,
                                image="SM",
                                annotate="SM",
                                fontsize=12,
                                title="",
                            )
                            axes[1].set_title("b) Soil moisture map")
                            X.plot(
                                axes[2],
                                extent=250,
                                image="SM",
                                annotate="Contributions",
                                fontsize=12,
                                title="",
                            )
                            axes[2].set_title("c) Estimated signal contributions")

                            tab1.pyplot(fig)

                        progress_bar.progress(
                            1.0, text=":material/check_circle: Completed."
                        )
                        tab4.code(output.getvalue())  # , height=300)

                    except Exception as e:
                        progress_bar.progress(
                            0, text=f":material/error: Error: {str(e)}"
                        )

                # Remove specific materials if needed
                if hasattr(X, "region_data"):
                    with st.container(border=True):
                        change_sm(
                            X, field_mean, sm_weighted_field_mean, N_weighted_field_mean
                        )

                    with st.container(border=True):
                        compare_uranos(X)

            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                # Clean up temporary file if it was created


Footprint_page = st.Page(
    Footprint_content,
    title="Footprint Analysis",
    icon=":material/barefoot:",
    default=True,
)
# Practical_page = st.Page(
#     Practical_content,
#     title="Practical Footprint",
#     icon=":material/barefoot:",
#     default=False,
# )
Signal_page = st.Page(
    Signal_content, title="Signal Contribution", icon=":material/globe:", default=False
)

pages = st.navigation([Footprint_page, Signal_page], position="top")
pages.run()
