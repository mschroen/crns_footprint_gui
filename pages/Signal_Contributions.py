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
    from lib.uranos import URANOS

except ImportError as e:
    st.error(f"Missing or problematic packages: {e}")


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


# def Signal_content():
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
                    progress_bar.progress(0, text=f":material/error: Error: {str(e)}")

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

show_footer()
