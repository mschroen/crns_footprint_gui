import streamlit as st

st.set_page_config(
    layout="wide",
)

Footprint_page = st.Page(
    "pages/Footprint_Analysis.py",
    title="Footprint Analysis",
    icon=":material/barefoot:",
    default=True,
)

Signal_page = st.Page(
    "pages/Signal_Contributions.py",
    title="Signal Contribution",
    icon=":material/globe:",
    default=False,
)

Incoming_page = st.Page(
    "pages/Incoming_neutrons.py",
    title="Incoming neutrons",
    icon=":material/trending_down:",
    default=False,
)

Soil_page = st.Page(
    "pages/Soil_parameters.py",
    title="Soil parameters",
    icon=":material/landslide:",
    default=False,
)



Vertical_page = st.Page(
    "pages/Vertical_averaging.py",
    title="Vertical averaging",
    icon=":material/line_style:",
    default=False,
)

pages = st.navigation(
    {
        "Footprint": [Footprint_page, Signal_page, Vertical_page],
        "External": [Incoming_page, Soil_page],
    },
    position="top",
)
pages.run()
