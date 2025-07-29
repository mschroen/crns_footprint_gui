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

pages = st.navigation(
    {"Footprint": [Footprint_page, Signal_page], "Corrections": [Incoming_page]},
    position="top",
)
pages.run()
