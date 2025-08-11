import streamlit as st
import sys
import contextlib
import io


@contextlib.contextmanager
def capture_stdout():
    old_stdout = sys.stdout
    stdout_capture = io.StringIO()
    try:
        sys.stdout = stdout_capture
        yield stdout_capture
    finally:
        sys.stdout = old_stdout


def show_footer():
    c1, c2 = st.columns(2)
    c2.image("assets/neptoon-logo.svg", "Made with <3 and Neptoon")


def show_footer_nmdb():
    c1, c2, c3 = st.columns(3)
    c2.image("assets/neptoon-logo.svg", "Made with <3 and Neptoon")
    c3.image("assets/nmdb.png", "Supported by NMDB.eu")
