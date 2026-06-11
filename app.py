import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="ERC Rover Intelligence Dashboard",
    page_icon="🤖",
    layout="wide"
)

# basic sidebar nav
st.sidebar.title("ERC Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Go to",
    ["Overview", "Historical Analysis", "AI Prediction", "Score Trends"]
)

# just importing here to keep app.py clean
if page == "Overview":
    from pages import overview
    overview.show()
elif page == "Historical Analysis":
    from pages import historical
    historical.show()
elif page == "AI Prediction":
    from pages import prediction
    prediction.show()
elif page == "Score Trends":
    from pages import trends
    trends.show()
