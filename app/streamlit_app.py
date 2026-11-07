"""Recovery Lab: does good prediction mean good statistics?

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import streamlit as st

import ui

st.set_page_config(
    page_title="Recovery Lab · Missing-data study",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)
ui.inject_css()

pages = [
    st.Page("views/home.py", title="Home", icon=":material/home:", default=True),
    st.Page("views/lab.py", title="Live Lab", icon=":material/science:", url_path="lab"),
    st.Page("views/explorer.py", title="Results Explorer", icon=":material/scatter_plot:", url_path="explorer"),
    st.Page("views/method.py", title="Method & About", icon=":material/account_tree:", url_path="method"),
]
st.navigation(pages, position="top").run()
