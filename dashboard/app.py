"""pharma-4 dashboard.

streamlit run dashboard/app.py
"""

import streamlit as st

from dashboard import views

st.set_page_config(page_title="pharma-4", layout="wide")
views.sidebar()
st.navigation(
    [
        st.Page(views.page_live, title="Live", url_path="live", default=True),
        st.Page(views.page_alerts, title="Alerts", url_path="alerts"),
        st.Page(views.page_yield, title="Yield", url_path="yield"),
        st.Page(views.page_graph, title="Graph", url_path="graph"),
        st.Page(views.page_uns, title="UNS browser", url_path="uns"),
    ]
).run()
