import streamlit as st
#? default page layout
st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
import requests 
import re
from pathlib import Path

# Sidebar logo
img_path = Path("assets/image.jpg")
if img_path.exists():
    st.sidebar.image(str(img_path), use_container_width=True)
else:
    st.sidebar.warning("Logo image not found.")

home = st.Page(
    page = "pages/home.py",
    title = "Home",
    icon = ":material/home:",
    default = True
)

pseudo_labeling = st.Page(
    page = "pages/pseudo.py",
    title = "Pseudo Labeling",
    icon = ":material/label:"
)

analysis = st.Page(
    page = "pages/analysis.py",
    title = "Analysis",
    icon = ":material/analytics:",
    default = False
)

model_training = st.Page(
    page = "pages/model_training.py",
    title = "Model Training",
    icon = ":material/network_intelligence_history:",
    default = False
)

evaluation = st.Page(
    page = "pages/evaluation.py",
    title = "Evaluation",
    icon = ":material/check_circle:",
    default = False
)

nav = st.navigation(pages=[home, pseudo_labeling, analysis, model_training, evaluation])

nav.run()