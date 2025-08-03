import streamlit as st
import pandas as pd
import numpy as np
import requests 
import re

# logo
st.logo("assets/logo2.png", size='large')

home = st.Page(
    page = "views/home.py",
    title = "Home",
    icon = ":material/home:",
    default = True
)

pseudo_labeling = st.Page(
    page = "views/pseudo_labeling.py",
    title = "Pseudo Labeling",
    icon = ":material/label:",
    default = False
)

analysis = st.Page(
    page = "views/analysis.py",
    title = "Analysis",
    icon = ":material/analytics:",
    default = False
)

model_training = st.Page(
    page = "views/model_training.py",
    title = "Model Training",
    icon = ":material/network_intelligence_history:",
    default = False
)

evaluation = st.Page(
    page = "views/evaluation.py",
    title = "Evaluation",
    icon = ":material/check_circle:",
    default = False
)

contact = st.Page(
    page = "views/contact.py",
    title = "Contact Developers",
    icon = ":material/contact_support:",
    default = False
)

documentation = st.Page(
    page = "views/documentation.py",
    title = "Documentation",
    icon = ":material/book:",
    default = False
)

nav = st.navigation(pages=[home, pseudo_labeling, analysis, model_training, evaluation, contact, documentation])

nav.run()