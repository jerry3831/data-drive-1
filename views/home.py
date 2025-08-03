import streamlit as st
from views import pseudo_labeling

# Set page config
st.set_page_config(
    page_title="Welcome | Pseudo Labeling App",
    page_icon=":material/in_home_mode:",
    layout="wide",
)

# Use query params to navigate to other pages
def navigate_to(tab_name):
    st.experimental_set_query_params(tab=tab_name)  # Correct method

# Title and Welcome
st.title("SSL Pseudo Labeling App")

st.markdown("""
Welcome to an interactive application for **Semi-Supervised Learning** using **Pseudo Labeling**. This tool allows you to label data using a machine learning model, analyze patterns, and evaluate results — all in one place!
""")

st.markdown("---")
st.subheader("🧩 App Structure and Overview")

# First row: Pseudo Labeling | Analysis
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📌 1. Pseudo Labeling")
    st.markdown("""
    - Upload and view your dataset  
    - Select a confidence threshold for labeling the data  
    - Generate pseudo labels for your unlabeled data
    """)

with col2:
    st.markdown("#### 📊 2. Analysis")
    st.markdown("""
    - Visualize distributions, label confidence, and class balance  
    - Identify patterns, outliers, or uncertainties in pseudo-labeled data
    """)

# Second row: Model Training | Evaluation
col3, col4 = st.columns(2)

with col3:
    st.markdown("#### 🖥 3. Model Training")
    st.markdown("""
    - Select important features for modeling and predictions  
    - Choose between different algorithms and architectures  
    - Monitor training progress and adjust parameters as needed
    """)

with col4:
    st.markdown("#### ✅ 4. Evaluation")
    st.markdown("""
    - Precision, Recall, F1-score, and Confusion Matrix  
    - Comparisons between labeled, pseudo-labeled, and ground truth data (if available)
    """)

st.markdown("---")

# Get Started Button
if st.button("🚀 Get Started with Pseudo Labeling"):
    st.switch_page("views/pseudo_labeling.py")  # Navigate to pseudo-labeling page

# Footer Note
st.markdown("""
<small><i>Tip: Use the sidebar to navigate through the app at any time.</i></small>
""", unsafe_allow_html=True)
