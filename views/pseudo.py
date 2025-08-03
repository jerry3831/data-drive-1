import streamlit as st
import pandas as pd
from models.pseudo_gen2 import PseudoLabeler

# -------------------------------
# Caching Functions
# -------------------------------

@st.cache_data(show_spinner="Reading uploaded file...")
def load_file(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file format")


@st.cache_data(show_spinner="Running pseudo-labeling...")
def run_pseudo_labeling(df, target_col, threshold):
    pl = PseudoLabeler(target_col=target_col, threshold=threshold)
    combined = pl.fit(df)
    return combined, pl


# -------------------------------
# Session State Initialization
# -------------------------------

defaults = {
    "df": None,
    "pl": None,
    "combined": None,
    "trained_model": None,
    "selected_features": None,
    "preprocessor": None
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# -------------------------------
# UI Layout
# -------------------------------

st.header("Pseudo-Labeling Configuration")
st.markdown("Configure the pseudo-labeling process by uploading a dataset and selecting the target variable with missing labels.")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xls", "xlsx"])

if uploaded_file:
    try:
        df = load_file(uploaded_file)
        st.session_state["df"] = df
        st.write("Preview of uploaded dataset:", df.head())
    except Exception as e:
        st.error(f"Error loading file: {e}")
        df = None
elif st.session_state["df"] is not None:
    df = st.session_state["df"]
    st.write("Using previously uploaded dataset (preview):", df.head())
else:
    st.info("Please upload a file to proceed.")
    df = None

# -------------------------------
# Pseudo-labeling Setup
# -------------------------------

if df is not None:
    target_col = st.selectbox("Select Target Column", df.columns, key="target_col_tab1")
    threshold = st.slider("Confidence Threshold for Pseudo-Labeling", 0.5, 1.0, 0.9, 0.01, key="threshold_tab1")

    if st.button("Run Pseudo-Labeling", key="run_button_tab1"):
        if target_col not in df.columns:
            st.error("Selected target column not found.")
        else:
            try:
                combined, pl = run_pseudo_labeling(df, target_col, threshold)
                st.success("Pseudo-labeling completed.")
                st.session_state["combined"] = combined
                st.session_state["pl"] = pl

                # Clear downstream state
                st.session_state["trained_model"] = None
                st.session_state["selected_features"] = None
                st.session_state["preprocessor"] = None

                # Preview results
                st.write("Combined Data Preview:", combined.head())
                st.write(f"Shape after pseudo-labeling: {combined.shape}")
                if hasattr(pl, "last_confidences") and pl.last_confidences is not None:
                    st.write(f"Number of newly assigned labels: {len(pl.last_confidences)}")
                else:
                    st.info("No label confidence data available.")

                # Options: Download or navigate
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download Combined Data",
                        data=combined.to_csv(index=False).encode("utf-8"),
                        file_name="combined_data.csv",
                        mime="text/csv",
                        key="download_button_tab1"
                    )
                with col2:
                    st.button("📊 Go to Analysis", on_click=lambda: st.switch_page("views/analysis.py"), key="analysis_button_tab1")

            except Exception as e:
                st.error(f"An error occurred during pseudo-labeling: {e}")
else:
    st.warning("Upload a file to configure pseudo-labeling.")
