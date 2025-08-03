import streamlit as st
from models.pseudo_gen2 import PseudoLabeler
import pandas as pd

#TODO: Initializing session state variables
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'pl' not in st.session_state:
    st.session_state['pl'] = None
if 'combined' not in st.session_state:
    st.session_state['combined'] = None
if 'trained_model' not in st.session_state:
    st.session_state['trained_model'] = None
if 'selected_features' not in st.session_state:
    st.session_state['selected_features'] = None
if 'preprocessor' not in st.session_state:
    st.session_state['preprocessor'] = None

st.header("Pseudo-Labeling Configuration")
st.markdown("""Configure the pseudo-labeling process by uploading a dataset and selecting the target variable with missing labels.""")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv", "xls", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state['df'] = df # Store original df in session state
    st.write("Preview of uploaded dataset:", df.head())
elif st.session_state['df'] is not None:
    df = st.session_state['df'] # Use already uploaded df from session state
    st.write("Using previously uploaded dataset (preview):", df.head())
else:
    st.info("Please upload a CSV file to proceed with pseudo-labeling.")
    df = None # Ensure df is None if no file is uploaded or in session state

if df is not None:
    target_col = st.selectbox("Select Target Column", df.columns, key="target_col_tab1")
    threshold = st.slider("Confidence Threshold for Pseudo-Labeling", 0.5, 1.0, 0.9, 0.01, key="threshold_tab1")
    run_button = st.button("Run Pseudo-Labeling", key="run_button_tab1")

    if run_button:
        # Ensure the target column is suitable for pseudo-labeling (e.g., has NaNs)
        if target_col not in df.columns:
            st.error(f"Selected target column '{target_col}' not found in the dataset.")
        else:
            with st.spinner("Performing pseudo-labeling..."): # Added loading animation here
                # Instantiate PseudoLabeler and run fit
                # Your actual PseudoLabeler should handle the training and pseudo-labeling logic
                pl = PseudoLabeler(target_col=target_col, threshold=threshold)
                
                # Assuming your PseudoLabeler's fit method returns the combined DataFrame
                try:
                    combined = pl.fit(df) 
                except Exception as e:
                    st.error(f"An error occurred during pseudo-labeling: {e}")
                    combined = None # Set combined to None on error

                if combined is not None:
                    st.success("Pseudo-labeling completed.")
                    st.write("Combined Data Preview (with pseudo-labels):", combined.head())
                    st.write(f"**Shape of DataFrame after pseudo-labeling:** {combined.shape}")

                    # Calculate and display the number of newly assigned labels
                    # This assumes pl.last_confidences stores the confidences for newly assigned labels
                    if hasattr(pl, 'last_confidences') and pl.last_confidences is not None:
                        newly_assigned_labels = len(pl.last_confidences)
                        st.write(f"**Number of newly assigned labels:** {newly_assigned_labels}")
                    else:
                        st.info("Could not determine the exact number of newly assigned labels.")
                    # Save to session state
                    st.session_state['pl'] = pl
                    st.session_state['combined'] = combined
                    # Reset model and features if new pseudo-labeling is run
                    st.session_state['trained_model'] = None
                    st.session_state['selected_features'] = None
                    st.session_state['preprocessor'] = None
                    # No need to reset evaluation metrics here as they are calculated on uploaded test data

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### 📥 Download Combined Data")
                        st.download_button(
                            label="Download Combined Data",
                            data=combined.to_csv(index=False).encode('utf-8'),
                            file_name='combined_data.csv',
                            mime='text/csv',
                            key="download_button_tab1"
                        )
                    with col2:
                        st.markdown("#### 📊 View analysis")
                        st.button("Analysis", on_click=lambda: st.switch_page("views/analysis.py"), key="analysis_button_tab1")
                else:
                    st.warning("Pseudo-labeling did not complete successfully. Check for errors above.")
else:
    st.warning("Upload a CSV file to configure pseudo-labeling.")