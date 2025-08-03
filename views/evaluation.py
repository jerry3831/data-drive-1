import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, roc_curve, auc
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import numpy as np

st.header("Model Evaluation on External Test Data")
st.markdown("""Evaluate your trained model on an external test dataset. -Upload a CSV file containing the test data.""")  

if st.session_state['trained_model'] is None:
    st.info("Please train a model in the 'Model Training & Saving' tab first.")
elif st.session_state['selected_features'] is None:
    st.info("Please perform feature selection in the 'Model Training & Saving' tab first.")
elif st.session_state['pl'] is None:
        st.info("Pseudo-labeling configuration (including target column) is missing. Please run pseudo-labeling.")
else:
    # Get necessary components from session state
    trained_model = st.session_state['trained_model']
    selected_features = st.session_state['selected_features']
    target_col = getattr(st.session_state['pl'], 'target_col', None)
    pl_instance = st.session_state['pl'] # Get the PseudoLabeler instance

    if target_col is None:
        st.error("Target column information is missing. Cannot evaluate model.")
    else:
        st.subheader("1. Upload Test Dataset")
        uploaded_test_file = st.file_uploader("Upload a CSV file for testing", type=["csv"], key="test_file_uploader")

        if uploaded_test_file:
            test_df = pd.read_csv(uploaded_test_file)
            st.write("Preview of uploaded test dataset:", test_df.head())

            # Ensure target column exists in the test data
            if target_col not in test_df.columns:
                st.error(f"The target column '{target_col}' selected during pseudo-labeling is not found in the uploaded test dataset.")
            else:
                st.subheader("2. Apply Pseudo-Labeling (if needed) and Evaluate Model")
                if st.button("Apply Pseudo-Labeling & Evaluate Model", key="evaluate_model_button"):
                    with st.spinner("Applying pseudo-labeling (if needed) and evaluating model..."):
                        test_df_processed = test_df.copy()

                        # Check if the target column in the test data has missing values
                        if test_df_processed[target_col].isnull().any():
                            st.info(f"Missing values found in '{target_col}' of the test dataset. Applying pseudo-labeling...")
                            try:
                                # Apply pseudo-labeling to the test data
                                test_df_processed = pl_instance.fit(test_df_processed)
                                st.success("Pseudo-labeling applied to test dataset.")
                            except Exception as e:
                                st.error(f"An error occurred during pseudo-labeling of the test data: {e}")
                        else:
                            st.info(f"No missing values in '{target_col}' of the test dataset. Skipping pseudo-labeling.")

                        # Drop rows where target_col is still NaN (if pseudo-labeling didn't fill all, or if no pseudo-labeling applied)
                        initial_test_rows = test_df_processed.shape[0]
                        test_df_cleaned = test_df_processed.dropna(subset=[target_col]).copy()
                        if test_df_cleaned.shape[0] < initial_test_rows:
                            st.warning(f"Removed {initial_test_rows - test_df_cleaned.shape[0]} rows with missing target labels from the test set before evaluation.")
                        
                        if test_df_cleaned.empty:
                            st.error("No complete labeled data in the test set after processing and dropping NaNs in target column. Cannot evaluate model.")
                        else:
                            X_test_raw = test_df_cleaned.drop(columns=[target_col])
                            y_test = test_df_cleaned[target_col]

                            # Filter X_test_raw to only include selected features
                            missing_features_in_test = [f for f in selected_features if f not in X_test_raw.columns]
                            if missing_features_in_test:
                                st.error(f"The following selected features are missing in the uploaded test dataset: {missing_features_in_test}. "
                                            "Please ensure your test dataset contains all features used for training.")
                                #return # Stop evaluation if critical features are missing

                            X_test_filtered = X_test_raw[selected_features]

                            if X_test_filtered.empty:
                                st.error("No features available in the test dataset after filtering by selected features.")
                            else:
                                try:
                                    y_pred = trained_model.predict(X_test_filtered)
                                    
                                    y_proba = None
                                    if hasattr(trained_model, 'predict_proba'):
                                        y_proba = trained_model.predict_proba(X_test_filtered)

                                    accuracy = accuracy_score(y_test, y_pred)
                                    report = classification_report(y_test, y_pred, output_dict=True)

                                    st.success("Model evaluation complete!")

                                    st.subheader("Model Performance Results")
                                    # Display accuracy as a clear, large percentage
                                    st.markdown(f"**<p style='font-size:24px;'>Accuracy: <span style='color:green;'>{accuracy:.2%}</span></p>**", unsafe_allow_html=True)
                                    
                                    st.write("Classification Report:")
                                    # Improved display for classification report
                                    report_df = pd.DataFrame(report).transpose()
                                    st.dataframe(report_df.style.format("{:.2f}"))

                                    # Create two columns for Confusion Matrix and ROC Curve
                                    col_cm, col_roc = st.columns(2)

                                    with col_cm:
                                        st.subheader("Confusion Matrix")
                                        # Improved display for confusion matrix
                                        fig_cm, ax_cm = plt.subplots(figsize=(7, 6)) # Adjusted size for side-by-side
                                        cm_display = ConfusionMatrixDisplay.from_predictions(
                                            y_test, y_pred, 
                                            cmap='Blues', 
                                            ax=ax_cm, 
                                            values_format='d',
                                            display_labels=np.unique(y_test) # Use actual labels
                                        )
                                        cm_display.plot(ax=ax_cm, cmap='Blues', values_format='d')
                                        st.pyplot(fig_cm)

                                    with col_roc:
                                        st.subheader("ROC Curve (Multiclass or Binary)")
                                        # Handle multiclass ROC
                                        unique_classes = np.unique(y_test)
                                        if len(unique_classes) > 2 and y_proba is not None:
                                            st.write("Displaying One-vs-Rest (OvR) ROC curves for each class.")
                                            
                                            # Binarize the output
                                            y_test_binarized = label_binarize(y_test, classes=unique_classes)
                                            
                                            fig_roc, ax_roc = plt.subplots(figsize=(7, 6)) # Adjusted size for side-by-side
                                            colors = plt.cm.get_cmap('tab10', len(unique_classes)) # Get distinct colors

                                            for i, class_label in enumerate(unique_classes):
                                                fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_proba[:, i])
                                                roc_auc = auc(fpr, tpr)
                                                ax_roc.plot(fpr, tpr, color=colors(i), lw=2,
                                                            label=f'ROC curve of class {class_label} (area = {roc_auc:.2f})')

                                            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                                            ax_roc.set_xlim([0.0, 1.0])
                                            ax_roc.set_ylim([0.0, 1.05])
                                            ax_roc.set_xlabel('False Positive Rate')
                                            ax_roc.set_ylabel('True Positive Rate')
                                            ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve - One-vs-Rest')
                                            ax_roc.legend(loc="lower right")
                                            st.pyplot(fig_roc)

                                        elif len(unique_classes) == 2 and y_proba is not None:
                                            st.write("Displaying ROC curve for binary classification.")
                                            # For binary, y_proba[:, 1] is the probability of the positive class
                                            fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])
                                            roc_auc = auc(fpr, tpr)

                                            fig_roc, ax_roc = plt.subplots(figsize=(7, 6)) # Adjusted size for side-by-side
                                            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                                            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                                            ax_roc.set_xlim([0.0, 1.0])
                                            ax_roc.set_ylim([0.0, 1.05])
                                            ax_roc.set_xlabel('False Positive Rate')
                                            ax_roc.set_ylabel('True Positive Rate')
                                            ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
                                            ax_roc.legend(loc="lower right")
                                            st.pyplot(fig_roc)
                                        else:
                                            st.info("ROC Curve requires model probabilities (`predict_proba`) and at least two classes.")
                                except Exception as e:
                                    st.error(f"An error occurred during model evaluation: {e}")
        else:
            st.info("Please upload a test CSV file to evaluate the model.")
