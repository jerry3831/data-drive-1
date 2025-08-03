import streamlit as st
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2

st.header("Model Training and Saving")

if st.session_state['combined'] is not None and st.session_state['pl'] is not None:
    combined_data = st.session_state['combined'].copy() # Work on a copy
    target_col = getattr(st.session_state['pl'], 'target_col', None)

    if target_col is None or target_col not in combined_data.columns:
        st.error("Target column not found in the combined data. Please re-run pseudo-labeling.")
    else:
        st.write("Combined data (with pseudo-labels) is available for model training.")
        st.dataframe(combined_data.head())

        # Drop rows where target_col is NaN (if any pseudo-labels were not assigned)
        initial_rows = combined_data.shape[0]
        combined_data.dropna(subset=[target_col], inplace=True)
        if combined_data.shape[0] < initial_rows:
            st.warning(f"Removed {initial_rows - combined_data.shape[0]} rows with missing target labels before training.")
        
        if combined_data.empty:
            st.error("No complete labeled data available for training after dropping NaNs in target column.")
        else:
            # Separate features (X) and target (y)
            X = combined_data.drop(columns=[target_col])
            y = combined_data[target_col]

            # Identify numerical and categorical features
            numerical_features = X.select_dtypes(include=np.number).columns.tolist()
            categorical_features = X.select_dtypes(include='object').columns.tolist()

            st.subheader("1. Feature Selection (Chi-squared)")
            st.write("Select the number of top features to keep using Chi-squared test. "
                        "This is suitable for categorical features and a categorical target.")
            
            if not categorical_features:
                st.info("No categorical features found for Chi-squared feature selection.")
                st.session_state['selected_features'] = X.columns.tolist() # Keep all if no categorical
            else:
                k_features = st.slider(
                    "Number of top features to select (Chi-squared)",
                    min_value=1,
                    max_value=len(categorical_features),
                    value=min(5, len(categorical_features)), # Default to 5 or max available
                    key="k_features_slider"
                )
                
                if st.button("Perform Feature Selection", key="run_feature_selection"):
                    with st.spinner("Performing Chi-squared feature selection..."):
                        # Preprocess categorical features for Chi2 (e.g., Label Encoding)
                        X_cat = X[categorical_features].apply(lambda col: LabelEncoder().fit_transform(col) if col.dtype == 'object' else col)

                        # Ensure target is also numerical for Chi2
                        y_encoded = LabelEncoder().fit_transform(y)

                        # Apply SelectKBest with chi2
                        selector = SelectKBest(chi2, k=k_features)
                        selector.fit(X_cat, y_encoded)

                        selected_feature_indices = selector.get_support(indices=True)
                        selected_chi2_features = [categorical_features[i] for i in selected_feature_indices]
                        
                        st.success(f"Selected {len(selected_chi2_features)} features using Chi-squared.")
                        st.write("Top features selected:", selected_chi2_features)
                        
                        # Combine selected categorical with all numerical features
                        st.session_state['selected_features'] = list(set(selected_chi2_features + numerical_features))
                        st.info(f"Features for model training: {st.session_state['selected_features']}")
                else:
                    if st.session_state['selected_features'] is None:
                        st.info("Click 'Perform Feature Selection' to select features, or training will use all features.")
                        st.session_state['selected_features'] = X.columns.tolist() # Default to all features if not run
                    else:
                        st.info(f"Current features for training: {st.session_state['selected_features']}")

            # Prepare data for model training using selected features
            features_to_use = st.session_state['selected_features']
            X_model = X[features_to_use]

            # Preprocessing for the model
            numerical_transformer = Pipeline(steps=[
                ('imputer', st.session_state['pl'].numerical_imputer if hasattr(st.session_state['pl'], 'numerical_imputer') else 'passthrough'), # Use imputer from PL if available
                ('scaler', st.session_state['pl'].numerical_scaler if hasattr(st.session_state['pl'], 'numerical_scaler') else 'passthrough') # Use scaler from PL if available
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', st.session_state['pl'].categorical_imputer if hasattr(st.session_state['pl'], 'categorical_imputer') else 'passthrough'), # Use imputer from PL if available
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            # Filter features based on what's actually in X_model
            numerical_features_model = [f for f in numerical_features if f in X_model.columns]
            categorical_features_model = [f for f in categorical_features if f in X_model.columns]

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features_model),
                    ('cat', categorical_transformer, categorical_features_model)
                ],
                remainder='passthrough' # Keep other columns not explicitly transformed
            )
            st.session_state['preprocessor'] = preprocessor # Store preprocessor

            st.subheader("2. Train Model on All Data")
            st.write("The model will be trained on the entire combined dataset (after pseudo-labeling and dropping NaNs).")

            if st.button("Train Model", key="train_model_button"): # Changed button label
                with st.spinner("Training model on all available data..."):
                    # Define the model pipeline
                    model_pipeline = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(random_state=42)) # Using RandomForest for robustness
                    ])
                    
                    # Train the model on ALL available data (X_model, y)
                    model_pipeline.fit(X_model, y)
                    st.session_state['trained_model'] = model_pipeline

                    st.success("Model training complete on the entire combined dataset!")
                    st.info("Now, upload a separate test dataset in the 'Model Evaluation' tab to evaluate the trained model.")


            st.subheader("3. Save Trained Model")
            if st.session_state['trained_model'] is not None:
                st.write("Save the trained model as a `.pkl` file.")
                
                # Serialize the model using pickle
                pickled_model = pickle.dumps(st.session_state['trained_model'])

                st.download_button(
                    label="Download Trained Model (.pkl)",
                    data=pickled_model,
                    file_name="trained_pseudo_label_model.pkl",
                    mime="application/octet-stream",
                    key="download_model_button"
                )
            else:
                st.info("Train a model first to enable saving.")

else:
    st.info("Please run the pseudo-labeling process in the 'Pseudo-Labeling' tab first to enable model training.")
