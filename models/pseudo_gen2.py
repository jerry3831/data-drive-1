import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

class PseudoLabeler:
    def __init__(self, target_col, model=None, threshold=0.9, random_state=42):
        self.target_col = target_col
        self.threshold = threshold
        self.random_state = random_state

        if model is None:
            model = RandomForestClassifier(random_state=random_state)
        self.model = model

        self.label_encoder = None 
        self.feature_encoders = {}
        self.combined_df = None
        self.is_fitted = False
        self.last_confidences = None

    def encode_features(self, df):
        df_encoded = df.copy()
        for col in df.columns:
            if df_encoded[col].dtype == 'object' or str(df_encoded[col].dtype) == 'category':
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.feature_encoders[col] = le
        return df_encoded

    def transform_features(self, df):
        df_encoded = df.copy()
        for col, le in self.feature_encoders.items():
            if col in df_encoded.columns:
                df_encoded[col] = le.transform(df_encoded[col].astype(str))
        return df_encoded

    def fit(self, df):
        df = df.copy()

        #? Split into labeled and unlabeled
        labeled_df = df[df[self.target_col].notna()].copy()
        unlabeled_df = df[df[self.target_col].isna()].drop(columns=[self.target_col]).copy()

        #TODO: Encode target
        self.label_encoder = LabelEncoder()
        labeled_df[self.target_col] = self.label_encoder.fit_transform(labeled_df[self.target_col].astype(str))

        #? Encode categorical features
        X_train = labeled_df.drop(columns=[self.target_col])
        X_train = self.encode_features(X_train)
        y_train = labeled_df[self.target_col]

        X_unlabeled = self.transform_features(unlabeled_df)

        self.model.fit(X_train, y_train)

        #? Predict and filter by threshold
        probs = self.model.predict_proba(X_unlabeled)
        preds = np.argmax(probs, axis=1)
        max_probs = np.max(probs, axis=1)
        confident_idx = np.where(max_probs >= self.threshold)[0]
        pseudo_labels = preds[confident_idx]

        #? Create pseudo-labeled data
        pseudo_df = unlabeled_df.iloc[confident_idx].copy()
        pseudo_df = self.transform_features(pseudo_df)
        pseudo_df[self.target_col] = pseudo_labels

        #* storing last_confidences
        self.last_confidences = max_probs[confident_idx]

        #? Combine and decode
        combined_encoded = pd.concat([X_train.assign(**{self.target_col: y_train}), pseudo_df], ignore_index=True)
        combined_encoded[self.target_col] = self.label_encoder.inverse_transform(combined_encoded[self.target_col])
        for col, encoder in self.feature_encoders.items():
            if col in combined_encoded.columns:
                combined_encoded[col] = encoder.inverse_transform(combined_encoded[col])

        self.combined_df = combined_encoded
        self.is_fitted = True
        return self.combined_df
    
    #? returning confidence data
    def get_last_confidences(self):
        if self.last_confidences is None:
            raise RuntimeError("No confidences available. Fit the model first.")
        return self.last_confidences

    #? Option to use trained model to make predictions with all features
    def predict(self, df):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first.")

        df_transformed = self.transform_features(df.copy())
        preds = self.model.predict(df_transformed)
        return self.label_encoder.inverse_transform(preds)

    #? Option to save the model using joblib
    def save(self, filepath_prefix):
        joblib.dump(self.model, f"{filepath_prefix}_model.pkl")
        joblib.dump(self.label_encoder, f"{filepath_prefix}_target_encoder.pkl")
        joblib.dump(self.feature_encoders, f"{filepath_prefix}_feature_encoders.pkl")
        joblib.dump(self.combined_df, f"{filepath_prefix}_combined_data.pkl")

    #? option to load the saved model after running P-L
    def load(self, filepath_prefix):
        self.model = joblib.load(f"{filepath_prefix}_model.pkl")
        self.label_encoder = joblib.load(f"{filepath_prefix}_target_encoder.pkl")
        self.feature_encoders = joblib.load(f"{filepath_prefix}_feature_encoders.pkl")
        self.combined_df = joblib.load(f"{filepath_prefix}_combined_data.pkl")
        self.is_fitted = True
