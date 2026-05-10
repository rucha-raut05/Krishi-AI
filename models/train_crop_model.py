"""
Krishi AI - Crop Recommendation Model Training
Trains a Random Forest Classifier on crop data.
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), 'data', 'crop_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'crop_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'crop_scaler.pkl')
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, 'crop_label_encoder.pkl')

def train():
    print("=" * 60)
    print("  Krishi AI - Crop Recommendation Model Training")
    print("=" * 60)
    print("\nLoading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Dataset shape: {df.shape}")
    print(f"  Crops: {df['label'].nunique()} unique")

    soil_encoder = LabelEncoder()
    df['soil_type_encoded'] = soil_encoder.fit_transform(df['soil_type'])

    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'soil_type_encoded']
    X = df[feature_cols].values
    y = df['label'].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    print("\nTraining Random Forest...")

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy: {accuracy*100:.2f}%")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump({
        'label_encoder': label_encoder,
        'soil_encoder': soil_encoder,
        'feature_cols': feature_cols
    }, LABEL_ENCODER_PATH)

    print(f"\n  Model saved to: {MODEL_PATH}")
    print("  Training complete!")

if __name__ == '__main__':
    train()
