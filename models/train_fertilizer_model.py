"""
Krishi AI - Fertilizer Recommendation Model Training
Trains a Random Forest Classifier for fertilizer recommendations.
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), 'data', 'fertilizer_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'fertilizer_model.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'fertilizer_label_encoder.pkl')

def train():
    print("=" * 60)
    print("  Krishi AI - Fertilizer Recommendation Model Training")
    print("=" * 60)
    print("\nLoading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Dataset shape: {df.shape}")
    print(f"  Fertilizers: {df['fertilizer'].nunique()} unique")

    soil_encoder = LabelEncoder()
    crop_encoder = LabelEncoder()
    fertilizer_encoder = LabelEncoder()

    df['soil_encoded'] = soil_encoder.fit_transform(df['soil_type'])
    df['crop_encoded'] = crop_encoder.fit_transform(df['crop'])
    df['fertilizer_encoded'] = fertilizer_encoder.fit_transform(df['fertilizer'])

    feature_cols = ['soil_encoded', 'crop_encoded', 'N', 'P', 'K']
    X = df[feature_cols].values
    y = df['fertilizer_encoded'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    print("\nTraining Random Forest...")

    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy: {accuracy*100:.2f}%")

    joblib.dump(model, MODEL_PATH)
    joblib.dump({
        'soil_encoder': soil_encoder,
        'crop_encoder': crop_encoder,
        'fertilizer_encoder': fertilizer_encoder,
        'feature_cols': feature_cols
    }, ENCODER_PATH)

    print(f"\n  Model saved to: {MODEL_PATH}")
    print("  Training complete!")

if __name__ == '__main__':
    train()
