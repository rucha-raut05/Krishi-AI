"""
Krishi AI - Configuration File
Contains all application settings, API keys, and paths.
"""

import os

# Base directory of the application
BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    """Main configuration class for Krishi AI application."""

    # ─── Flask Settings ───────────────────────────────────────────────
    SECRET_KEY = os.environ.get('SECRET_KEY', 'krishi-ai-secret-key-change-in-production')
    DEBUG = True

    # ─── Database Settings ────────────────────────────────────────────
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{os.path.join(BASE_DIR, 'database', 'krishi.db')}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ─── Upload Settings ──────────────────────────────────────────────
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

    # ─── OpenWeatherMap API ───────────────────────────────────────────
    # Get your free API key at: https://openweathermap.org/api
    WEATHER_API_KEY = os.environ.get('WEATHER_API_KEY', 'YOUR_API_KEY_HERE')
    WEATHER_API_URL = 'https://api.openweathermap.org/data/2.5'

    # ─── Model Paths ─────────────────────────────────────────────────
    CROP_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'crop_model.pkl')
    CROP_SCALER_PATH = os.path.join(BASE_DIR, 'models', 'crop_scaler.pkl')
    CROP_LABEL_ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'crop_label_encoder.pkl')
    FERTILIZER_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'fertilizer_model.pkl')
    FERTILIZER_LABEL_ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'fertilizer_label_encoder.pkl')
    DISEASE_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'disease_model.h5')
    DISEASE_CLASSES_PATH = os.path.join(BASE_DIR, 'models', 'disease_classes.json')
