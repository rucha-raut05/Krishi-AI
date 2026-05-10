# 🌾 Krishi AI – Smart Farming Assistant

An AI-powered web application that helps farmers make data-driven decisions about crops, fertilizers, plant diseases, and weather conditions.

## ✨ Features

- **🌱 Crop Recommendation** – ML-based crop suggestions based on soil & climate data
- **🧪 Fertilizer Recommendation** – Smart fertilizer advice based on soil nutrients
- **🔬 Plant Disease Detection** – Upload plant images to detect diseases via CNN
- **🌤️ Weather Integration** – Real-time weather data with farming tips
- **🤖 AI Chatbot** – Ask farming questions and get instant answers
- **🎤 Voice Support** – Speech-to-text input & text-to-speech output
- **📊 History Tracking** – View all past queries and recommendations
- **🔐 User Authentication** – Secure signup/login with session management

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5, CSS3, JavaScript (ES6+) |
| Backend | Python 3.9+, Flask |
| Database | SQLite |
| ML Models | Scikit-learn (Random Forest) |
| Deep Learning | TensorFlow / Keras (MobileNetV2) |
| APIs | OpenWeatherMap |

## 🚀 Setup Instructions

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- Git (optional)

### Step 1: Clone / Download the Project
```bash
cd "Krishi-ai full"
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Set Up API Key
1. Go to [OpenWeatherMap](https://openweathermap.org/api) and sign up for a free API key
2. Open `config.py` and replace `YOUR_API_KEY_HERE` with your key:
   ```python
   WEATHER_API_KEY = 'your-actual-api-key'
   ```
   Or set it as an environment variable:
   ```bash
   set WEATHER_API_KEY=your-actual-api-key  # Windows
   export WEATHER_API_KEY=your-actual-api-key  # Linux/macOS
   ```

### Step 5: Train the ML Models
```bash
python models/train_crop_model.py
python models/train_fertilizer_model.py
python models/train_disease_model.py
```

### Step 6: Run the Application
```bash
python app.py
```

### Step 7: Open in Browser
Navigate to: **http://localhost:5000**

## 📁 Project Structure

```
Krishi-ai full/
├── app.py                    # Main Flask application
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── models/                   # ML model training & saved models
├── data/                     # Datasets and reference data
├── database/                 # SQLite database (auto-created)
├── static/                   # CSS, JS, and images
├── templates/                # HTML templates
└── uploads/                  # User-uploaded images
```

## 📝 Notes

- The disease detection model included is a demo/mock model. For production use, train on the full [PlantVillage dataset](https://www.kaggle.com/emmarex/plantdisease).
- The chatbot uses rule-based NLP. For more advanced responses, integrate with an LLM API.
- Always use a strong `SECRET_KEY` in production.

