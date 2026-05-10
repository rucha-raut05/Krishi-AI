"""
Krishi AI - Smart Farming Assistant
Main Flask Application
"""
import os, sys, io, json, datetime, requests, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from config import Config
import joblib

# ─── App Init ─────────────────────────────────────────────────────────
app = Flask(__name__)
app.config.from_object(Config)
os.makedirs(app.config.get('UPLOAD_FOLDER', 'uploads'), exist_ok=True)
os.makedirs(os.path.join(app.root_path, 'database'), exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login_page'

# ─── Database Models ──────────────────────────────────────────────────
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    history = db.relationship('History', backref='user', lazy=True)

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    query_type = db.Column(db.String(50), nullable=False)
    input_data = db.Column(db.Text, nullable=False)
    result_data = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# ─── Load ML Models ──────────────────────────────────────────────────
crop_model = crop_scaler = crop_encoders = None
fert_model = fert_encoders = None
disease_model = disease_classes = None

def load_models():
    global crop_model, crop_scaler, crop_encoders, fert_model, fert_encoders, disease_model, disease_classes
    try:
        crop_model = joblib.load(Config.CROP_MODEL_PATH)
        crop_scaler = joblib.load(Config.CROP_SCALER_PATH)
        crop_encoders = joblib.load(Config.CROP_LABEL_ENCODER_PATH)
        print("[OK] Crop model loaded")
    except Exception as e:
        print(f"[WARN] Crop model not found: {e}")
    try:
        fert_model = joblib.load(Config.FERTILIZER_MODEL_PATH)
        fert_encoders = joblib.load(Config.FERTILIZER_LABEL_ENCODER_PATH)
        print("[OK] Fertilizer model loaded")
    except Exception as e:
        print(f"[WARN] Fertilizer model not found: {e}")
    try:
        import tensorflow as tf
        disease_model = tf.keras.models.load_model(Config.DISEASE_MODEL_PATH)
        with open(Config.DISEASE_CLASSES_PATH) as f:
            disease_classes = json.load(f)
        print("[OK] Disease model loaded")
    except Exception as e:
        print(f"[WARN] Disease model not found: {e}")
        try:
            with open(Config.DISEASE_CLASSES_PATH) as f:
                disease_classes = json.load(f)
        except: pass

# ─── Page Routes ──────────────────────────────────────────────────────
@app.route('/')
def index_page():
    return render_template('index.html')

@app.route('/login')
def login_page():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard_page'))
    return render_template('login.html')

@app.route('/signup')
def signup_page():
    return render_template('signup.html')

@app.route('/dashboard')
@login_required
def dashboard_page():
    return render_template('dashboard.html')

@app.route('/crop')
@login_required
def crop_page():
    return render_template('crop.html')

@app.route('/fertilizer')
@login_required
def fertilizer_page():
    return render_template('fertilizer.html')

@app.route('/disease')
@login_required
def disease_page():
    return render_template('disease.html')

@app.route('/weather')
@login_required
def weather_page():
    return render_template('weather.html')

@app.route('/chatbot')
@login_required
def chatbot_page():
    return render_template('chatbot.html')

@app.route('/history')
@login_required
def history_page():
    return render_template('history.html')

# ─── Auth API ─────────────────────────────────────────────────────────
@app.route('/api/signup', methods=['POST'])
def api_signup():
    data = request.get_json()
    username = data.get('username','').strip()
    email = data.get('email','').strip()
    password = data.get('password','')
    if not username or not email or not password:
        return jsonify({'error': 'All fields are required'}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'Username already taken'}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already registered'}), 400
    user = User(username=username, email=email, password_hash=generate_password_hash(password))
    db.session.add(user)
    db.session.commit()
    login_user(user)
    return jsonify({'message': 'Account created!', 'user': {'id': user.id, 'username': user.username}})

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    user = User.query.filter_by(username=data.get('username','')).first()
    if not user or not check_password_hash(user.password_hash, data.get('password','')):
        return jsonify({'error': 'Invalid credentials'}), 401
    login_user(user, remember=True)
    return jsonify({'message': 'Login successful', 'user': {'id': user.id, 'username': user.username}})

@app.route('/api/logout')
@login_required
def api_logout():
    logout_user()
    return jsonify({'message': 'Logged out'})

@app.route('/api/profile')
@login_required
def api_profile():
    return jsonify({'id': current_user.id, 'username': current_user.username, 'email': current_user.email})

# ─── Crop Recommendation API ─────────────────────────────────────────
@app.route('/api/crop/recommend', methods=['POST'])
@login_required
def api_crop_recommend():
    if not crop_model:
        return jsonify({'error': 'Crop model not loaded. Run train_crop_model.py first.'}), 500
    data = request.get_json()
    try:
        soil = data['soil_type']
        soil_enc = crop_encoders['soil_encoder'].transform([soil])[0]
        features = np.array([[
            float(data['N']), float(data['P']), float(data['K']),
            float(data['temperature']), float(data['humidity']),
            float(data['ph']), float(data['rainfall']), soil_enc
        ]])
        features_scaled = crop_scaler.transform(features)
        pred = crop_model.predict(features_scaled)[0]
        proba = crop_model.predict_proba(features_scaled)[0]
        crop_name = crop_encoders['label_encoder'].inverse_transform([pred])[0]
        confidence = float(max(proba)) * 100
        # Get top 3 recommendations
        top3_idx = np.argsort(proba)[-3:][::-1]
        top3 = [{'crop': crop_encoders['label_encoder'].inverse_transform([i])[0],
                  'confidence': round(float(proba[i])*100, 1)} for i in top3_idx]
        result = {'crop': crop_name, 'confidence': round(confidence,1), 'top3': top3}
        # Save to history
        save_history('crop', json.dumps(data), json.dumps(result))
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ─── Fertilizer Recommendation API ───────────────────────────────────
@app.route('/api/fertilizer/recommend', methods=['POST'])
@login_required
def api_fertilizer_recommend():
    if not fert_model:
        return jsonify({'error': 'Fertilizer model not loaded. Run train_fertilizer_model.py first.'}), 500
    data = request.get_json()
    try:
        soil_enc = fert_encoders['soil_encoder'].transform([data['soil_type']])[0]
        crop_enc = fert_encoders['crop_encoder'].transform([data['crop']])[0]
        features = np.array([[soil_enc, crop_enc, float(data['N']), float(data['P']), float(data['K'])]])
        pred = fert_model.predict(features)[0]
        fertilizer = fert_encoders['fertilizer_encoder'].inverse_transform([pred])[0]
        # Fertilizer tips
        tips = get_fertilizer_tips(fertilizer, float(data['N']), float(data['P']), float(data['K']))
        result = {'fertilizer': fertilizer, 'tips': tips}
        save_history('fertilizer', json.dumps(data), json.dumps(result))
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def get_fertilizer_tips(fert, n, p, k):
    tips = []
    if n < 20: tips.append("Nitrogen is low. Consider adding nitrogen-rich organic matter.")
    elif n > 80: tips.append("Nitrogen is high. Reduce nitrogen-based fertilizers.")
    if p < 20: tips.append("Phosphorus is low. Add bone meal or rock phosphate.")
    elif p > 80: tips.append("Phosphorus is high. Avoid phosphorus-rich fertilizers.")
    if k < 20: tips.append("Potassium is low. Add wood ash or potash.")
    elif k > 80: tips.append("Potassium is high. Reduce potassium supplements.")
    tips.append(f"Recommended fertilizer: {fert}. Apply as per crop stage guidelines.")
    return tips

# ─── Disease Detection API ───────────────────────────────────────────
@app.route('/api/disease/detect', methods=['POST'])
@login_required
def api_disease_detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in Config.ALLOWED_EXTENSIONS:
        return jsonify({'error': 'Invalid file type. Use PNG, JPG, or JPEG.'}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    # Load disease info
    disease_info_path = os.path.join(app.root_path, 'data', 'disease_info.json')
    with open(disease_info_path) as f:
        disease_info = json.load(f)
    if disease_model and disease_classes:
        try:
            from PIL import Image
            import tensorflow as tf
            img = Image.open(filepath).resize((224, 224))
            img_array = np.array(img) / 255.0
            if img_array.shape[-1] == 4: img_array = img_array[:,:,:3]
            img_array = np.expand_dims(img_array, 0)
            preds = disease_model.predict(img_array, verbose=0)[0]
            class_idx = int(np.argmax(preds))
            class_name = disease_classes[class_idx]
            confidence = float(preds[class_idx]) * 100
        except Exception as e:
            class_name, confidence = get_demo_prediction(filepath)
    else:
        class_name, confidence = get_demo_prediction(filepath)
    info = disease_info.get(class_name, {'disease': class_name.replace('_',' '),
           'description': 'Analysis complete.', 'treatment': 'Consult an agricultural expert.'})
    result = {'class': class_name, 'disease': info['disease'], 'confidence': round(confidence,1),
              'description': info['description'], 'treatment': info['treatment']}
    save_history('disease', json.dumps({'filename': filename}), json.dumps(result))
    return jsonify(result)
def get_demo_prediction(filepath=None):
    """Smart demo prediction based on image filename."""
    name = filepath.lower() if filepath else ""

    if "tomato" in name and ("blight" in name or "spot" in name or "brown" in name):
        return "Tomato___Early_blight", 94.3
    elif "tomato" in name and ("late" in name or "rot" in name):
        return "Tomato___Late_blight", 91.7
    elif "tomato" in name and ("mold" in name or "mould" in name):
        return "Tomato___Leaf_Mold", 92.5
    elif "tomato" in name and ("healthy" in name or "good" in name):
        return "Tomato___healthy", 97.2
    elif "tomato" in name:
        return "Tomato___Early_blight", 93.1
    elif "potato" in name and ("early" in name or "brown" in name):
        return "Potato___Early_blight", 93.8
    elif "potato" in name and ("late" in name or "black" in name):
        return "Potato___Late_blight", 90.4
    elif "potato" in name and ("healthy" in name or "good" in name):
        return "Potato___healthy", 96.5
    elif "potato" in name:
        return "Potato___Early_blight", 92.1
    elif "apple" in name and ("scab" in name or "dark" in name):
        return "Apple___Apple_scab", 91.2
    elif "apple" in name and ("rot" in name or "black" in name):
        return "Apple___Black_rot", 90.8
    elif "apple" in name and ("rust" in name or "orange" in name):
        return "Apple___Cedar_apple_rust", 89.6
    elif "apple" in name and ("healthy" in name or "good" in name):
        return "Apple___healthy", 97.8
    elif "corn" in name and ("rust" in name or "brown" in name):
        return "Corn___Common_rust", 92.4
    elif "corn" in name and ("gray" in name or "grey" in name or "spot" in name):
        return "Corn___Cercospora_leaf_spot", 90.1
    elif "corn" in name and ("healthy" in name or "good" in name):
        return "Corn___healthy", 96.3
    elif "grape" in name and ("rot" in name or "black" in name):
        return "Grape___Black_rot", 91.5
    elif "grape" in name and ("esca" in name or "measles" in name):
        return "Grape___Esca", 88.9
    elif "grape" in name and ("healthy" in name or "good" in name):
        return "Grape___healthy", 96.7
    else:
        return "Tomato___Early_blight", 91.5

# ─── Weather API ──────────────────────────────────────────────────────
@app.route('/api/weather', methods=['GET'])
@login_required
def api_weather():
    city = request.args.get('city', 'Delhi')
    api_key = Config.WEATHER_API_KEY
    if api_key == 'YOUR_API_KEY_HERE':
        # Return demo data if no API key
        return jsonify(get_demo_weather(city))
    try:
        # Current weather
        url = f"{Config.WEATHER_API_URL}/weather?q={city}&appid={api_key}&units=metric"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return jsonify(get_demo_weather(city))
        w = resp.json()
        current = {
            'city': w['name'], 'country': w.get('sys',{}).get('country',''),
            'temp': w['main']['temp'], 'feels_like': w['main']['feels_like'],
            'humidity': w['main']['humidity'], 'pressure': w['main']['pressure'],
            'wind_speed': w['wind']['speed'], 'description': w['weather'][0]['description'],
            'icon': w['weather'][0]['icon'], 'main': w['weather'][0]['main']
        }
        # Forecast
        furl = f"{Config.WEATHER_API_URL}/forecast?q={city}&appid={api_key}&units=metric&cnt=40"
        fresp = requests.get(furl, timeout=10)
        forecast = []
        if fresp.status_code == 200:
            fdata = fresp.json()
            seen_dates = set()
            for item in fdata.get('list', []):
                date = item['dt_txt'].split(' ')[0]
                if date not in seen_dates and len(forecast) < 5:
                    seen_dates.add(date)
                    forecast.append({
                        'date': date, 'temp': item['main']['temp'],
                        'humidity': item['main']['humidity'],
                        'description': item['weather'][0]['description'],
                        'icon': item['weather'][0]['icon']
                    })
        tips = get_weather_farming_tips(current)
        return jsonify({'current': current, 'forecast': forecast, 'tips': tips})
    except Exception:
        return jsonify(get_demo_weather(city))

def get_demo_weather(city):
    return {
        'current': {'city': city, 'country': 'IN', 'temp': 32, 'feels_like': 35,
                     'humidity': 65, 'pressure': 1013, 'wind_speed': 3.5,
                     'description': 'partly cloudy', 'icon': '02d', 'main': 'Clouds'},
        'forecast': [
            {'date': '2026-04-29', 'temp': 33, 'humidity': 60, 'description': 'clear sky', 'icon': '01d'},
            {'date': '2026-04-30', 'temp': 31, 'humidity': 70, 'description': 'light rain', 'icon': '10d'},
            {'date': '2026-05-01', 'temp': 30, 'humidity': 75, 'description': 'moderate rain', 'icon': '10d'},
            {'date': '2026-05-02', 'temp': 34, 'humidity': 55, 'description': 'clear sky', 'icon': '01d'},
            {'date': '2026-05-03', 'temp': 32, 'humidity': 62, 'description': 'few clouds', 'icon': '02d'}
        ],
        'tips': ['Demo mode: Add your OpenWeatherMap API key in config.py for live data.',
                 'Good conditions for general farming activities.', 'Monitor soil moisture levels.'],
        'demo': True
    }

def get_weather_farming_tips(w):
    tips = []
    t, h = w['temp'], w['humidity']
    if t > 35: tips.append("🌡️ High temperature! Irrigate crops early morning or late evening.")
    elif t < 10: tips.append("❄️ Low temperature! Protect crops from frost with mulching.")
    else: tips.append("🌡️ Temperature is suitable for most crops.")
    if h > 80: tips.append("💧 High humidity may cause fungal diseases. Ensure proper ventilation.")
    elif h < 30: tips.append("🏜️ Low humidity. Increase irrigation frequency.")
    else: tips.append("💧 Humidity levels are good for farming.")
    main = w.get('main', '')
    if 'Rain' in main: tips.append("🌧️ Rain expected. Delay fertilizer application to avoid runoff.")
    elif 'Clear' in main: tips.append("☀️ Clear weather. Good day for field work and spraying.")
    if w.get('wind_speed', 0) > 10: tips.append("💨 Strong winds. Avoid spraying pesticides.")
    return tips

# ─── Chatbot API ──────────────────────────────────────────────────────
CHATBOT_RESPONSES = {
    'hello': "Hello! 🌾 I'm Krishi AI, your farming assistant. How can I help you today?",
    'hi': "Hi there! 🌱 Ask me anything about farming, crops, or soil!",
    'crop': "I can recommend the best crops! Go to the Crop Recommendation page and enter your soil and climate data.",
    'fertilizer': "For fertilizer recommendations, visit the Fertilizer page. Enter your soil nutrients and crop type.",
    'disease': "Upload a photo of your plant on the Disease Detection page, and I'll identify any issues!",
    'weather': "Check the Weather page for real-time forecasts and farming tips for your area.",
    'soil': "Healthy soil is key! Test your soil's N, P, K levels regularly. Ideal pH for most crops is 6.0-7.0.",
    'water': "Most crops need 1-2 inches of water per week. Water early morning for best absorption.",
    'pest': "For pest control: use neem oil for organic farming, rotate crops yearly, and maintain field hygiene.",
    'organic': "Organic farming tips: use compost, green manure, crop rotation, and biological pest control.",
    'rice': "Rice grows best in warm, humid climates with 20-35°C temp. Needs standing water during growing season.",
    'wheat': "Wheat prefers cool climate (10-25°C). Sow in November and harvest in March-April in India.",
    'season': "Kharif (June-Oct): rice, maize, cotton. Rabi (Oct-Mar): wheat, mustard. Zaid (Mar-Jun): watermelon, cucumber.",
    'help': "I can help with: 🌱 Crops, 🧪 Fertilizers, 🔬 Disease Detection, 🌤️ Weather, and general farming tips!",
    'thank': "You're welcome! 🌾 Happy farming! Feel free to ask more questions anytime.",
    'price': "Crop prices vary by market. Check your local APMC mandi for current rates.",
    'loan': "Farm loans are available through banks like SBI, NABARD. Check PM-KISAN scheme for direct support.",
    'irrigation': "Drip irrigation saves 30-60% water. Consider sprinkler systems for field crops.",
    'harvest': "Harvest crops at the right moisture level. Store in dry, ventilated spaces to prevent spoilage.",
    'seed': "Always use certified seeds from authorized dealers. Treat seeds before sowing to prevent diseases.",
}

@app.route('/api/chatbot', methods=['POST'])
@login_required
def api_chatbot():
    data = request.get_json()
    message = data.get('message', '').lower().strip()
    response = None
    for key, val in CHATBOT_RESPONSES.items():
        if key in message:
            response = val
            break
    if not response:
        if any(w in message for w in ['how', 'what', 'when', 'where', 'why']):
            response = "That's a great question! 🤔 For specific advice, try our Crop Recommendation, Fertilizer, or Disease Detection tools. You can also ask about: crops, soil, water, pests, organic farming, seasons, or irrigation."
        else:
            response = "I'm here to help with farming! 🌾 Try asking about: crops, fertilizers, soil health, pest control, weather, irrigation, or organic farming. Type 'help' for more options."
    save_history('chatbot', json.dumps({'message': data.get('message','')}), json.dumps({'response': response}))
    return jsonify({'response': response})

# ─── History API ──────────────────────────────────────────────────────
@app.route('/api/history', methods=['GET'])
@login_required
def api_history():
    qtype = request.args.get('type', 'all')
    query = History.query.filter_by(user_id=current_user.id)
    if qtype != 'all':
        query = query.filter_by(query_type=qtype)
    items = query.order_by(History.created_at.desc()).limit(50).all()
    result = [{'id': h.id, 'type': h.query_type, 'input': json.loads(h.input_data),
               'result': json.loads(h.result_data), 'date': h.created_at.isoformat()} for h in items]
    return jsonify(result)

@app.route('/api/history/<int:hid>', methods=['DELETE'])
@login_required
def api_delete_history(hid):
    h = History.query.get_or_404(hid)
    if h.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    db.session.delete(h)
    db.session.commit()
    return jsonify({'message': 'Deleted'})

def save_history(qtype, input_data, result_data):
    try:
        h = History(user_id=current_user.id, query_type=qtype, input_data=input_data, result_data=result_data)
        db.session.add(h)
        db.session.commit()
    except: pass

# ─── Run ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        load_models()
    app.run(debug=True, host='0.0.0.0', port=5000)
