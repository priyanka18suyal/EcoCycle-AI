import os
import random
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, redirect
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

# =============================================
# APP SETUP
# =============================================

app = Flask(__name__)
app.secret_key = "EcoMind2025_Final"

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=15)

db = SQLAlchemy(app)
CORS(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'

# =============================================
# DATABASE MODELS
# =============================================

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    phone = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default='user')
    points = db.Column(db.Integer, default=0)


class Scan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    waste_type = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


with app.app_context():
    db.create_all()


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# =============================================
# MODEL LOAD
# =============================================

try:
    model = load_model('waste_model_binary.keras')

except Exception as e:

    model = None


# =============================================
# CONSTANTS
# =============================================

ACTIONS = {
    'bio': 'Compost Bin (Green)',
    'non_bio': 'Recycling / General Bin'
}

SAFETY_GUIDE = {
    'bio': {'safety': 'N95 + Gloves + Boots', 'tools': 'Double Bag'},
    'non_bio': {'safety': 'Gloves + Mask', 'tools': 'Recycling Bag'}
}

ECO_FACTS = {
    'bio': 'Composting reduces methane emissions!',
    'non_bio': 'Plastic takes 400+ years to decompose!'
}


# =============================================
# IMAGE PREPROCESS
# =============================================

def prepare_image(img):
    img = img.convert("RGB").resize((224, 224))
    arr = img_to_array(img)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


# =============================================
# ROUTES
# =============================================

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(phone=request.form['phone']).first()
        if user and check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect('/')
        return "<h1>Wrong Credentials</h1>"
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        if User.query.filter_by(phone=request.form['phone']).first():
            return "<h1>Phone already exists</h1>"

        user = User(
            phone=request.form['phone'],
            email=request.form['email'],
            password=generate_password_hash(request.form['password']),
            role=request.form['role']
        )

        db.session.add(user)
        db.session.commit()
        login_user(user)

        return redirect('/')

    return render_template('register.html')


@app.route('/logout')
def logout():
    logout_user()
    return redirect('/')


# =============================================
# API
# =============================================

@app.route('/api/classify', methods=['POST'])
def classify():

    if not model or 'image' not in request.files:
        return jsonify({"error": "Model not loaded or image missing"}), 400

    try:
        file = request.files['image']
        img = Image.open(file)

        pred = model.predict(prepare_image(img))[0][0]

        if pred > 0.5:
            label = 'non_bio'
            conf = float(pred)
        else:
            label = 'bio'
            conf = float(1 - pred)

        if conf < 0.40:
            return jsonify({
                "class": "Unknown",
                "action": "Try clearer image"
            })

        guide = SAFETY_GUIDE[label]

        return jsonify({
            "class": label,
            "action": ACTIONS[label],
            "confidence": f"{conf*100:.1f}%",
            "safety": guide['safety'],
            "tools": guide['tools'],
            "fact": ECO_FACTS[label]
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Prediction failed"}), 500


# =============================================
# RUN
# =============================================

if __name__ == '__main__':
    app.run(debug=True)