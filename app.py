import os
print("Starting SafePay Flask Application...")
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fraud_detection.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Models
class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    amount = db.Column(db.Float, nullable=False)
    utr_number = db.Column(db.String(50), nullable=False)
    location = db.Column(db.String(100), nullable=False)
    time = db.Column(db.String(50), nullable=False)
    frequency = db.Column(db.Integer, nullable=False)
    is_fraud = db.Column(db.Boolean, default=False)
    prediction_score = db.Column(db.Float, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "amount": self.amount,
            "utr_number": self.utr_number,
            "location": self.location,
            "time": self.time,
            "frequency": self.frequency,
            "is_fraud": self.is_fraud,
            "prediction_score": self.prediction_score
        }

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# Load model and scaler lazily
def load_ml_model():
    global model, scaler
    try:
        import joblib
        import os
        
        model_path = 'saved_models/fraud_model.joblib'
        scaler_path = 'saved_models/scaler.pkl'
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            print("Random Forest Model and Scaler loaded successfully.")
        else:
            print(f"Model files not found at {model_path}. Using fallback prediction logic.")
            model = None
            scaler = None
    except Exception as e:
        print(f"Error loading model/scaler: {e}")
        model = None
        scaler = None

model = None
scaler = None

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    amount = float(data.get('amount', 0))
    # Map time (HH:MM) to hour (0-23)
    time_str = data.get('time', '00:00')
    hour = int(time_str.split(':')[0])
    frequency = int(data.get('frequency', 1))
    
    # Simple location encoding for demo
    location_map = {"Mumbai": 0, "Delhi": 1, "Bangalore": 2, "Chennai": 3, "Others": 4}
    location_idx = location_map.get(data.get('location'), 4)

    if model and scaler:
        # Prepare input features: amount, hour, frequency, location_idx
        features = np.array([[amount, hour, frequency, location_idx]])
        features_scaled = scaler.transform(features)
        
        # Predict probability for class 1 (Fraud)
        prediction_score = float(model.predict_proba(features_scaled)[0][1])
        is_fraud = prediction_score > 0.5
    else:
        # Fallback dummy logic if model not loaded
        prediction_score = 0.9 if amount > 10000 else 0.1
        is_fraud = amount > 10000
    
    new_transaction = Transaction(
        amount=amount,
        utr_number=data.get('utr_number', 'N/A'),
        location=data.get('location', 'Global'),
        time=time_str,
        frequency=frequency,
        is_fraud=is_fraud,
        prediction_score=prediction_score
    )
    db.session.add(new_transaction)
    db.session.commit()
    
    return jsonify({
        "status": "success",
        "is_fraud": bool(is_fraud),
        "prediction_score": prediction_score
    })

@app.route('/api/transactions', methods=['GET'])
def get_transactions():
    transactions = Transaction.query.all()
    return jsonify([t.to_dict() for t in transactions])

@app.route('/api/stats', methods=['GET'])
def get_stats():
    total = Transaction.query.count()
    fraudulent = Transaction.query.filter_by(is_fraud=True).count()
    genuine = total - fraudulent
    return jsonify({
        "total": total,
        "fraudulent": fraudulent,
        "genuine": genuine
    })

@app.route('/api/model_status', methods=['GET'])
def model_status():
    return jsonify({
        "loaded": model is not None,
        "scaler_loaded": scaler is not None
    })

if __name__ == '__main__':
    load_ml_model()
    app.run(debug=True, port=5000)
