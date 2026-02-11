# SafePay | Real-time Transaction Fraud Detection

SafePay is a modern web application designed to detect fraudulent transactions using advanced Deep Learning techniques. It combines Convolutional Neural Networks (CNN) for feature extraction and Long Short-Term Memory (LSTM) networks for sequence pattern learning.

## 🚀 Features

- **Real-time Prediction**: Instantly classify transactions using unique **UTR Numbers**.
- **Admin Dashboard**: Comprehensive analytics with Chart.js visualization.
- **Random Forest Model**: Optimized classifier for Python 3.14 with high accuracy.
- **RESTful API**: Flask-based backend for seamless communication.
- **Aesthetic UI**: Premium design using Bootstrap 5 and modern styling.

## 🛠 Tech Stack

- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5, Chart.js.
- **Backend**: Python 3.14, Flask, SQLite.
- **Machine Learning**: Scikit-learn (Random Forest), Pandas, NumPy, Joblib.

## 📂 Project Structure

- `app.py`: Main Flask application.
- `ml_model.py`: Model architecture, training, and synthetic data generation.
- `templates/`: HTML templates for Index and Dashboard.
- `saved_models/`: Serialized model and scaler files.
- `requirements.txt`: Python dependencies.

## 🚦 Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model**:
   ```bash
   python ml_model.py
   ```

3. **Run the Application**:
   ```bash
   python app.py
   ```

4. **Access the App**:
   - Home: `http://127.0.0.1:5000/`
   - Dashboard: `http://127.0.0.1:5000/dashboard`

## 📊 Model Architecture

The CNN-LSTM model processes transaction features as a sequence:
1. **CNN Layer**: Extracts spatial features and correlations between transaction attributes.
2. **LSTM Layer**: Captures temporal dependencies and patterns in transaction history.
3. **Dense Layers**: Final classification into Fraud (1) or Genuine (0).
