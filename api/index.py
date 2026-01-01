# FILE 1: api/index.py
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

app = Flask(__name__)

# Global variables
model = None
le_agency = None
le_incident = None
feature_medians = None

def train_model():
    """Train the model on startup"""
    global model, le_agency, le_incident, feature_medians
    
    # Find CSV file
    csv_path = "emergency_data_trimmed.csv"
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), "..", "emergency_data_trimmed.csv")
    
    df = pd.read_csv(csv_path)
    
    df['week_start'] = pd.to_datetime(df['week_start'])
    df['weekday'] = df['week_start'].dt.weekday
    
    le_agency = LabelEncoder()
    le_incident = LabelEncoder()
    df['agency_enc'] = le_agency.fit_transform(df['Agency'])
    df['incident_enc'] = le_incident.fit_transform(df['incident_type'])
    
    df['delay'] = (df['avg_travel_time'] > 10).astype(int)
    
    features = ['agency_enc', 'incident_enc', 'num_incidents', 'call_to_pickup', 'weekday']
    
    X = df[features].fillna(df[features].median())
    y = df['delay']
    
    feature_medians = X.median().to_dict()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

# Train model once
try:
    train_model()
except Exception as e:
    print(f"Warning: Could not train model: {e}")

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict delay for an emergency call"""
    try:
        data = request.get_json()
        
        agency_enc = le_agency.transform([data['Agency']])[0]
        incident_enc = le_incident.transform([data['incident_type']])[0]
        
        sample_input = {
            'agency_enc': agency_enc,
            'incident_enc': incident_enc,
            'num_incidents': data['num_incidents'],
            'call_to_pickup': data['call_to_pickup'],
            'weekday': data['weekday']
        }
        
        sample_df = pd.DataFrame([sample_input])
        sample_df = sample_df.fillna(feature_medians)
        
        pred = model.predict(sample_df)[0]
        prob = model.predict_proba(sample_df)[0]
        
        return jsonify({
            'prediction': 'Delay' if pred == 1 else 'No Delay',
            'delay_probability': float(prob[1])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/agencies', methods=['GET'])
def get_agencies():
    """Get list of valid agencies"""
    return jsonify({'agencies': le_agency.classes_.tolist()})

@app.route('/api/incident_types', methods=['GET'])
def get_incident_types():
    """Get list of valid incident types"""
    return jsonify({'incident_types': le_incident.classes_.tolist()})

# Vercel serverless function handler
def handler(request):
    with app.request_context(request.environ):
        return app.full_dispatch_request()
