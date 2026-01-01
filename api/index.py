from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load the pre-trained model
model_path = os.path.join(os.path.dirname(__file__), '..', 'model.pkl')
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
le_agency = model_data['le_agency']
le_incident = model_data['le_incident']
feature_medians = model_data['feature_medians']

@app.route('/api/predict', methods=['POST'])
def predict():
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
    return jsonify({'agencies': le_agency.classes_.tolist()})

@app.route('/api/incident_types', methods=['GET'])
def get_incident_types():
    return jsonify({'incident_types': le_incident.classes_.tolist()})
