from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Global variables
model = None
le_agency = None
le_incident = None
feature_medians = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Emergency Delay Predictor</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f5f5f5;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border: 2px solid #000;
            padding: 40px;
            max-width: 500px;
            width: 100%;
        }
        
        h1 {
            color: #000;
            margin-bottom: 10px;
            font-size: 28px;
        }
        
        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 500;
            font-size: 14px;
        }
        
        select, input {
            width: 100%;
            padding: 12px;
            border: 2px solid #000;
            background: white;
            font-size: 14px;
        }
        
        select:focus, input:focus {
            outline: none;
            border-color: #000;
        }
        
        button {
            width: 100%;
            padding: 14px;
            background: #000;
            color: white;
            border: 2px solid #000;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
        }
        
        button:hover {
            background: #333;
        }
        
        button:active {
            background: #000;
        }
        
        .result {
            margin-top: 30px;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            display: none;
        }
        
        .result.delay {
            background: #fee;
            border: 2px solid #f88;
        }
        
        .result.no-delay {
            background: #efe;
            border: 2px solid #8f8;
        }
        
        .result-title {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .result.delay .result-title {
            color: #d00;
        }
        
        .result.no-delay .result-title {
            color: #0a0;
        }
        
        .probability {
            font-size: 18px;
            color: #666;
        }
        
        .loading {
            display: none;
            text-align: center;
            color: #667eea;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Emergency Delay Predictor</h1>
        <p class="subtitle">Predict if an emergency response will be delayed</p>
        
        <form id="predictionForm">
            <div class="form-group">
                <label for="agency">Agency</label>
                <select id="agency" required>
                    <option value="">Select an agency...</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="incident_type">Incident Type</label>
                <select id="incident_type" required>
                    <option value="">Select incident type...</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="num_incidents">Number of Incidents</label>
                <input type="number" id="num_incidents" required min="1" placeholder="e.g., 267">
            </div>
            
            <div class="form-group">
                <label for="call_to_pickup">Call to Pickup Time (minutes)</label>
                <input type="number" id="call_to_pickup" required step="0.01" min="0" placeholder="e.g., 3.46">
            </div>
            
            <div class="form-group">
                <label for="weekday">Day of Week</label>
                <select id="weekday" required>
                    <option value="">Select day...</option>
                    <option value="0">Monday</option>
                    <option value="1">Tuesday</option>
                    <option value="2">Wednesday</option>
                    <option value="3">Thursday</option>
                    <option value="4">Friday</option>
                    <option value="5">Saturday</option>
                    <option value="6">Sunday</option>
                </select>
            </div>
            
            <button type="submit">Predict Delay</button>
        </form>
        
        <div class="loading" id="loading">
            <p>Analyzing...</p>
        </div>
        
        <div class="result" id="result">
            <div class="result-title" id="resultTitle"></div>
            <div class="probability" id="resultProb"></div>
        </div>
    </div>
    
    <script>
        // Load agencies and incident types on page load
        fetch('/agencies')
            .then(r => r.json())
            .then(data => {
                const select = document.getElementById('agency');
                data.agencies.forEach(agency => {
                    const option = document.createElement('option');
                    option.value = agency;
                    option.textContent = agency;
                    select.appendChild(option);
                });
            });
        
        fetch('/incident_types')
            .then(r => r.json())
            .then(data => {
                const select = document.getElementById('incident_type');
                data.incident_types.forEach(type => {
                    const option = document.createElement('option');
                    option.value = type;
                    option.textContent = type;
                    select.appendChild(option);
                });
            });
        
        // Handle form submission
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            
            loading.style.display = 'block';
            result.style.display = 'none';
            
            const data = {
                Agency: document.getElementById('agency').value,
                incident_type: document.getElementById('incident_type').value,
                num_incidents: parseInt(document.getElementById('num_incidents').value),
                call_to_pickup: parseFloat(document.getElementById('call_to_pickup').value),
                weekday: parseInt(document.getElementById('weekday').value)
            };
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                
                const pred = await response.json();
                
                loading.style.display = 'none';
                result.style.display = 'block';
                
                if (pred.prediction === 'Delay') {
                    result.className = 'result delay';
                    document.getElementById('resultTitle').textContent = '⚠️ DELAY EXPECTED';
                } else {
                    result.className = 'result no-delay';
                    document.getElementById('resultTitle').textContent = '✅ NO DELAY';
                }
                
                document.getElementById('resultProb').textContent = 
                    `Delay probability: ${(pred.delay_probability * 100).toFixed(1)}%`;
                
            } catch (error) {
                loading.style.display = 'none';
                alert('Error making prediction: ' + error);
            }
        });
    </script>
</body>
</html>
"""

def train_model():
    """Train the model on startup"""
    global model, le_agency, le_incident, feature_medians
    
    df = pd.read_csv("emergency_data_trimmed.csv")
    
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
    
    print("Model trained successfully!")
    print(f"Test accuracy: {model.score(X_test, y_test):.3f}")

@app.route('/')
def home():
    """Serve the web interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
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

@app.route('/agencies', methods=['GET'])
def get_agencies():
    """Get list of valid agencies"""
    return jsonify({'agencies': le_agency.classes_.tolist()})

@app.route('/incident_types', methods=['GET'])
def get_incident_types():
    """Get list of valid incident types"""
    return jsonify({'incident_types': le_incident.classes_.tolist()})

if __name__ == '__main__':
    print("Training model...")
    train_model()
    
    print("\nStarting web app...")
    print("Open http://localhost:5000 in your browser!")
    app.run(debug=True, host='0.0.0.0', port=5000)