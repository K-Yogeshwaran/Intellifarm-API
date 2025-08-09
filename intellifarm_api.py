from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, db, firestore
import threading
import time
import pickle
import joblib
import numpy as np
import requests

# --- Initialize Flask ---
app = Flask(__name__)
CORS(app)

# --- Firebase Initialization ---
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(
    cred,
    {
        "databaseURL": "https://intellifarm-marketplace-default-rtdb.asia-southeast1.firebasedatabase.app/"
    },
)
realtime_db_ref = db.reference("sensor/moisture")
firestore_db = firestore.client()

latest_data = {"moisture": None, "status": "Unknown", "timestamp": ""}

# --- Helper Function: Moisture Status ---
def get_status(value):
    if value is None:
        return "Unknown"
    min_val, max_val = 2000, 4095
    percent = ((max_val - value) / (max_val - min_val))
    if percent < 0.3:
        return "Dry"
    elif percent < 0.6:
        return "Moderate"
    else:
        return "Wet"

# --- Background Sync: Read from Realtime DB and update Firestore ---
def sync_data_loop():
    while True:
        try:
            value = realtime_db_ref.get()

            if value is None:
                print("⚠ No moisture data found in Realtime DB.")
                time.sleep(5)
                continue

            value = int(float(value))
            status = get_status(value)

            # Save to Firestore
            doc_ref = firestore_db.collection("soilmoisture").document("Sensor1")
            doc_ref.set({
                "moisture": value,
                "status": status,
                "timestamp": firestore.SERVER_TIMESTAMP,
            })

            latest_data["moisture"] = value
            latest_data["status"] = status
            latest_data["timestamp"] = time.strftime('%Y-%m-%d %H:%M:%S')

            print(f"✅ Synced Moisture: {value} ({status})")
        except Exception as e:
            print(f"❌ Sync Error: {e}")
        time.sleep(5)

# --- Load Models ---
with open("rain_prediction_model.pkl", "rb") as f:
    rain_model = pickle.load(f)

crop_model = joblib.load("crop_advisory_model.pkl")

# --- Groq Configuration ---
GROQ_API_KEY = "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"  # Replace with yours
GROQ_MODEL = "llama3-8b-8192"

# --- ROUTES ---

@app.route("/")
def home():
    html = """
    <html>
        <head><title>IntelliFarm Dashboard</title></head>
        <body style="font-family:Arial;padding:40px">
            <h1>🌱 IntelliFarm - Soil Moisture Monitor</h1>
            <hr>
            <h3>💧 Moisture Value: <span style="color:green">{{ moisture }}</span></h3>
            <h3>📊 Status: <span style="color:blue">{{ status }}</span></h3>
            <p>⏱ Last Updated: {{ timestamp }}</p>
        </body>
    </html>
    """
    return render_template_string(
        html,
        moisture=latest_data["moisture"],
        status=latest_data["status"],
        timestamp=latest_data["timestamp"]
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_data = np.array([[float(data["temperature"]), float(data["humidity"]), float(data["wind_speed"])]])
        prediction = rain_model.predict(input_data)
        result = "Rain" if prediction[0] == 1 else "No Rain"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/get_advisory", methods=["GET"])
def get_advisory():
    crop = request.args.get("crop")
    if not crop:
        return jsonify({"error": "No crop name provided"}), 400
    try:
        output = crop_model.predict([crop])[0]
        steps = output.strip().split("\n")
        return jsonify({"crop": crop, "steps": steps})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    prompt = data.get("prompt", "")
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        }
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        result = response.json()
        if "choices" in result and result["choices"]:
            reply = result["choices"][0]["message"]["content"]
            return jsonify({"response": reply})
        return jsonify({"error": "No response from Groq API"})
    except Exception as e:
        return jsonify({"error": str(e)})

# --- Start background thread ---
t = threading.Thread(target=sync_data_loop, daemon=True)
t.start()

# --- Run Flask ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)