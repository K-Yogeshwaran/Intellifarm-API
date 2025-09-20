# intellifarm_api.py
# ---------------------------
# IMPORTS
# ---------------------------
from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, db, firestore
import threading, time, os, pickle, joblib, numpy as np, requests, json
import faiss
import vertexai
from vertexai.language_models import TextEmbeddingModel
from google.oauth2 import service_account
from google.auth.transport import requests as grequests   # for token refresh
from datetime import datetime

# ---------------------------
# FLASK & FIREBASE INIT
# ---------------------------
app = Flask(_name_)
CORS(app)

# ---------------------------
# GOOGLE VERTEX AI INIT
# ---------------------------
sa_path = r"D:\NEW\INTELLIFARM\lib\intellifarm-backend.json"

scoped_creds = service_account.Credentials.from_service_account_file(
    sa_path,
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
scoped_creds = scoped_creds.with_scopes(["https://www.googleapis.com/auth/cloud-platform"])
scoped_creds.refresh(grequests.Request())

vertexai.init(
    project="intellifarm-marketplace",
    location="us-central1",
    credentials=scoped_creds
)

print("✅ VertexAI initialized with service account:", sa_path)
print("   Credentials valid?", scoped_creds.valid)

# ---------------------------
# FIREBASE INIT
# ---------------------------
cred = credentials.Certificate("intellifarm-backend.json")
firebase_admin.initialize_app(
    cred,
    {"databaseURL": "https://intellifarm-marketplace-default-rtdb.asia-southeast1.firebasedatabase.app/"}
)
realtime_db_ref = db.reference("sensor/moisture")
firestore_db = firestore.client()
latest_data = {"moisture": None, "status": "Unknown", "timestamp": ""}

# ---------------------------
# STATUS TRACKING (for / route)
# ---------------------------
sync_status = {
    "last_rag_sync": None,
    "rag_entries": 0,
    "rag_dim": 0,
    "rag_error": None,
    "last_moisture": None,
    "last_moisture_status": None,
    "last_moisture_time": None,
}

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
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

# ---------------------------
# BACKGROUND THREAD: SYNC DATA
# ---------------------------
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
            firestore_db.collection("soilmoisture").document("Sensor1").set({
                "moisture": value,
                "status": status,
                "timestamp": firestore.SERVER_TIMESTAMP,
            })
            latest_data.update({
                "moisture": value,
                "status": status,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            })
            # 🔄 update sync_status for / route
            sync_status.update({
                "last_moisture": value,
                "last_moisture_status": status,
                "last_moisture_time": latest_data["timestamp"],
            })
            print(f"✅ Synced Moisture: {value} ({status})")
        except Exception as e:
            print(f"❌ Sync Error: {e}")
        time.sleep(5)

# ---------------------------
# LOAD ML MODELS
# ---------------------------
with open("rain_prediction_model.pkl", "rb") as f:
    rain_model = pickle.load(f)
crop_model = joblib.load("crop_advisory_model.pkl")

# ---------------------------
# VERTEX AI EMBEDDINGS
# ---------------------------
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")

# ---------------------------
# FAISS CONFIG
# ---------------------------
GCS_BUCKET_NAME = "intellifarm-vectors"
FAISS_INDEX_PATH_LOCAL = "faiss_index_local.index"
FAISS_INDEX_PATH_GCS = "faiss/intellifarm_faiss.index"

faiss_index = None
kb_texts = []
faiss_lock = threading.Lock()

# ---------------------------
# Build knowledge base
# ---------------------------
def build_knowledge_base_from_firestore():
    kb = []
    try:
        # ---------------- Products ----------------
        products_ref = firestore_db.collection("products").stream()
        for doc in products_ref:
            data = doc.to_dict()
            farmer = data.get("farmerName", "Unknown")
            product = data.get("name", "Unknown")
            quantity = data.get("quantity", "N/A")
            price = data.get("price", "N/A")
            negotiable = "negotiable" if data.get("negotiable", False) else "fixed price"
            kb.append(
                f"Farmer {farmer} offers {quantity}kg of {product} at ₹{price} per kg. Price is {negotiable}."
            )

        # ---------------- Dealer Orders ----------------
        orders_ref = firestore_db.collection("dealer_orders").stream()
        for doc in orders_ref:
            data = doc.to_dict()
            farmer = data.get("farmerName", "Unknown")
            dealer = data.get("dealerName", "Unknown")
            product = data.get("productName", "Unknown")
            category = data.get("category", "General")
            quantity = data.get("quantity", "N/A")
            price = data.get("price", "N/A")
            status = data.get("acceptStatus", "pending")
            timestamp = data.get("timestamp", "Unknown Date")
            kb.append(
                f"Dealer {dealer} placed an order for {quantity}kg of {product} ({category}) "
                f"from Farmer {farmer} at ₹{price} per kg. Order status is {status}. Ordered on {timestamp}."
            )

        # ---------------- Market Prices ----------------
        prices_ref = firestore_db.collection("market_prices").stream()
        for doc in prices_ref:
            data = doc.to_dict()
            commodity = data.get("commodity", "Unknown")
            variety = data.get("variety", "Unknown")
            district = data.get("district", "Unknown")
            market = data.get("market", "Unknown")
            min_price = data.get("min_price", "N/A")
            max_price = data.get("max_price", "N/A")
            modal_price = data.get("modal_price_per_kg", "N/A")
            date = data.get("date", "Unknown Date")

            kb.append(
                f"In {district}, at {market}, the price of {commodity} ({variety}) "
                f"on {date} ranged from ₹{min_price} to ₹{max_price}, "
                f"with a modal price of ₹{modal_price} per kg."
            )

    except Exception as e:
        print("❌ Error building KB from Firestore:", e)

    return kb


# ---------------------------
# Sync RAG loop
# ---------------------------
def sync_rag_loop(poll_interval=5):
    global faiss_index, kb_texts
    last_kb = None

    def batch_embeddings(texts, batch_size=250, sleep_time=1.0):
        """Generate embeddings in safe batches (max 250 per request)."""
        all_embeds = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i+batch_size]

            embeds_objs = embedding_model.get_embeddings(chunk)
            all_embeds.extend([e.values for e in embeds_objs])

            # Avoid 429 quota errors
            time.sleep(sleep_time)

        return np.array(all_embeds, dtype="float32")

    while True:
        try:
            new_kb = build_knowledge_base_from_firestore()
            if new_kb and new_kb != last_kb:
                print("🔄 Updating FAISS index...")

                # ✅ Use batching to respect 250 limit
                kb_embeddings = batch_embeddings(new_kb, batch_size=250, sleep_time=1.0)

                dim = kb_embeddings.shape[1] if kb_embeddings.size else 0
                if dim > 0:
                    new_index = faiss.IndexFlatL2(dim)
                    new_index.add(kb_embeddings)

                    with faiss_lock:
                        faiss_index = new_index
                        kb_texts = new_kb.copy()

                    try:
                        faiss.write_index(faiss_index, FAISS_INDEX_PATH_LOCAL)
                        with open("kb_texts.pkl", "wb") as f:
                            pickle.dump(kb_texts, f)
                        print("💾 FAISS + KB saved locally.")
                    except Exception as e:
                        print("⚠ Save error:", e)

                    last_kb = new_kb.copy()
                    # 🔄 update sync_status for /
                    sync_status.update({
                        "last_rag_sync": datetime.utcnow().isoformat(),
                        "rag_entries": len(kb_texts),
                        "rag_dim": dim,
                        "rag_error": None
                    })
                    print(f"✅ RAG index rebuilt with {len(kb_texts)} entries (dim={dim}).")
        except Exception as e:
            print("❌ Error in RAG sync loop:", e)
            sync_status["rag_error"] = str(e)

        time.sleep(poll_interval)


# ---------------------------
# Load FAISS locally
# ---------------------------
def try_load_local_faiss():
    global faiss_index, kb_texts
    try:
        if os.path.exists(FAISS_INDEX_PATH_LOCAL) and os.path.exists("kb_texts.pkl"):
            loaded_index = faiss.read_index(FAISS_INDEX_PATH_LOCAL)
            with open("kb_texts.pkl", "rb") as f:
                loaded_kb = pickle.load(f)
            with faiss_lock:
                faiss_index = loaded_index
                kb_texts = loaded_kb
            print("✅ Loaded FAISS + KB locally.")
            return True
    except Exception as e:
        print("⚠ Could not load local FAISS:", e)
    return False

try_load_local_faiss()

# ---------------------------
# Retrieve entries
# ---------------------------
def retrieve_similar_entries(query, top_k=3):
    global faiss_index, kb_texts
    with faiss_lock:
        if faiss_index is None or not kb_texts:
            temp_kb = build_knowledge_base_from_firestore()
            return temp_kb[:top_k]
        q_emb_obj = embedding_model.get_embeddings([query])[0]
        q_emb = np.array(q_emb_obj.values).astype("float32").reshape(1, -1)
        D, I = faiss_index.search(q_emb, top_k)
        return [kb_texts[i] for i in I[0] if i < len(kb_texts)]

# ---------------------------
# GROQ CONFIG
# ---------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY") or "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"
GROQ_MODEL = "llama-3.1-8b-instant"

# ---------------------------
# CHAT HISTORY (FIRESTORE STORAGE)
# ---------------------------
MAX_HISTORY = 25

def load_chat_history(uid):
    try:
        ref = firestore_db.collection("users").document(uid).collection("chatHistory").order_by("timestamp").limit(MAX_HISTORY)
        docs = ref.stream()
        return [{"role": d.to_dict()["role"], "content": d.to_dict()["content"]} for d in docs]
    except Exception as e:
        print(f"⚠ Error loading chat history for {uid}: {e}")
        return []

def save_message(uid, role, content):
    try:
        firestore_db.collection("users").document(uid).collection("chatHistory").add({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow()
        })
    except Exception as e:
        print(f"⚠ Error saving message for {uid}: {e}")

@app.route("/chat/history", methods=["POST"])
def get_chat_history():
    data = request.get_json() or {}
    uid = data.get("uid")
    if not uid:
        return jsonify({"error": "UID required"}), 400
    history = load_chat_history(uid)
    return jsonify({"history": history[-5:]})

# ---------------------------
# USER INIT ENDPOINT
# ---------------------------
@app.route("/init/user", methods=["POST"])
def init_user():
    data = request.get_json() or {}
    uid = data.get("uid")
    role = data.get("role")
    name = data.get("name")

    if not uid:
        return jsonify({"error": "No UID provided"}), 400

    user_ref = firestore_db.collection("users").document(uid)
    user_ref.set({"uid": uid, "role": role, "name": name}, merge=True)

    # Add system entry in chatHistory
    save_message(uid, "system", f"User info: {json.dumps({'uid': uid, 'role': role, 'name': name})}")

    return jsonify({"message": "User initialized", "uid": uid})

# ---------------------------
# FLASK ROUTES
# ---------------------------
from flask import render_template_string, request

@app.route("/", methods=["GET", "POST"])
def home():
    query = None
    results = []

    if request.method == "POST":
        query = request.form.get("query")
        if query:
            results = retrieve_similar_entries(query, top_k=5)

    # HTML template with search
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>IntelliFarm Dashboard</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background: #f4f9f4;
                color: #333;
            }
            h1 {
                color: #2e7d32;
            }
            .card {
                background: white;
                padding: 20px;
                margin: 15px 0;
                border-radius: 12px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            }
            .card h2 {
                margin-top: 0;
                color: #388e3c;
            }
            .entry {
                padding: 10px;
                margin: 5px 0;
                border-bottom: 1px solid #eee;
            }
            .search-box {
                margin: 20px 0;
            }
            input[type=text] {
                padding: 10px;
                width: 70%;
                border-radius: 8px;
                border: 1px solid #ccc;
                font-size: 16px;
            }
            button {
                padding: 10px 20px;
                background: #2e7d32;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                cursor: pointer;
            }
            button:hover {
                background: #1b5e20;
            }
        </style>
    </head>
    <body>
        <h1>🌱 IntelliFarm Dashboard</h1>

        <div class="card">
            <h2>Soil Moisture</h2>
            <p><b>Value:</b> {{ moisture.value }}</p>
            <p><b>Status:</b> {{ moisture.status }}</p>
            <p><b>Timestamp:</b> {{ moisture.timestamp }}</p>
        </div>

        <div class="card">
            <h2>RAG Status</h2>
            <p><b>Last Sync:</b> {{ rag.last_sync }}</p>
            <p><b>Entries:</b> {{ rag.entries }}</p>
            <p><b>Dimension:</b> {{ rag.dimension }}</p>
            <p><b>Error:</b> {{ rag.error }}</p>
        </div>

        <div class="card">
            <h2>Knowledge Base Entries (Vectorized)</h2>
            {% if kb_texts %}
                {% for entry in kb_texts %}
                    <div class="entry">{{ loop.index }}. {{ entry }}</div>
                {% endfor %}
            {% else %}
                <p>No entries found.</p>
            {% endif %}
        </div>

        <div class="card">
            <h2>🔍 Search Knowledge Base</h2>
            <form method="POST">
                <div class="search-box">
                    <input type="text" name="query" placeholder="Ask something..." value="{{ query or '' }}">
                    <button type="submit">Search</button>
                </div>
            </form>

            {% if query %}
                <h3>Results for "{{ query }}":</h3>
                {% if results %}
                    {% for res in results %}
                        <div class="entry">{{ loop.index }}. {{ res }}</div>
                    {% endfor %}
                {% else %}
                    <p>No relevant results found.</p>
                {% endif %}
            {% endif %}
        </div>
    </body>
    </html>
    """

    return render_template_string(
        html,
        moisture={
            "value": sync_status.get("last_moisture"),
            "status": sync_status.get("last_moisture_status"),
            "timestamp": sync_status.get("last_moisture_time"),
        },
        rag={
            "last_sync": sync_status.get("last_rag_sync"),
            "entries": sync_status.get("rag_entries"),
            "dimension": sync_status.get("rag_dim"),
            "error": sync_status.get("rag_error"),
        },
        kb_texts=kb_texts if kb_texts else [],
        query=query,
        results=results
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json() or {}
        input_data = np.array([[float(data["temperature"]), float(data["humidity"]), float(data["wind_speed"])]])
        prediction = rain_model.predict(input_data)
        return jsonify({"prediction": "Rain" if prediction[0] == 1 else "No Rain"})
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
    data = request.get_json() or {}
    uid = data.get("uid")
    user_prompt = data.get("prompt", "")

    if not uid or not user_prompt:
        return jsonify({"error": "Missing uid or prompt"}), 400

    # Load history
    history = load_chat_history(uid)
    save_message(uid, "user", user_prompt)

    try:
        top_entries = retrieve_similar_entries(user_prompt)
        context_text = "\n".join(top_entries) if top_entries else "No context available."

        # Build messages list
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for msg in history:
            if "role" in msg and "content" in msg:
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_prompt})
        messages.append({"role": "system", "content": f"Extra context:\n{context_text}"})

        payload = {"model": GROQ_MODEL, "messages": messages}
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60  # ✅ Increase timeout
        )

        # Check HTTP response
        if response.status_code != 200:
            print(f"❌ Groq API HTTP {response.status_code}: {response.text}")
            return jsonify({"error": f"Groq API returned status {response.status_code}", "details": response.text}), 500

        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            reply = result["choices"][0]["message"]["content"]
            save_message(uid, "assistant", reply)
            return jsonify({"response": reply, "history": load_chat_history(uid)})
        else:
            print("❌ Groq API returned no choices:", result)
            return jsonify({"error": "No response from Groq API", "details": result}), 500

    except Exception as e:
        print("❌ Exception in /chat:", str(e))
        return jsonify({"error": str(e)}), 500


# ---------------------------
# START BACKGROUND THREADS
# ---------------------------
threading.Thread(target=sync_data_loop, daemon=True).start()
threading.Thread(target=sync_rag_loop, daemon=True).start()

# ---------------------------
# RUN FLASK
# ---------------------------
if _name_ == "_main_":
    app.run(host="0.0.0.0", port=5000, debug=True)
