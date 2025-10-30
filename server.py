# server.py
# AI Farmer Assistant – Intermediate Backend (FastAPI, FAISS, SQLite)
# ---------------------------------------------------------------
# Endpoints:
#  POST /api/answer        -> RAG Q&A over local scheme snippets
#  POST /api/detect        -> Leaf disease detection (base64)
#  POST /api/detect-file   -> Leaf disease detection (file upload)
#  POST /api/feedback      -> Store feedback on an answer
#  POST /api/users         -> Create demo user (no auth flow)
#  GET  /api/health        -> Health + counts
#  --- Admin/Utility ---
#  POST /admin/reload      -> Rebuild embeddings from data/schemes
#  POST /admin/schemes     -> Upload a scheme snippet (title + text)
#  GET  /admin/search?q=   -> Debug: search the index
#
# Storage: SQLite (data/app.db)
# Knowledge: plaintext .txt under data/schemes/*.txt
# Embeddings: sentence-transformers (MiniLM) + FAISS
# Vision: EfficientNet-B0 classifier (drop weights at models/disease_model.pt)
#
# Optional:
#  - API key via env API_KEY (applies to /api/* and /admin/* except /api/health)
#  - OPENAI_API_KEY -> if set, we compose a final answer with OpenAI for nicer wording

import os, re, io, base64, json, time, glob, sqlite3
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, Form, Depends, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image

# --------------------
# Configuration
# --------------------
DATA_DIR = os.getenv("DATA_DIR", "data")
SCHEMES_DIR = os.path.join(DATA_DIR, "schemes")
DB_PATH = os.path.join(DATA_DIR, "app.db")
MODELS_DIR = os.getenv("MODELS_DIR", "models")
DISEASE_MODEL_PATH = os.path.join(MODELS_DIR, "disease_model.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "4"))

API_KEY = os.getenv("API_KEY")            # optional simple header key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # optional beautifier

# Rate limiting (IP-based token bucket in memory)
RATE_LIMIT_RPS = float(os.getenv("RATE_LIMIT_RPS", "2.0"))
RATE_LIMIT_BURST = int(os.getenv("RATE_LIMIT_BURST", "6"))

# Class names (edit order to match your model training)
CLASS_NAMES = [
    "Apple___Black_rot", "Apple___healthy",
    "Corn_(maize)___Common_rust", "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites",
    "Tomato___Target_Spot", "Tomato___YellowLeaf__Curl_Virus", "Tomato___mosaic_virus",
    "Tomato___healthy"
]

# --------------------
# App + CORS
# --------------------
app = FastAPI(title="AI Farmer Assistant – Backend", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# --------------------
# Simple API key auth
# --------------------
def require_key(x_api_key: Optional[str] = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

# --------------------
# Rate Limiter
# --------------------
_buckets = {}  # ip -> (tokens, last_time)

def rate_limiter(client_ip: str):
    now = time.time()
    tokens, last = _buckets.get(client_ip, (RATE_LIMIT_BURST, now))
    # refill
    delta = now - last
    tokens = min(RATE_LIMIT_BURST, tokens + delta * RATE_LIMIT_RPS)
    if tokens < 1:
        raise HTTPException(status_code=429, detail="Too many requests, slow down.")
    _buckets[client_ip] = (tokens - 1, now)

def get_client_ip(x_forwarded_for: Optional[str] = Header(None)):
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()
    return "local"

# --------------------
# DB (SQLite)
# --------------------
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SCHEMES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db():
    conn = db()
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, email TEXT UNIQUE, phone TEXT,
        created_at TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS chats(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, message TEXT, answer TEXT,
        citations TEXT, created_at TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS feedback(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER, rating INTEGER, note TEXT,
        created_at TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS detections(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, label TEXT, confidence REAL,
        created_at TEXT
    )""")
    conn.commit()
    conn.close()

init_db()

# --------------------
# RAG: load corpus, embed, index
# --------------------
embedder = SentenceTransformer(EMBED_MODEL_NAME)
INDEX = None
DOCS = []  # list[dict{id,title,text,source}]

def _read_txt_files() -> List[dict]:
    docs = []
    for path in sorted(glob.glob(os.path.join(SCHEMES_DIR, "*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            continue
        title = os.path.splitext(os.path.basename(path))[0]
        docs.append({"id": len(docs), "title": title, "text": text, "source": path})
    return docs

def _build_index():
    global INDEX, DOCS
    DOCS = _read_txt_files()
    if not DOCS:
        INDEX = None
        return
    texts = [d["text"] for d in DOCS]
    embs = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    INDEX = index

_build_index()

def retrieve(query: str, k: int = TOP_K):
    if INDEX is None or not DOCS:
        return []
    q = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = INDEX.search(q, k)
    hits = []
    for rank, idx in enumerate(I[0]):
        if idx == -1: continue
        d = DOCS[idx]
        hits.append({"rank": rank+1, "title": d["title"], "text": d["text"], "source": d["source"]})
    return hits

# --------------------
# Vision model
# --------------------
def load_vision():
    model = efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    if os.path.exists(DISEASE_MODEL_PATH):
        state = torch.load(DISEASE_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state)
    model.eval().to(DEVICE)
    return model

vision_model = load_vision()
IMG_TF = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def predict_leaf(img: Image.Image) -> Tuple[str, float]:
    with torch.no_grad():
        t = IMG_TF(img.convert("RGB")).unsqueeze(0).to(DEVICE)
        logits = vision_model(t)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    i = int(np.argmax(probs))
    return CLASS_NAMES[i], float(probs[i])

# --------------------
# Optional: OpenAI compose (if key present)
# --------------------
def llm_compose(user_q: str, bullets: List[str]) -> str:
    if not OPENAI_API_KEY:
        # fallback: simple extractive composition
        return "Here’s what I found:\n" + "\n".join(f"• {b}" for b in bullets)
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        prompt = (
            "You are AgriAdvisorLite, an expert on Indian agriculture schemes/loans.\n"
            "Draft a clear, helpful answer using only these facts:\n" +
            "\n".join(f"- {b}" for b in bullets) +
            f"\n\nUser question: {user_q}\nAnswer in 5-8 concise bullet points."
        )
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":prompt}],
            temperature=0.2
        )
        return resp["choices"][0]["message"]["content"]
    except Exception:
        return "Here’s what I found:\n" + "\n".join(f"• {b}" for b in bullets)

# --------------------
# Schemas
# --------------------
class Ask(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    k: Optional[int] = TOP_K
    user_id: Optional[int] = None

class DetectPayload(BaseModel):
    image_base64: str
    user_id: Optional[int] = None

class Feedback(BaseModel):
    chat_id: int
    rating: int = Field(..., ge=1, le=5)
    note: Optional[str] = ""

class NewUser(BaseModel):
    name: str
    email: str
    phone: Optional[str] = ""

class NewScheme(BaseModel):
    title: str
    text: str

# --------------------
# Routes
# --------------------
@app.get("/api/health")
def health():
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM chats"); chats = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM detections"); detections = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM users"); users = cur.fetchone()[0]
    conn.close()
    return {
        "ok": True,
        "device": DEVICE,
        "docs": len(DOCS),
        "index_ready": INDEX is not None,
        "chats": chats, "detections": detections, "users": users
    }

@app.post("/api/users")
def create_user(user: NewUser, ip: str = Depends(get_client_ip), _=Depends(require_key)):
    rate_limiter(ip)
    conn = db()
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO users(name,email,phone,created_at) VALUES(?,?,?,?)",
                (user.name, user.email, user.phone, datetime.utcnow().isoformat()))
    conn.commit()
    cur.execute("SELECT id,name,email,phone,created_at FROM users WHERE email=?", (user.email,))
    row = cur.fetchone()
    conn.close()
    return {"user": {"id": row[0], "name": row[1], "email": row[2], "phone": row[3], "created_at": row[4]}}

@app.post("/api/answer")
def answer(q: Ask, ip: str = Depends(get_client_ip), _=Depends(require_key)):
    rate_limiter(ip)
    hits = retrieve(q.message, q.k or TOP_K)
    if not hits:
        text = "I don't have that in my knowledge base yet. Try another query or add more scheme notes."
        citations = []
    else:
        bullets = [h["text"] for h in hits]
        text = llm_compose(q.message, bullets)
        citations = [{"title": h["title"], "source": h["source"], "rank": h["rank"]} for h in hits]

    conn = db()
    cur = conn.cursor()
    cur.execute("INSERT INTO chats(user_id,message,answer,citations,created_at) VALUES(?,?,?,?,?)",
                (q.user_id, q.message, text, json.dumps(citations), datetime.utcnow().isoformat()))
    chat_id = cur.lastrowid
    conn.commit(); conn.close()

    return {"chat_id": chat_id, "answer": text, "citations": citations}

@app.post("/api/feedback")
def give_feedback(fb: Feedback, ip: str = Depends(get_client_ip), _=Depends(require_key)):
    rate_limiter(ip)
    conn = db()
    cur = conn.cursor()
    cur.execute("INSERT INTO feedback(chat_id,rating,note,created_at) VALUES(?,?,?,?)",
                (fb.chat_id, fb.rating, fb.note, datetime.utcnow().isoformat()))
    conn.commit(); conn.close()
    return {"ok": True}

@app.post("/api/detect")
def detect(payload: DetectPayload, ip: str = Depends(get_client_ip), _=Depends(require_key)):
    rate_limiter(ip)
    b64 = payload.image_base64
    if "," in b64:  # strip data URL
        b64 = b64.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    label, conf = predict_leaf(img)

    conn = db()
    cur = conn.cursor()
    cur.execute("INSERT INTO detections(user_id,label,confidence,created_at) VALUES(?,?,?,?)",
                (payload.user_id, label, conf, datetime.utcnow().isoformat()))
    conn.commit(); conn.close()

    return {"label": label, "confidence": round(conf, 4)}

@app.post("/api/detect-file")
async def detect_file(file: UploadFile = File(...), ip: str = Depends(get_client_ip), _=Depends(require_key)):
    rate_limiter(ip)
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    label, conf = predict_leaf(img)
    return {"label": label, "confidence": round(conf, 4)}

# ---------- Admin ----------
@app.post("/admin/reload")
def admin_reload(ip: str = Depends(get_client_ip), _=Depends(require_key)):
    rate_limiter(ip)
    _build_index()
    return {"ok": True, "docs": len(DOCS), "index_ready": INDEX is not None}

@app.post("/admin/schemes")
def admin_add_scheme(s: NewScheme, ip: str = Depends(get_client_ip), _=Depends(require_key)):
    rate_limiter(ip)
    # write to file -> rebuild
    safe = re.sub(r"[^a-zA-Z0-9_\-]+", "_", s.title.strip())[:80]
    path = os.path.join(SCHEMES_DIR, f"{safe}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(s.text.strip())
    _build_index()
    return {"ok": True, "saved_as": path, "docs": len(DOCS)}

@app.get("/admin/search")
def admin_search(q: str = Query(..., min_length=2), k: int = TOP_K, ip: str = Depends(get_client_ip), _=Depends(require_key)):
    rate_limiter(ip)
    return {"hits": retrieve(q, k)}

# -------------- Run --------------
# Run: uvicorn server:app --reload
