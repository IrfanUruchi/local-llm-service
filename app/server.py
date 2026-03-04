# app/server.py
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from llama_cpp import Llama
import os
from os import getenv

import sqlite3
import uuid
from datetime import datetime

try:
    from model_utils import maybe_solve_direct
except ImportError:
    def maybe_solve_direct(_):
        return None

MODEL_PATH   = getenv("MODEL_PATH", "/models/model.gguf")
MODEL_NAME   = getenv("MODEL_NAME", "local-llm")
MAX_TOKENS   = 512
TEMPERATURE  = 0.20
TOP_P        = 0.90
N_GPU_LAYERS = 35

CHAT_DB_PATH = getenv("CHAT_DB_PATH", "/app/data/chat_history.db")

db_dir = os.path.dirname(CHAT_DB_PATH)
if db_dir:
    os.makedirs(db_dir, exist_ok=True)

conn = sqlite3.connect(CHAT_DB_PATH, check_same_thread=False)

conn.execute("PRAGMA journal_mode=WAL;")
conn.execute("PRAGMA synchronous=NORMAL;")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    created_at TEXT,
    title TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT,
    role TEXT,
    content TEXT,
    timestamp TEXT
)
""")
conn.commit()

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=0,
    n_gpu_layers=N_GPU_LAYERS,
)

SYSTEM_PROMPT = (
    "You are a precise and professional AI assistant running locally.\n"
    "• Provide clear, direct answers.\n"
    "• Avoid unnecessary verbosity.\n"
    "• If uncertain, say you do not know instead of guessing."
)

CHAT_TEMPLATE = (
    "<|system|>{system}<|end|>\n"
    "<|user|>{question}<|end|>\n"
    "<|assistant|>"
)
STOP_WORDS = ["<|end|>"]

app = FastAPI(title="Local LLM Service API")

class ChatReq(BaseModel):
    prompt: str
    conversation_id: str | None = None


@app.post("/chat")
def chat(req: ChatReq):

    conversation_id = req.conversation_id or str(uuid.uuid4())

    cursor.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,))
    if not cursor.fetchone():
        cursor.execute(
            "INSERT INTO conversations (id, created_at, title) VALUES (?, ?, ?)",
            (conversation_id, datetime.utcnow().isoformat(), "New Chat"),
        )
        conn.commit()

    cursor.execute(
        "INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (conversation_id, "user", req.prompt.strip(), datetime.utcnow().isoformat()),
    )
    conn.commit()

    cursor.execute("SELECT title FROM conversations WHERE id = ?", (conversation_id,))
    current_title = cursor.fetchone()

    if current_title and current_title[0] == "New Chat":
        auto_title = req.prompt.strip()[:45]
        cursor.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            (auto_title, conversation_id),
        )
        conn.commit()

    direct = maybe_solve_direct(req.prompt)
    if direct:
        answer = direct.split("Final Answer:")[-1].strip()
        
        cursor.execute(
            "INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (conversation_id, "assistant", answer, datetime.utcnow().isoformat()),
        )
        conn.commit()
        return {"response": answer, "conversation_id": conversation_id}

    prompt = CHAT_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        question=req.prompt.strip(),
    )

    out = llm(
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        stop=STOP_WORDS,
    )["choices"][0]["text"]

    answer = out.strip()
    
    cursor.execute(
        "INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (conversation_id, "assistant", answer, datetime.utcnow().isoformat()),
    )
    conn.commit()

    return {"response": answer, "conversation_id": conversation_id}


@app.get("/", include_in_schema=False)
def root():
    return FileResponse("web/index.html") 

@app.get("/health")
def health():
    return {"status": "running"}

@app.get("/info")
def info():
    return {
        "service": "Local LLM Service",
        "model": MODEL_NAME,
        "model_path": MODEL_PATH,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE
    }


@app.get("/history/{conversation_id}")
def get_history(conversation_id: str):
    cursor.execute(
        "SELECT role, content, timestamp FROM messages WHERE conversation_id = ? ORDER BY id ASC",
        (conversation_id,),
    )
    rows = cursor.fetchall()
    return {
        "conversation_id": conversation_id,
        "messages": [
            {"role": r[0], "content": r[1], "timestamp": r[2]} for r in rows
        ],
    }

@app.get("/conversations")
def list_conversations():
    cursor.execute("SELECT id, created_at, title FROM conversations ORDER BY created_at DESC")
    rows = cursor.fetchall()
    return {
        "conversations": [
            {"id": r[0], "created_at": r[1], "title": r[2] or "New Chat"} for r in rows
        ]
    }


@app.put("/conversation/{conversation_id}/title")
def rename_conversation(conversation_id: str, data: dict):
    new_title = (data.get("title") or "").strip()[:45]
    if not new_title:
        return {"error": "Title cannot be empty"}

    cursor.execute(
        "UPDATE conversations SET title = ? WHERE id = ?",
        (new_title, conversation_id),
    )
    conn.commit()
    return {"status": "updated", "title": new_title}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")
