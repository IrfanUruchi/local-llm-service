from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.model_utils import load_model, generate_response

app = FastAPI(title="ML")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

model, tok, _device = load_model()

class ChatRequest(BaseModel):
    prompt: str = Field(..., description="User message")

@app.post("/chat")
async def chat(req: ChatRequest):
    text = generate_response(model, tok, req.prompt)
    return {"response": text}

@app.get("/", include_in_schema=False)
async def index():
    return FileResponse("web/index.html")

app.mount("/static", StaticFiles(directory="web"), name="static")
