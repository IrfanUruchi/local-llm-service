# Local LLM Service

Self-hosted local LLM service with:

- Persistent multi-chat history
- Conversation rename support
- Default baked-in GGUF model
- Model override via environment variables
- Multi-architecture Docker image (amd64 + arm64)
- Clean minimal dark UI
- MIT Licensed

---

## Quick Start

Run instantly:

```bash
docker run -p 8000:8000 irfanuruchi/local-llm-service:latest
```

Then open 

```bash
http://localhost:8000
```

---

## Eanble persistence

To persist conversations between restarts:

```bash
docker run \
  -v $(pwd)/data:/data \
  -p 8000:8000 \
  irfanuruchi/local-llm-service:latest
```

---

## Override model

Use your own GGUF model:
```bash
docker run \
  -v $(pwd)/models:/models \
  -e MODEL_PATH=/models/custom.gguf \
  -p 8000:8000 \
  irfanuruchi/local-llm-service:latest
```

---

## Environment variables

| Variable       | Default                     | Description                    |
|---------------|-----------------------------|--------------------------------|
| MODEL_PATH     | /models/model.gguf          | Path to GGUF model             |
| MODEL_NAME     | local-llm                   | Model display name             |
| N_GPU_LAYERS   | 35                          | GPU layers for inference       |
| N_CTX          | 4096                        | Context window size            |
| CHAT_DB_PATH   | /data/chat_history.db       | SQLite DB location             |

---

# Arxhitecture

- FastAPI backend
- SQLite persistent storage
- llama-cpp-python inference runtime
- Multi-stage Docker build
- OCI image metadata
- Multi-platform build (linux/amd64, linux/arm64)

---

# Docker image 

Docker Hub:

https://hub.docker.com/r/irfanuruchi/local-llm-service
