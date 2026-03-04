ARG BASE_IMAGE=python:3.11-slim
ARG DEFAULT_MODEL_URL=https://huggingface.co/second-state/Phi-3-mini-4k-instruct-GGUF/resolve/main/Phi-3-mini-4k-instruct-Q4_0.gguf
FROM ${BASE_IMAGE} AS build

LABEL org.opencontainers.image.title="Local LLM Service" \
      org.opencontainers.image.version="1.0" \
      org.opencontainers.image.description="Self-hosted local LLM service with persistent multi-chat history and optional model override." \
      org.opencontainers.image.authors="Irfan"
WORKDIR /wheels

COPY app/requirements.txt ./requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git \
    && pip wheel -r requirements.txt \
    && rm -rf /var/lib/apt/lists/*

FROM ${BASE_IMAGE}
ARG DEFAULT_MODEL_URL
WORKDIR /app
RUN mkdir -p /models
RUN mkdir -p /data
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/models/model.gguf
ENV MODEL_NAME=local-llm
ENV N_GPU_LAYERS=35
ENV N_CTX=4096
ENV CHAT_DB_PATH=/data/chat_history.db

COPY --from=build /wheels /tmp/wheels
RUN pip install --no-cache-dir --no-index --find-links=/tmp/wheels /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels

RUN apt-get update \
    && apt-get install -y --no-install-recommends wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN if [ ! -f /models/model.gguf ]; then \
        echo "Downloading default model..." && \
        wget -c --show-progress ${DEFAULT_MODEL_URL} -O /models/model.gguf; \
    else \
        echo "Model already exists, skipping download."; \
    fi

COPY app /app/app
COPY web /app/web  

EXPOSE 8000
VOLUME ["/models", "/data"]
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]

