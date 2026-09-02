# ============================================================
# Multi-stage Dockerfile for ControlPlane.ai
# Stage 1: builder  — installs all Python deps
# Stage 2: runtime  — lean image, non-root user, healthcheck
# ============================================================

FROM python:3.11-slim AS builder

WORKDIR /build

# Install system build deps (needed for some compiled wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ============================================================
FROM python:3.11-slim AS runtime
# ============================================================

# Install curl for HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (UID 1000)
RUN useradd --uid 1000 --no-create-home --shell /sbin/nologin appuser

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application source
COPY backend   ./backend
COPY frontend  ./frontend
COPY policies  ./policies
COPY rag       ./rag
COPY ml        ./ml
COPY rlhf      ./rlhf
COPY .env.example .

# Create writable data directory owned by appuser
RUN mkdir -p /data && chown appuser:appuser /data

# Switch to non-root
USER appuser

# Bake structured JSON logs on for all containers by default
ENV CONTROLPLANE_JSON_LOGS=true \
    CONTROLPLANE_DB_PATH=/data/controlplane.db \
    CONTROLPLANE_POLICIES_DIR=/app/policies

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]