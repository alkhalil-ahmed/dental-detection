## Stage 1: fetch LFS model files from public repo ─────────────────────────
FROM alpine/git:latest AS lfs
RUN git lfs install
RUN git clone --filter=blob:none --no-checkout \
        https://github.com/alkhalil-ahmed/dental-detection.git /repo && \
    cd /repo && \
    git lfs fetch origin main --include="models/*.pt" && \
    git checkout HEAD -- models/

## Stage 2: runtime image ───────────────────────────────────────────────────
FROM python:3.13-slim

WORKDIR /app

# Minimal system lib for OpenCV headless (GLib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps, then force-replace opencv-python (pulled by
# ultralytics) with the headless variant so no X11/libxcb is needed.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python 2>/dev/null; \
    pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless>=4.8.0

# Copy project files (LFS pointers for .pt files)
COPY . .

# Overwrite LFS pointers with actual model binaries from stage 1
COPY --from=lfs /repo/models/ ./models/

# Railway injects PORT; app.py already reads it from the environment
EXPOSE ${PORT:-5001}
CMD ["python", "app.py"]
