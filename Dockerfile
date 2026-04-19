FROM python:3.13-slim

WORKDIR /app

# Minimal system library needed by OpenCV headless for GLib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps, then force-replace opencv-python (pulled by
# ultralytics) with the headless variant so no X11/libxcb is needed.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python 2>/dev/null; \
    pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless>=4.8.0

COPY . .

# Railway injects PORT; app.py already reads it from the environment
EXPOSE ${PORT:-5001}
CMD ["python", "app.py"]
