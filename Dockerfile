# The photo booth web app: FastAPI + the CV pipeline (rembg, OpenPose, MiDaS).
#
# Deliberately does NOT contain ComfyUI. They are separate concerns with
# different hardware needs and wildly different image sizes, and coupling them
# would mean rebuilding a multi-gigabyte image to change a line of Python.
# COMFY_ADDRESS points this container at a ComfyUI instance wherever it lives
# -- another container (see docker-compose.yml), another host on the LAN, or a
# GPU box entirely.
#
# Note on what this image is for: it runs the *orchestration* half. The CV
# stages inside it are CPU-only here, matching how the app already runs
# natively (see the onnxruntime-gpu/CUDA-13 note in the README) -- so this
# image needs no GPU runtime and stays portable. The GPU work happens in
# ComfyUI.

FROM python:3.12-slim AS base

# libglib/libgl: OpenCV's shared-library dependencies. Even the headless wheel
# links against these, and their absence surfaces as a bare "ImportError:
# libGL.so.1" on the first `import cv2` -- which reads as a broken build
# rather than a missing OS package.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Requirements first, as their own layer: the dependency install is the slow
# part (torch and friends), and copying source before it would invalidate that
# layer on every code change.
COPY requirements.txt ./
# torch comes in via controlnet_aux, and on Linux pip defaults to the CUDA
# build: ~6GB of nvidia/cu* wheels for a container that, by the design note
# above, never touches a GPU. Pinning the CPU index first means the later
# `torch` requirement is already satisfied, and the image drops from ~9.4GB to
# ~3GB. If this image is ever given a GPU, delete this line -- don't try to
# have both.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
    && pip install --no-cache-dir -r requirements.txt

# Model weights for rembg/controlnet_aux download on first use and are large.
# Kept on a volume (see docker-compose.yml) so a rebuilt container doesn't
# re-download them, and pointed somewhere writable by the non-root user below.
ENV U2NET_HOME=/models/rembg \
    TORCH_HOME=/models/torch \
    HF_HOME=/models/huggingface

COPY . .

# Non-root, because there is no reason for this process to be root and a
# container that only needs to read its own source shouldn't be.
RUN useradd --create-home --uid 10001 booth \
    && mkdir -p /models \
    && chown -R booth:booth /app /models
USER booth

EXPOSE 8000

# ComfyUI's address is the one thing that genuinely differs per deployment.
ENV COMFY_ADDRESS=host.docker.internal:8188 \
    GENERATION_WORKERS=2

# A container is healthy once the app can answer, which is only after the
# ~30-60s model warmup in the lifespan handler -- hence the generous
# start_period rather than a short interval that would flap during startup.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/queue', timeout=4)"

# --host 0.0.0.0 rather than the app's 127.0.0.1 default: inside a container
# the loopback interface is only reachable from within the container itself,
# so the default would publish a port nothing could connect to.
CMD ["python", "web_server.py", "--host", "0.0.0.0", "--port", "8000"]
