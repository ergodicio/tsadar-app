# Thomson analysis browser: one image serving the SPA and the API.
#
# Separate from docker/app (Streamlit) and docker/runner (the Batch job) -- this
# is the browser from ergodicio/tsadar-app#37, and it is the image
# ergodicio/continuum-infra#132 pins.
#
# Build from the repository root:
#   docker build -f docker/browser/Dockerfile -t thomson-browser:dev .

# -- stage 1: build the SPA ----------------------------------------------------
FROM node:22-slim AS frontend

WORKDIR /build

# Dependencies first, so a source-only change does not re-resolve the tree.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
# The generated client is committed, so this needs no Python: the check that it
# matches the backend runs in CI, not here.
RUN npm run build

# -- stage 2: serve ------------------------------------------------------------
FROM python:3.11-slim AS runtime

# h5py and netCDF reading come from wheels, so no build toolchain is needed.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-browser.txt ./
RUN pip install --no-cache-dir -r requirements-browser.txt

COPY tsadar_browser/ ./tsadar_browser/
COPY --from=frontend /build/dist/ ./static/

# Where the SPA lives, and a cache location inside the container. In the Fargate
# task the cache is ephemeral local storage; it is a cache, so losing it on a
# task replacement costs a re-download and nothing else.
ENV STATIC_DIR=/app/static \
    CACHE_DIR=/var/cache/tsadar-browser \
    CACHE_MAX_GB=5 \
    PORT=8000

# Same-origin in the container, so no CORS is needed.
ENV CORS_ORIGINS=""

RUN mkdir -p /var/cache/tsadar-browser && \
    useradd --create-home --uid 10001 browser && \
    chown -R browser:browser /app /var/cache/tsadar-browser
USER browser

EXPOSE 8000

# The ALB target group health-checks this path. It answers 200 even when MLflow
# is unreachable -- reporting `degraded` rather than failing, so a tracking-server
# outage does not make the load balancer recycle otherwise healthy tasks.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

# One worker: the artifact cache is per-process (its LRU bookkeeping and download
# deduplication are in-process locks), so multiple workers would each keep their
# own view of it. Scale with tasks rather than workers.
CMD ["sh", "-c", "uvicorn tsadar_browser.app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
