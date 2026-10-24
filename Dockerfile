FROM python:3.12.14-slim-bookworm
LABEL org.opencontainers.image.title="Options surface risk competition submission"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    SERVICE_HOST=0.0.0.0 SERVICE_PORT=8000 RISK_DEPLOYMENT_PROFILE=competition
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-predictive.lock.txt ./
RUN python -m pip install --no-cache-dir -r requirements-predictive.lock.txt \
    && groupadd --gid 10001 risk && useradd --uid 10001 --gid risk --create-home risk
COPY --chown=risk:risk . /app
USER risk
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python scripts/healthcheck.py
CMD ["python", "scripts/serve_submission.py"]
