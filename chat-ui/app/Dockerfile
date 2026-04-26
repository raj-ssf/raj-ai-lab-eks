FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY chainlit.md .
COPY launch.py .

RUN useradd -u 1000 -m -s /usr/sbin/nologin app && chown -R app:app /app
USER 1000

EXPOSE 8000

# launch.py monkeypatches uvicorn.Config to inject proxy_headers=True
# and forwarded_allow_ips='*' before invoking Chainlit's normal CLI.
# Required because Chainlit's CLI doesn't pass these flags itself —
# without them, request.url_for() inside Chainlit's OAuth error path
# reads the upstream Host (chat-ui.chat.svc.cluster.local — set by
# NGINX's upstream-vhost rewrite for Istio mTLS routing) and
# constructs DNS-unresolvable redirect URLs.
CMD ["python", "launch.py"]
