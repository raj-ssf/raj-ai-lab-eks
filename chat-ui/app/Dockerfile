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

RUN useradd -u 1000 -m -s /usr/sbin/nologin app && chown -R app:app /app
USER 1000

EXPOSE 8000

# Chainlit's launcher binds 0.0.0.0:8000 by default but its CLI manages
# the asgi server lifecycle. -h disables the open-browser side effect
# that's noisy in container logs.
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000", "-h"]
