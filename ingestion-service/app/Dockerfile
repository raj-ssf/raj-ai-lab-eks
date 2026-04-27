FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Phase 3 will need OS-level deps for unstructured (libgl1, libglib2,
# tesseract-ocr, poppler-utils for PDF rendering). Adding them now keeps
# the Phase 3 Dockerfile diff smaller and the layer cache stable.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libmagic1 \
        poppler-utils \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data so Unstructured's sent_tokenize / POS tagger
# don't try to runtime-fetch a tarball from Unstructured's S3 bucket
# at first /upload call. That URL returns HTTP 403 from our pod
# (and possibly everywhere — bundle is stale or now access-restricted).
#
# Unstructured's check_for_nltk_package() calls nltk.data.find() for
# packages like tokenizers/punkt/english.pickle. If we populate
# NLTK_DATA at build time using NLTK's own download (which goes to
# raw.githubusercontent.com — public, reachable), Unstructured's
# "if not present" check returns True and skips its broken download.
ENV NLTK_DATA=/usr/share/nltk_data
RUN mkdir -p $NLTK_DATA && \
    python -c "import nltk; \
        nltk.download('punkt', download_dir='$NLTK_DATA'); \
        nltk.download('punkt_tab', download_dir='$NLTK_DATA'); \
        nltk.download('averaged_perceptron_tagger', download_dir='$NLTK_DATA'); \
        nltk.download('averaged_perceptron_tagger_eng', download_dir='$NLTK_DATA')"

COPY main.py .

RUN useradd -u 1000 -m -s /usr/sbin/nologin app && chown -R app:app /app
USER 1000

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
