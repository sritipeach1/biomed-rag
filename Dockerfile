FROM python:3.11-slim

# libgomp is needed by faiss. Run as a normal user (Hugging Face Spaces expects user id 1000).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 user
ENV HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PORT=7860
WORKDIR /app

# The CPU-only build of PyTorch first. The default wheel pulls gigabytes of GPU libraries
# that a free CPU host cannot use.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# An editable install keeps the code in /app/src, so config.ROOT (and with it index/ and static/)
# is /app. A normal install would put the code in site-packages and break those paths.
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .

COPY static ./static
RUN mkdir -p /app/index && chown -R user:user /app /home/user
USER user

# Download the two small models now, so the first visitor does not wait for them.
RUN python -c "from biomedrag import config; \
from sentence_transformers import CrossEncoder, SentenceTransformer; \
SentenceTransformer(config.EMBED_MODEL); CrossEncoder(config.RERANK_MODEL)"

EXPOSE 7860
CMD ["sh", "-c", "uvicorn biomedrag.api.app:app --host 0.0.0.0 --port ${PORT}"]
