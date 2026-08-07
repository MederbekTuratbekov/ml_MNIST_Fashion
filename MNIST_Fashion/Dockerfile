FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /fashion_app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libsndfile1 \
    gcc \
    g++ \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==2.2.0+cpu \
    torchvision==0.17.0+cpu \
    torchaudio==2.2.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html && \
    pip cache purge && \
    find /usr -name "*.pyc" -delete && \
    find /usr -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null; true

COPY ml_req_deploy.txt .
RUN pip install --no-cache-dir -r ml_req_deploy.txt

COPY main.py .
COPY model_CheckImage_MnistFashion.pth .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]