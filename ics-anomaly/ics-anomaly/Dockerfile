# NOTE: authored without a Docker daemon available; build and test it on your machine before relying on it.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir -e . && python -m icsguard train --demo --out models/demo --budget 5
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["python", "-m", "icsguard", "serve", "--model", "models/demo.npz", "--stream", "models/demo_stream.npz", "--port", "8000"]
