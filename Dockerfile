FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    gfortran \
    libhdf5-dev \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY README.md .

RUN pip install --no-cache-dir -e ".[dev]"

COPY . .

CMD ["bash"]
