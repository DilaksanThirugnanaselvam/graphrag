FROM python:3.12-slim

# Install build dependencies for asyncpg
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libc-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY schema.sql /app/schema.sql

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "scripts/run_indexing.py"]