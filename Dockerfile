FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgeos-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Persist paper state and logs across Railway restarts.
RUN mkdir -p /data \
    && rm -rf /app/data \
    && ln -sfn /data /app/data

ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/data

CMD ["python", "-m", "paris_today_bot.main", "--serve-paper", "--interval", "60"]
