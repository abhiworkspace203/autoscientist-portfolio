FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir numpy pandas scipy scikit-learn matplotlib

RUN groupadd --system sandboxuser && useradd --system --gid sandboxuser --create-home sandboxuser

WORKDIR /workspace
USER sandboxuser