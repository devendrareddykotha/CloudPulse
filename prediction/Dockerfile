FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app

COPY prediction/requirements.txt /app/prediction/requirements.txt
RUN pip install --no-cache-dir -r /app/prediction/requirements.txt

COPY prediction /app/prediction
COPY results /app/results