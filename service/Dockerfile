FROM python:3.12-slim

WORKDIR /app

COPY service/requirements.txt /app/service/requirements.txt
RUN pip install --no-cache-dir -r /app/service/requirements.txt

COPY service /app/service
COPY testing /app/testing
COPY evaluation /app/evaluation
RUN mkdir -p /app/results/raw /app/results/processed

EXPOSE 8000

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]