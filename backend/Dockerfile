FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt
COPY backend/app ./app
COPY frontend /frontend
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
