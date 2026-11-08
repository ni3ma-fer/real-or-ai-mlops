FROM python:3.13-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY monitoring/drift/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY monitoring/drift/drift.py .
RUN useradd --create-home appuser
USER appuser
EXPOSE 8001
CMD ["python", "drift.py"]
