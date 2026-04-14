FROM python:3.12-slim

WORKDIR /app

COPY requirements-common.txt /requirements-common.txt
COPY services/mcp/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

CMD ["python", "-m", "uvicorn", "services.mcp.app.main:app", "--host", "0.0.0.0", "--port", "8080"]
