FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application and pre-built official Strudel REPL static website
COPY app.py .
COPY website/dist ./website/dist

EXPOSE 7860

CMD ["python", "app.py"]
