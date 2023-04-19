FROM python:3.11.2-slim

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY . /

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]