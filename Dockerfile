FROM python:3.12-slim

WORKDIR /code

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual application code
COPY ./app ./app

EXPOSE 8000

# --host 0.0.0.0 is REQUIRED inside Docker — "localhost" inside a
# container refers only to the container itself, unreachable from
# outside. 0.0.0.0 means "listen on all network interfaces."
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]