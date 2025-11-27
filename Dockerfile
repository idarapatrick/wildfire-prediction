# 1. Base image
FROM python:3.9-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy dependencies
COPY requirements.txt .

# 4. Install dependencies (and libgl1 for cv2 if needed later, though we stick to tf)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application code
COPY . .

# 6. Expose the port FastAPI runs on
EXPOSE 8000

# 7. Command to run the API
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]