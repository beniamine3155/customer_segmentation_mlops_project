# Use an official Python 3.10 image from Docker Hub
FROM python:3.10-slim-buster

# Set the working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt /app/

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . /app

# Create a non-root user for security (Hugging Face best practice)
RUN useradd -m -u 1000 user
USER user

# Expose the port FastAPI will run on
EXPOSE 7860

# Command to run the FastAPI app
CMD ["python3", "app.py"]