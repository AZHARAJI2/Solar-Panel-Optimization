# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependency file
COPY requirements.txt .

# Install dependencies
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy the application source code and the saved model
COPY src/ ./src/
COPY solar_model.pkl .

# Expose the Flask port
EXPOSE 5000

# Command to run the application
CMD ["python", "src/app.py"]
