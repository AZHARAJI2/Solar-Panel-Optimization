# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory in the container
# Set the working directory to where the app code is
WORKDIR /app/src

# Copy the dependency file to the root of the app
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the entire src directory into the container
COPY src/ /app/src/

# Copy the model file to the parent directory (where app expects it)
COPY solar_model.pkl /app/

# Expose the Flask port
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
