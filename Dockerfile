# Use an official Python runtime as a base image
FROM python:3.12.2

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install necessaary libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Create raw_images directory
RUN mkdir -p /app/raw_images

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV NAME World

# Run app.py when the container launches using Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000"]
