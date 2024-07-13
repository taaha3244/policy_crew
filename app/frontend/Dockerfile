# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the frontend directory contents into the container at /app/frontend
COPY ./app/frontend /app/frontend

# Copy the backend directory contents into the container at /app/backend
COPY ./app/backend /app/backend

# Install any needed packages specified in frontend/requirements.txt
RUN pip install --no-cache-dir -r /app/frontend/requirements.txt

# Set environment variable
ENV PYTHONPATH="${PYTHONPATH}:/app/backend"

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run Streamlit when the container launches
CMD ["streamlit", "run", "/app/frontend/app.py"]
