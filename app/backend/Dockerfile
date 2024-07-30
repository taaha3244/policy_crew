# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container to match the project root
WORKDIR /policy_crew

# Set the PYTHONPATH environment variable to the working directory
ENV PYTHONPATH=/policy_crew

# Copy the requirements file first to leverage Docker cache
COPY app/backend/requirements.txt /policy_crew/app/backend/requirements.txt

# Copy the entire project to the working directory
COPY . /policy_crew

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r /policy_crew/app/backend/requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run FastAPI when the container launches
CMD ["uvicorn", "app.backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
