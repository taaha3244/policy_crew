# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container to match the project root
WORKDIR /policy_crew

# Set the PYTHONPATH environment variable to the working directory
ENV PYTHONPATH=/policy_crew

# Copy the requirements file first to leverage Docker cache
COPY app/frontend/requirements.txt /policy_crew/app/frontend/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r /policy_crew/app/frontend/requirements.txt

# Copy the entire project to the working directory
COPY . /policy_crew

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run Streamlit when the container launches
CMD ["streamlit", "run", "app/frontend/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
