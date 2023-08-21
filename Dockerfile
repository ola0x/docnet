FROM python:3.8-slim-buster

RUN apt-get update -y --no-install-recommends

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable for Flask to run in production mode
ENV FLASK_ENV=production

RUN ls -la /app/

# Run app.py when the container launches
CMD ["python", "api.py"]