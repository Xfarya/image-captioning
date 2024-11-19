# Use the official Python image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy application files
COPY app.py /app/
COPY static/ static/



# Update pip to the latest version
RUN pip install --upgrade pip

# Install dependencies
RUN pip install fastapi
RUN pip install uvicorn
RUN pip install transformers
RUN pip install torch
RUN pip install pillow
RUN pip install python-multipart
RUN pip install --no-cache-dir fastapi

# Expose port 80
EXPOSE 80

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
