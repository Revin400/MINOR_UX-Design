# Gebruik een officiële Python image
FROM python:3.10-slim

# Zet working directory
WORKDIR /app

# Kopieer projectbestanden
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Installeer Python dependencies
RUN pip install numpy matplotlib scipy pillow opencv-python

# Standaard command
CMD ["python"]
