"""Configuration settings for the face sorter application."""

import os

# Database configuration
DATABASE_PATH = "faces.db"

# Face detection configuration
FACE_DETECTION_CONFIDENCE = 0.9  # MTCNN confidence threshold
MIN_FACE_SIZE = 40  # Minimum face size in pixels

# Face recognition configuration
SIMILARITY_THRESHOLD = 0.6  # Lower = more similar (Euclidean distance)
CLUSTERING_DISTANCE_THRESHOLD = 0.6  # For DBSCAN clustering

# Output configuration
OUTPUT_DIR = "output"
COPY_ORIGINAL_IMAGES = True

# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}

# UI configuration
PREVIEW_IMAGE_SIZE = (200, 200)  # Size for preview thumbnails
PREVIEW_GRID_COLUMNS = 4  # Number of columns in the preview grid

# Paths
def get_output_dir():
    """Get the output directory path."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR
