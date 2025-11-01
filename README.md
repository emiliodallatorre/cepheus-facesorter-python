# Cepheus Face Sorter

A Python application with CLI and Web Dashboard for detecting, clustering, labeling, and sorting faces from images using TensorFlow models.

## Features

- 🔍 **Face Detection**: Detect faces in images using MTCNN (TensorFlow-based)
- 🧠 **Face Recognition**: Generate face embeddings using FaceNet
- 🔗 **Smart Clustering**: Automatically cluster faces by similarity (hierarchical clustering)
- � **Web Dashboard**: Interactive dashboard to view, manage, and refine clusters
- 🏷️ **Interactive Labeling**: Remove incorrect faces before confirming clusters
- 📁 **Face Sorting**: Organize faces into folders by person
- 💾 **SQLite Database**: Cache embeddings and store all face data

## New Workflow (Recommended)

### 1. Scan Photos
```bash
python main.py scan /path/to/photos
```
Scans all pictures, detects faces, generates and caches embeddings in SQLite.

### 2. Cluster Faces
```bash
python main.py cluster
```
Automatically groups similar faces into clusters (one cluster per person ideally).
- Uses hierarchical clustering with auto-detection
- Faces that don't match well become outliers
- Minimum 2 faces per cluster (configurable with `--min-size`)

### 3. Launch Dashboard
```bash
python main.py dashboard
```
Opens web interface at `http://127.0.0.1:5000`

**Dashboard Features:**
- View all unlabeled clusters with preview images
- Click a cluster to see all faces with similarity distances
- Remove faces that don't belong (color-coded by distance)
- Assign name, surname, Instagram to cluster
- System refines cluster definition based on your removals
- Export all labeled faces to folders when ready

### 4. Iterative Refinement
Your face removals improve the cluster:
- Cluster center is recalculated without removed faces
- Better accuracy for future matching
- Labeled data is marked as "confirmed"

### 5. Export to Folders
Click "Export All to Folders" in dashboard or:
```bash
python main.py sort
```
Creates `output/Name_Surname/` folders with:
- Extracted face images
- Original images (optional)

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Scan your photos
python main.py scan ./photos

# 2. Cluster faces automatically
python main.py cluster

# 3. Launch dashboard
python main.py dashboard

# 4. Open browser to http://127.0.0.1:5000
# 5. Label clusters interactively
# 6. Click "Export" when done
```

## CLI Commands

### scan
Scan directory for images and detect faces:
```bash
python main.py scan /path/to/images [--force]
```
- Automatically skips already processed images
- Use `--force` to re-process all images

### cluster  
Cluster faces using hierarchical clustering:
```bash
python main.py cluster [--threshold 0.5] [--min-size 2]
```
- `--threshold`: Distance threshold (default 0.5, lower = stricter)
- `--min-size`: Minimum faces per cluster (default 2)

### dashboard
Launch web dashboard:
```bash
python main.py dashboard [--host 127.0.0.1] [--port 5000]
```

### clean
Remove duplicate faces from database:
```bash
python main.py clean
```

### stats
View database statistics:
```bash
python main.py stats
```

### sort
Export labeled faces to folders:
```bash
python main.py sort [--output ./output]
```

## Configuration

Edit `config.py` to customize:
- `CLUSTERING_DISTANCE_THRESHOLD = 0.5` - Clustering threshold (lower = smaller, stricter clusters)
- `FACE_DETECTION_CONFIDENCE = 0.9` - Minimum face detection confidence
- `OUTPUT_DIR = "output"` - Where to save sorted faces
- `COPY_ORIGINAL_IMAGES = True` - Copy full images to person folders

## Database Schema

### Persons Table
- id, name, surname, instagram
- is_confirmed (marked after labeling)
- cluster_center (refined center after removals)

### Faces Table
- id, person_id, cluster_id
- original_image_path, face_coordinates
- face_encoding (cached 512-dim FaceNet embedding)
- is_removed (marked when excluded from cluster)
- confidence, extracted_at

### Images Table
- id, file_path, person_id
- has_faces, manually_assigned
- Tracks all images including those without faces

## Tips

1. **Optimal Clustering**: Adjust `--threshold` if you get too many/few clusters:
   - Too many small clusters → increase threshold (try 0.6)
   - Too few large clusters → decrease threshold (try 0.4)

2. **Face Distances**: In dashboard, check distance colors:
   - Green (< 0.3): Very similar
   - Blue (0.3-0.4): Similar  
   - Orange (0.4-0.5): Possibly same
   - Red (> 0.5): Probably different - consider removing

3. **Outliers**: Single faces or poorly matched faces become outliers (cluster_id = -1)
   - Can be processed separately or manually assigned

4. **Incremental Scanning**: Re-run scan on same folder - only new images are processed

5. **GPU Acceleration**: Install TensorFlow GPU for faster embedding generation

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- OpenCV, MTCNN, FaceNet, scikit-learn, Flask
- See `requirements.txt` for full list

## License

MIT License

## Author

Emilio Dalla Torre
