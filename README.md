# Cepheus Face Sorter

A Python application with CLI interface for detecting, clustering, labeling, and sorting faces from images using TensorFlow models.

## Features

- 🔍 **Face Detection**: Detect faces in images using MTCNN (TensorFlow-based)
- 🧠 **Face Recognition**: Generate face embeddings using FaceNet
- 🔗 **Face Clustering**: Automatically cluster similar faces using DBSCAN
- 🏷️ **Face Labeling**: Interactive GUI for labeling faces with person information
- 📁 **Face Sorting**: Organize faces into folders by person
- 💾 **SQLite Database**: Store all face data, coordinates, embeddings, and person info

## Installation

1. Clone the repository:
```bash
git clone https://github.com/emiliodallatorre/cepheus-facesorter-python.git
cd cepheus-facesorter-python
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Complete Pipeline

Run the entire workflow in one command:

```bash
python main.py pipeline /path/to/images
```

### Step-by-Step Workflow

#### 1. Scan for Faces

Scan a directory for images and detect faces:

```bash
python main.py scan /path/to/images
```

This will:
- Find all supported image formats (.jpg, .jpeg, .png, .bmp, .gif, .tiff)
- Detect faces using MTCNN
- Generate face embeddings using FaceNet
- Store everything in the SQLite database

#### 2. Cluster Faces

Cluster similar faces together:

```bash
python main.py cluster
```

This groups similar faces automatically using DBSCAN clustering.

#### 3. Label Faces

Interactively label faces with person information:

```bash
python main.py label
```

For each cluster:
- View all faces in a grid
- Confirm if they belong to the same person
- Enter name, surname, and Instagram username (optional)

#### 4. Sort Faces

Sort labeled faces into organized folders:

```bash
python main.py sort
```

This creates folders like `output/John_Doe/` containing:
- Extracted face images
- Original images (if enabled in config)

### Other Commands

#### View Statistics

```bash
python main.py stats
```

Shows database statistics including total faces, persons, clusters, etc.

## Configuration

Edit `config.py` to customize:

- **Detection confidence**: Minimum face detection confidence (default: 0.9)
- **Similarity threshold**: Face matching threshold (default: 0.6)
- **Clustering threshold**: Face clustering threshold (default: 0.6)
- **Output directory**: Where to save sorted faces (default: "output")
- **Preview settings**: Image size and grid layout for the viewer

## Project Structure

```
cepheus-facesorter-python/
├── main.py              # CLI interface
├── config.py            # Configuration settings
├── database.py          # SQLite database manager
├── face_detector.py     # Face detection using MTCNN
├── face_recognizer.py   # Face recognition using FaceNet
├── face_clusterer.py    # Face clustering using DBSCAN
├── face_viewer.py       # Tkinter GUI for face viewing
├── face_sorter.py       # Face sorting and file management
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Database Schema

### Persons Table
- id, name, surname, instagram
- created_at, updated_at

### Faces Table
- id, person_id, cluster_id
- original_image_path, face_coordinates
- face_encoding (512-dim FaceNet embedding)
- confidence, extracted_at

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- OpenCV
- MTCNN
- FaceNet (keras-facenet)
- scikit-learn
- Pillow
- Click

## Tips

1. **Adjust Thresholds**: If clustering is too loose/tight, adjust `CLUSTERING_DISTANCE_THRESHOLD` in `config.py`
2. **GPU Acceleration**: Install TensorFlow with GPU support for faster processing
3. **Large Datasets**: Process in batches by running scan on subdirectories
4. **Outliers**: Faces with cluster_id = -1 are outliers (didn't cluster well)

## License

MIT License

## Author

Emilio Dalla Torre
