# Face Clustering Guide

## Problem: Always Getting 1 Cluster

If you're getting only a single cluster with multiple different people, the **distance threshold is too high**.

## Where Clustering Happens

**File:** `face_clusterer.py`  
**Method:** `auto_cluster_faces(embeddings, min_cluster_size=2)`  
**Lines:** 23-70

### Algorithm Details

```python
clusterer = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=self.distance_threshold,  # ← This is the key parameter
    metric='euclidean',
    linkage='average'
)
```

- **Algorithm:** Hierarchical Agglomerative Clustering
- **Linkage:** Average (UPGMA)
- **Metric:** Euclidean distance
- **Current default threshold:** 0.5

## How Distance Threshold Works

- **Lower threshold (0.2-0.3):** Stricter - creates more, smaller clusters
- **Higher threshold (0.6-0.8):** Looser - creates fewer, larger clusters

### For 3 Similar-Looking People

With FaceNet embeddings (512 dimensions), typical distances:
- **Same person:** 0.1 - 0.4
- **Different people (similar):** 0.4 - 0.8
- **Very different people:** 0.8 - 1.2+

**Recommended for your case:** Try threshold = **0.3** or **0.25**

## Testing Different Thresholds

### Method 1: Via Dashboard (Recommended)

1. Launch dashboard: `python main.py dashboard`
2. Open http://127.0.0.1:5000
3. In the "Processing" section, adjust:
   - **Distance Threshold:** Try 0.3, 0.25, 0.2
   - **Min Cluster Size:** Keep at 2
4. Click "Cluster Faces"
5. Check results and repeat with different values

### Method 2: Via config.py

Edit `config.py`:
```python
CLUSTERING_DISTANCE_THRESHOLD = 0.3  # Change from 0.5 to 0.3
```

Then run:
```bash
python main.py cluster --threshold 0.3
```

### Method 3: Direct CLI

```bash
python main.py cluster --threshold 0.3 --min-size 2
```

## Manual Clustering Test

If you want to test clustering yourself with Python:

```python
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from database import DatabaseManager
import config

# Load embeddings
db = DatabaseManager(config.DATABASE_PATH)
faces = db.get_all_faces()
embeddings = np.array([face['face_encoding'] for face in faces])

# Test different thresholds
for threshold in [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric='euclidean',
        linkage='average'
    )
    
    labels = clusterer.fit_predict(embeddings)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    print(f"Threshold {threshold}: {n_clusters} clusters")
    print(f"  Cluster distribution: {np.bincount(labels[labels >= 0])}")
    print()

db.close()
```

## Debugging Tips

### 1. Check Face Distances

```python
from scipy.spatial.distance import pdist, squareform

# Calculate pairwise distances
distances = squareform(pdist(embeddings, metric='euclidean'))

print("Distance statistics:")
print(f"  Min: {distances[distances > 0].min():.3f}")
print(f"  Max: {distances.max():.3f}")
print(f"  Mean: {distances[distances > 0].mean():.3f}")
print(f"  Median: {np.median(distances[distances > 0]):.3f}")
```

### 2. Visualize Dendrogram

```python
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

Z = linkage(embeddings, method='average', metric='euclidean')

plt.figure(figsize=(12, 6))
dendrogram(Z)
plt.title('Face Clustering Dendrogram')
plt.xlabel('Face Index')
plt.ylabel('Distance')
plt.axhline(y=0.3, color='r', linestyle='--', label='threshold=0.3')
plt.axhline(y=0.5, color='b', linestyle='--', label='threshold=0.5')
plt.legend()
plt.show()
```

This will show you exactly where the cuts should be made.

## Common Issues

### Getting Only 1 Cluster
- **Cause:** Threshold too high (0.5+)
- **Solution:** Lower to 0.2-0.3

### Getting Too Many Clusters (each face separate)
- **Cause:** Threshold too low (<0.15)
- **Solution:** Raise to 0.25-0.35

### All Faces as Outliers
- **Cause:** Threshold extremely low or min_cluster_size too high
- **Solution:** Increase threshold or decrease min_cluster_size to 1

## Optimal Workflow

1. **Scan** photos
2. **Cluster** with threshold=0.3 first
3. Check results in dashboard
4. If still 1 cluster → try 0.25 or 0.2
5. If too many clusters → try 0.35 or 0.4
6. **Label** clusters when satisfied
7. **Export** to folders
