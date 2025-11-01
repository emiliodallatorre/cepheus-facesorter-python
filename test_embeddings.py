"""Test if face embeddings are being generated correctly."""

import cv2
import numpy as np
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer

# Initialize
detector = FaceDetector()
recognizer = FaceRecognizer()

# Test with a few images
test_images = [
    "/Users/emiliodallatorre/Desktop/sorter/025294A9-7E5D-4339-B733-064344C7A57A.jpeg",
    "/Users/emiliodallatorre/Desktop/sorter/92083DCD-E9DD-4B35-974D-41A9D3445D21.jpeg",
    "/Users/emiliodallatorre/Desktop/sorter/D7E04EAE-684C-4275-A94A-E4A0FDDFB636.jpeg"
]

print("Testing face embedding generation...\n")

embeddings = []
for img_path in test_images:
    print(f"Processing: {img_path.split('/')[-1]}")
    
    # Detect faces
    faces = detector.detect_faces(img_path)
    print(f"  Detected {len(faces)} face(s)")
    
    if faces:
        face = faces[0]
        
        # Show face image stats
        print(f"  Face image shape: {face['image'].shape}")
        print(f"  Face image min/max: {face['image'].min()}/{face['image'].max()}")
        print(f"  Face image mean: {face['image'].mean():.2f}")
        
        # Preprocess
        face_normalized = detector.extract_face_for_encoding(face['image'])
        print(f"  Normalized shape: {face_normalized.shape}")
        print(f"  Normalized min/max: {face_normalized.min():.3f}/{face_normalized.max():.3f}")
        print(f"  Normalized mean: {face_normalized.mean():.3f}")
        
        # Generate embedding
        embedding = recognizer.get_embedding(face_normalized)
        print(f"  Embedding shape: {embedding.shape}")
        print(f"  Embedding min/max: {embedding.min():.3f}/{embedding.max():.3f}")
        print(f"  Embedding mean: {embedding.mean():.3f}")
        print(f"  Embedding norm: {np.linalg.norm(embedding):.3f}")
        print(f"  First 10 values: {embedding[:10]}")
        
        embeddings.append(embedding)
    print()

# Compare embeddings
if len(embeddings) >= 2:
    print("=== EMBEDDING COMPARISONS ===")
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            print(f"Distance between face {i} and face {j}: {dist:.3f}")
            
            # Check if they're identical
            if np.array_equal(embeddings[i], embeddings[j]):
                print(f"  WARNING: Embeddings {i} and {j} are IDENTICAL!")
