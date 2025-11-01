"""Face recognition module using DeepFace."""

import numpy as np
from deepface import DeepFace
from typing import List
import config
import cv2


class FaceRecognizer:
    """Generates face embeddings using DeepFace with Facenet512 model."""
    
    def __init__(self):
        """Initialize the DeepFace model."""
        # Use Facenet512 which gives 512-dimensional embeddings
        self.model_name = "Facenet512"
        # Pre-load the model to avoid loading it every time
        print(f"Loading {self.model_name} model...")
        DeepFace.build_model(self.model_name)
        print(f"{self.model_name} model loaded successfully")
    
    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Generate face embedding for a single face.
        
        Args:
            face_image: Preprocessed face image (RGB, normalized to [-1, 1] or [0, 255])
            
        Returns:
            Face embedding vector (512-dimensional)
        """
        # DeepFace expects BGR and uint8 [0-255]
        # Our face_image is RGB, normalized to [-1, 1]
        
        # Convert from [-1, 1] to [0, 255]
        if face_image.max() <= 1.0:
            face_uint8 = ((face_image + 1.0) * 127.5).astype(np.uint8)
        else:
            face_uint8 = face_image.astype(np.uint8)
        
        # Convert RGB to BGR for DeepFace/OpenCV
        face_bgr = cv2.cvtColor(face_uint8, cv2.COLOR_RGB2BGR)
        
        # Generate embedding using DeepFace
        embedding_objs = DeepFace.represent(
            img_path=face_bgr,
            model_name=self.model_name,
            enforce_detection=False,  # We already detected the face
            detector_backend='skip'    # Skip detection, use the image as-is
        )
        
        # Extract the embedding vector from the result
        embedding = np.array(embedding_objs[0]["embedding"])
        
        return embedding
    
    def get_embeddings_batch(self, face_images: List[np.ndarray]) -> np.ndarray:
        """
        Generate face embeddings for multiple faces.
        
        Args:
            face_images: List of preprocessed face images
            
        Returns:
            Array of face embeddings
        """
        if not face_images:
            return np.array([])
        
        # Process each face individually
        embeddings = []
        for face_image in face_images:
            embedding = self.get_embedding(face_image)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def calculate_distance(self, embedding1: np.ndarray, 
                          embedding2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two face embeddings.
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Euclidean distance (lower = more similar)
        """
        return np.linalg.norm(embedding1 - embedding2)
    
    def are_same_person(self, embedding1: np.ndarray, 
                       embedding2: np.ndarray,
                       threshold: float = None) -> bool:
        """
        Determine if two embeddings belong to the same person.
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            threshold: Distance threshold (default from config)
            
        Returns:
            True if embeddings are similar enough
        """
        if threshold is None:
            threshold = config.SIMILARITY_THRESHOLD
        
        distance = self.calculate_distance(embedding1, embedding2)
        return distance < threshold
    
    def find_most_similar(self, query_embedding: np.ndarray,
                         embeddings: List[np.ndarray],
                         threshold: float = None) -> List[int]:
        """
        Find all embeddings similar to the query embedding.
        
        Args:
            query_embedding: Query face embedding
            embeddings: List of face embeddings to compare against
            threshold: Distance threshold (default from config)
            
        Returns:
            List of indices of similar embeddings
        """
        if threshold is None:
            threshold = config.SIMILARITY_THRESHOLD
        
        similar_indices = []
        for idx, embedding in enumerate(embeddings):
            distance = self.calculate_distance(query_embedding, embedding)
            if distance < threshold:
                similar_indices.append(idx)
        
        return similar_indices
    
    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distance matrix for all embeddings.
        
        Args:
            embeddings: Array of face embeddings (n_faces, embedding_dim)
            
        Returns:
            Distance matrix (n_faces, n_faces)
        """
        n = len(embeddings)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                dist = self.calculate_distance(embeddings[i], embeddings[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
