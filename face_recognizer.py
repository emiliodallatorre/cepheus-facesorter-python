"""Face recognition module using FaceNet (TensorFlow-based)."""

import numpy as np
from keras_facenet import FaceNet
from typing import List
import config


class FaceRecognizer:
    """Generates face embeddings using FaceNet model."""
    
    def __init__(self):
        """Initialize the FaceNet model."""
        self.model = FaceNet()
    
    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Generate face embedding for a single face.
        
        Args:
            face_image: Preprocessed face image (160x160, normalized)
            
        Returns:
            Face embedding vector (512-dimensional)
        """
        # Ensure the image has the right shape
        if len(face_image.shape) == 3:
            face_image = np.expand_dims(face_image, axis=0)
        
        # Generate embedding
        embedding = self.model.embeddings(face_image)
        
        return embedding[0]
    
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
        
        # Stack images into batch
        batch = np.array(face_images)
        
        # Generate embeddings
        embeddings = self.model.embeddings(batch)
        
        return embeddings
    
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
