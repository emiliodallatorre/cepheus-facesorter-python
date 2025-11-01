"""Face clustering module using hierarchical and DBSCAN clustering."""

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from typing import List, Dict, Optional
import config


class FaceClusterer:
    """Clusters similar faces together using multiple algorithms."""
    
    def __init__(self, distance_threshold: float = None):
        """
        Initialize the face clusterer.
        
        Args:
            distance_threshold: Maximum distance between faces in the same cluster
        """
        self.distance_threshold = distance_threshold or config.CLUSTERING_DISTANCE_THRESHOLD
    
    def auto_cluster_faces(self, embeddings: np.ndarray, min_cluster_size: int = 2) -> np.ndarray:
        """
        Automatically cluster faces trying to find optimal number of clusters.
        Uses hierarchical clustering with automatic cutting.
        
        Args:
            embeddings: Array of face embeddings (n_faces, embedding_dim)
            min_cluster_size: Minimum number of faces to form a cluster
            
        Returns:
            Array of cluster labels (n_faces,)
            Label -1 indicates outliers (single faces or poorly matched)
        """
        if len(embeddings) == 0:
            return np.array([])
        
        if len(embeddings) == 1:
            return np.array([-1])  # Single face is outlier
        
        # Use hierarchical clustering with distance threshold
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.distance_threshold,
            metric='euclidean',
            linkage='average'
        )
        
        labels = clusterer.fit_predict(embeddings)
        
        # Filter out small clusters (less than min_cluster_size)
        # These become outliers
        unique_labels, counts = np.unique(labels, return_counts=True)
        small_clusters = unique_labels[counts < min_cluster_size]
        
        # Convert small clusters to outliers
        final_labels = labels.copy()
        for small_label in small_clusters:
            final_labels[labels == small_label] = -1
        
        # Renumber remaining clusters to be consecutive
        valid_labels = [l for l in np.unique(final_labels) if l != -1]
        label_mapping = {old: new for new, old in enumerate(sorted(valid_labels))}
        label_mapping[-1] = -1
        
        final_labels = np.array([label_mapping[l] for l in final_labels])
        
        return final_labels
    
    def cluster_faces(self, embeddings: np.ndarray, min_samples: int = 2) -> np.ndarray:
        """
        Cluster face embeddings using DBSCAN.
        
        Args:
            embeddings: Array of face embeddings (n_faces, embedding_dim)
            
        Returns:
            Array of cluster labels (n_faces,)
            Label -1 indicates outliers (faces that don't cluster well)
        """
        if len(embeddings) == 0:
            return np.array([])
        
        # Use DBSCAN for clustering
        # metric='euclidean' for Euclidean distance
        # eps is the maximum distance between two samples for them to be in the same cluster
        # min_samples controls minimum cluster size
        clusterer = DBSCAN(
            eps=self.distance_threshold,
            min_samples=min_samples,
            metric='euclidean'
        )
        
        cluster_labels = clusterer.fit_predict(embeddings)
        
        return cluster_labels
    
    def refine_cluster(self, embeddings: np.ndarray, removed_indices: List[int]) -> Dict:
        """
        Refine cluster definition by removing outlier faces.
        Recalculates cluster center and statistics.
        
        Args:
            embeddings: Original cluster embeddings
            removed_indices: Indices of faces to remove
            
        Returns:
            Dictionary with refined cluster info
        """
        # Keep only selected faces
        kept_indices = [i for i in range(len(embeddings)) if i not in removed_indices]
        refined_embeddings = embeddings[kept_indices]
        
        if len(refined_embeddings) == 0:
            return {
                'center': None,
                'mean_distance': None,
                'max_distance': None,
                'count': 0
            }
        
        # Calculate new cluster center
        center = np.mean(refined_embeddings, axis=0)
        
        # Calculate distances from center
        distances = [np.linalg.norm(emb - center) for emb in refined_embeddings]
        
        return {
            'center': center,
            'mean_distance': np.mean(distances),
            'max_distance': np.max(distances) if distances else 0,
            'std_distance': np.std(distances) if distances else 0,
            'count': len(refined_embeddings),
            'kept_indices': kept_indices
        }
    
    def get_cluster_statistics(self, cluster_labels: np.ndarray) -> Dict:
        """
        Get statistics about the clustering results.
        
        Args:
            cluster_labels: Array of cluster labels
            
        Returns:
            Dictionary with statistics
        """
        unique_labels = set(cluster_labels)
        
        # Count outliers
        n_outliers = np.sum(cluster_labels == -1)
        
        # Count clusters (excluding outliers)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        # Calculate cluster sizes
        cluster_sizes = {}
        for label in unique_labels:
            if label != -1:
                cluster_sizes[label] = np.sum(cluster_labels == label)
        
        return {
            'n_clusters': n_clusters,
            'n_outliers': n_outliers,
            'n_total': len(cluster_labels),
            'cluster_sizes': cluster_sizes,
            'avg_cluster_size': np.mean(list(cluster_sizes.values())) if cluster_sizes else 0
        }
    
    def get_faces_in_cluster(self, cluster_labels: np.ndarray, 
                            cluster_id: int) -> np.ndarray:
        """
        Get indices of faces belonging to a specific cluster.
        
        Args:
            cluster_labels: Array of cluster labels
            cluster_id: ID of the cluster
            
        Returns:
            Array of face indices in the cluster
        """
        return np.where(cluster_labels == cluster_id)[0]
    
    def merge_clusters(self, cluster_labels: np.ndarray, 
                      cluster_id1: int, cluster_id2: int) -> np.ndarray:
        """
        Merge two clusters into one.
        
        Args:
            cluster_labels: Array of cluster labels
            cluster_id1: First cluster ID (this ID will be kept)
            cluster_id2: Second cluster ID (will be merged into first)
            
        Returns:
            Updated cluster labels
        """
        new_labels = cluster_labels.copy()
        new_labels[cluster_labels == cluster_id2] = cluster_id1
        return new_labels
    
    def split_cluster(self, embeddings: np.ndarray, cluster_labels: np.ndarray,
                     cluster_id: int) -> np.ndarray:
        """
        Re-cluster a specific cluster with stricter parameters.
        
        Args:
            embeddings: Array of face embeddings
            cluster_labels: Current cluster labels
            cluster_id: Cluster to split
            
        Returns:
            Updated cluster labels
        """
        # Get faces in this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_embeddings = embeddings[cluster_mask]
        
        # Re-cluster with stricter threshold
        stricter_threshold = self.distance_threshold * 0.7
        sub_clusterer = DBSCAN(
            eps=stricter_threshold,
            min_samples=1,
            metric='euclidean'
        )
        
        sub_labels = sub_clusterer.fit_predict(cluster_embeddings)
        
        # Update labels
        new_labels = cluster_labels.copy()
        max_label = np.max(cluster_labels)
        
        # Map sub-cluster labels to new global labels
        unique_sub_labels = set(sub_labels)
        label_mapping = {}
        for i, sub_label in enumerate(sorted(unique_sub_labels)):
            if sub_label == -1:
                label_mapping[sub_label] = -1
            else:
                label_mapping[sub_label] = max_label + i + 1
        
        # Apply new labels
        cluster_indices = np.where(cluster_mask)[0]
        for idx, sub_label in zip(cluster_indices, sub_labels):
            new_labels[idx] = label_mapping[sub_label]
        
        return new_labels
