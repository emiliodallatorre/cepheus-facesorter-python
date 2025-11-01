"""Face detection module using MTCNN (TensorFlow-based)."""

import os
import cv2
import numpy as np
from mtcnn import MTCNN
from typing import List, Tuple, Optional
from PIL import Image

import config


class FaceDetector:
    """Detects faces in images using MTCNN."""
    
    def __init__(self):
        """Initialize the MTCNN face detector."""
        # The MTCNN implementation being used does not accept a min_face_size
        # constructor argument, so instantiate the detector with defaults and
        # keep the configured MIN_FACE_SIZE separately for filtering detections.
        self.detector = MTCNN()
        self.min_face_size = getattr(config, "MIN_FACE_SIZE", 20)
    
    def detect_faces(self, image_path: str) -> List[dict]:
        """
        Detect faces in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of dictionaries containing face information:
            - box: (x, y, width, height)
            - confidence: detection confidence
            - keypoints: facial landmarks
            - image: cropped face image
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert BGR to RGB (MTCNN expects RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        detections = self.detector.detect_faces(rgb_image)
        
        # Filter by confidence and extract face images
        faces = []
        for detection in detections:
            confidence = detection['confidence']
            if confidence >= config.FACE_DETECTION_CONFIDENCE:
                box = detection['box']
                x, y, width, height = box
                
                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                width = min(width, rgb_image.shape[1] - x)
                height = min(height, rgb_image.shape[0] - y)
                
                # Calculate face area (used for size-based filtering)
                area = width * height
                
                # Extract face region
                face_image = rgb_image[y:y+height, x:x+width]
                
                faces.append({
                    'box': (x, y, width, height),
                    'confidence': confidence,
                    'keypoints': detection['keypoints'],
                    'image': face_image,
                    'area': area
                })
        
        # Filter faces by size: keep max 2 faces, second must be >= 75% size of largest
        if len(faces) > 1:
            # Sort by area (largest first)
            faces.sort(key=lambda f: f['area'], reverse=True)
            
            # Keep largest face
            filtered_faces = [faces[0]]
            
            # Check if second largest is at least 75% the size of the largest
            if len(faces) > 1:
                largest_area = faces[0]['area']
                second_area = faces[1]['area']
                
                if second_area >= 0.75 * largest_area:
                    filtered_faces.append(faces[1])
            
            faces = filtered_faces
        
        # Remove the 'area' field before returning (not needed by caller)
        for face in faces:
            face.pop('area', None)
        
        return faces
    
    def extract_face_for_encoding(self, face_image: np.ndarray, 
                                  target_size: Tuple[int, int] = (160, 160)) -> np.ndarray:
        """
        Prepare face image for encoding.
        
        Args:
            face_image: Face image array (RGB)
            target_size: Target size for the face (for FaceNet)
            
        Returns:
            Preprocessed face image
        """
        # Resize to target size
        face_resized = cv2.resize(face_image, target_size)
        
        # Normalize pixel values to [-1, 1] (for FaceNet)
        face_normalized = (face_resized - 127.5) / 127.5
        
        return face_normalized
    
    def save_face_image(self, face_image: np.ndarray, output_path: str):
        """
        Save a face image to disk.
        
        Args:
            face_image: Face image array (RGB)
            output_path: Path where to save the image
        """
        # Convert RGB to BGR for OpenCV
        face_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, face_bgr)
    
    def draw_faces_on_image(self, image_path: str, faces: List[dict], 
                           output_path: str):
        """
        Draw bounding boxes on detected faces and save the result.
        
        Args:
            image_path: Path to the original image
            faces: List of face detections
            output_path: Path where to save the annotated image
        """
        image = cv2.imread(image_path)
        
        for face in faces:
            x, y, width, height = face['box']
            confidence = face['confidence']
            
            # Draw rectangle
            cv2.rectangle(image, (x, y), (x+width, y+height), (0, 255, 0), 2)
            
            # Draw confidence
            label = f"{confidence:.2f}"
            cv2.putText(image, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw keypoints
            for key, point in face['keypoints'].items():
                cv2.circle(image, point, 2, (0, 0, 255), 2)
        
        cv2.imwrite(output_path, image)


def scan_directory_for_images(directory: str) -> List[str]:
    """
    Scan a directory for image files.
    
    Args:
        directory: Path to the directory to scan
        
    Returns:
        List of image file paths
    """
    image_paths = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in config.SUPPORTED_FORMATS:
                image_paths.append(os.path.join(root, file))
    
    return image_paths
