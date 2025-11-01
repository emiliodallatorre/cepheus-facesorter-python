"""Face sorting and file management module."""

import os
import shutil
from typing import List, Dict
import cv2
import numpy as np
from pathlib import Path

import config
from database import DatabaseManager


class FaceSorter:
    """Handles sorting and copying face images to organized folders."""
    
    def __init__(self, db: DatabaseManager, output_dir: str = None):
        """
        Initialize the face sorter.
        
        Args:
            db: Database manager instance
            output_dir: Base output directory (default from config)
        """
        self.db = db
        self.output_dir = output_dir or config.get_output_dir()
    
    def create_person_folder(self, person: dict) -> str:
        """
        Create a folder for a person.
        
        Args:
            person: Person dictionary with 'name' and 'surname'
            
        Returns:
            Path to the created folder
        """
        folder_name = f"{person['name']}_{person['surname']}"
        folder_path = os.path.join(self.output_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        return folder_path
    
    def extract_and_save_face(self, face: dict, output_path: str):
        """
        Extract a face from the original image and save it.
        
        Args:
            face: Face dictionary with 'original_image_path' and 'face_coordinates'
            output_path: Path where to save the extracted face
        """
        # Read original image
        image = cv2.imread(face['original_image_path'])
        if image is None:
            raise ValueError(f"Could not read image: {face['original_image_path']}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract face region
        x, y, width, height = face['face_coordinates']
        face_image = image_rgb[y:y+height, x:x+width]
        
        # Convert back to BGR for saving
        face_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
        
        # Save
        cv2.imwrite(output_path, face_bgr)
    
    def copy_original_image(self, source_path: str, dest_path: str):
        """
        Copy the original image to the destination.
        
        Args:
            source_path: Source image path
            dest_path: Destination path
        """
        shutil.copy2(source_path, dest_path)
    
    def sort_person_faces(self, person_id: int) -> Dict[str, int]:
        """
        Sort all faces for a person into their folder.
        
        Args:
            person_id: Person ID
            
        Returns:
            Dictionary with statistics (faces_sorted, originals_copied)
        """
        # Get person info
        person = self.db.get_person(person_id)
        if not person:
            raise ValueError(f"Person {person_id} not found")
        
        # Create person folder
        person_folder = self.create_person_folder(person)
        
        # Get all faces for this person
        faces = self.db.get_faces_by_person(person_id)
        
        # Track original images we've already copied
        copied_originals = set()
        faces_sorted = 0
        originals_copied = 0
        
        for face in faces:
            face_id = face['id']
            original_path = face['original_image_path']
            
            # Generate filename for extracted face
            original_filename = Path(original_path).stem
            face_filename = f"face_{face_id}_{original_filename}.jpg"
            face_output_path = os.path.join(person_folder, face_filename)
            
            # Extract and save face
            self.extract_and_save_face(face, face_output_path)
            faces_sorted += 1
            
            # Copy original image if configured and not already copied
            if config.COPY_ORIGINAL_IMAGES and original_path not in copied_originals:
                original_filename_full = Path(original_path).name
                original_output_path = os.path.join(person_folder, f"original_{original_filename_full}")
                self.copy_original_image(original_path, original_output_path)
                copied_originals.add(original_path)
                originals_copied += 1
        
        return {
            'faces_sorted': faces_sorted,
            'originals_copied': originals_copied,
            'folder': person_folder
        }
    
    def sort_all_labeled_faces(self) -> Dict[str, any]:
        """
        Sort all labeled faces into their respective person folders.
        
        Returns:
            Dictionary with overall statistics
        """
        persons = self.db.get_all_persons()
        
        total_faces = 0
        total_originals = 0
        persons_processed = 0
        
        results = {}
        
        for person in persons:
            person_id = person['id']
            person_name = f"{person['name']} {person['surname']}"
            
            try:
                stats = self.sort_person_faces(person_id)
                total_faces += stats['faces_sorted']
                total_originals += stats['originals_copied']
                persons_processed += 1
                
                results[person_name] = stats
            except Exception as e:
                results[person_name] = {'error': str(e)}
        
        return {
            'total_faces_sorted': total_faces,
            'total_originals_copied': total_originals,
            'persons_processed': persons_processed,
            'details': results
        }
    
    def create_contact_sheet(self, face_images: List[np.ndarray], 
                            output_path: str, cols: int = 4):
        """
        Create a contact sheet (grid) of multiple face images.
        
        Args:
            face_images: List of face image arrays
            output_path: Path where to save the contact sheet
            cols: Number of columns in the grid
        """
        if not face_images:
            return
        
        # Resize all faces to same size
        target_size = (200, 200)
        resized_faces = []
        for face in face_images:
            resized = cv2.resize(face, target_size)
            resized_faces.append(resized)
        
        # Calculate grid dimensions
        n_faces = len(resized_faces)
        rows = (n_faces + cols - 1) // cols
        
        # Create blank canvas
        canvas_height = rows * target_size[0]
        canvas_width = cols * target_size[1]
        
        # Determine number of channels
        if len(resized_faces[0].shape) == 3:
            canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
        else:
            canvas = np.ones((canvas_height, canvas_width), dtype=np.uint8) * 255
        
        # Place faces on canvas
        for idx, face in enumerate(resized_faces):
            row = idx // cols
            col = idx % cols
            
            y_start = row * target_size[0]
            y_end = y_start + target_size[0]
            x_start = col * target_size[1]
            x_end = x_start + target_size[1]
            
            canvas[y_start:y_end, x_start:x_end] = face
        
        # Convert RGB to BGR if needed and save
        if len(canvas.shape) == 3 and canvas.shape[2] == 3:
            canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, canvas_bgr)
        else:
            cv2.imwrite(output_path, canvas)
