"""Database manager for storing person and face information."""

import sqlite3
import json
from typing import List, Optional, Tuple
from datetime import datetime
import numpy as np
import threading


class DatabaseManager:
    """Manages SQLite database operations for face data."""
    
    def __init__(self, db_path: str):
        """Initialize database connection and create tables if needed."""
        self.db_path = db_path
        self.local = threading.local()
        self.create_tables()
    
    def get_connection(self):
        """Get or create a database connection for the current thread."""
        if not hasattr(self.local, 'conn') or self.local.conn is None:
            self.local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.local.conn.row_factory = sqlite3.Row
        return self.local.conn
    
    def connect(self):
        """Establish database connection (legacy method)."""
        return self.get_connection()
    
    def create_tables(self):
        """Create database tables if they don't exist."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Persons table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                surname TEXT NOT NULL,
                instagram TEXT,
                is_confirmed BOOLEAN DEFAULT 0,
                cluster_center TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Faces table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                original_image_path TEXT NOT NULL,
                face_coordinates TEXT NOT NULL,
                face_encoding TEXT NOT NULL,
                confidence REAL NOT NULL,
                cluster_id INTEGER,
                is_removed BOOLEAN DEFAULT 0,
                extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES persons (id)
            )
        """)
        
        # Images table (for all images, including those without faces)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL UNIQUE,
                person_id INTEGER,
                has_faces BOOLEAN DEFAULT 0,
                manually_assigned BOOLEAN DEFAULT 0,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES persons (id)
            )
        """)
        
        # Create indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_faces_person_id 
            ON faces(person_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_faces_cluster_id 
            ON faces(cluster_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_images_person_id 
            ON images(person_id)
        """)
        
        conn.commit()
    
    def add_person(self, name: str, surname: str, instagram: Optional[str] = None) -> int:
        """Add a new person to the database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO persons (name, surname, instagram)
            VALUES (?, ?, ?)
        """, (name, surname, instagram))
        conn.commit()
        return cursor.lastrowid
    
    def get_person(self, person_id: int) -> Optional[dict]:
        """Get person information by ID."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM persons WHERE id = ?", (person_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_person_by_name(self, name: str, surname: str) -> Optional[dict]:
        """Get person by name and surname."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM persons WHERE name = ? AND surname = ?
        """, (name, surname))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def update_person(self, person_id: int, name: str = None, 
                     surname: str = None, instagram: str = None):
        """Update person information."""
        updates = []
        params = []
        
        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if surname is not None:
            updates.append("surname = ?")
            params.append(surname)
        if instagram is not None:
            updates.append("instagram = ?")
            params.append(instagram)
        
        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(person_id)
        
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
            UPDATE persons SET {', '.join(updates)}
            WHERE id = ?
        """, params)
        conn.commit()
    
    def add_face(self, original_image_path: str, face_coordinates: Tuple[int, int, int, int],
                 face_encoding: np.ndarray, confidence: float, 
                 person_id: Optional[int] = None, cluster_id: Optional[int] = None) -> int:
        """Add a face detection to the database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Convert coordinates and encoding to JSON
        coords_json = json.dumps(face_coordinates)
        encoding_json = json.dumps(face_encoding.tolist())
        
        cursor.execute("""
            INSERT INTO faces (original_image_path, face_coordinates, face_encoding,
                             confidence, person_id, cluster_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (original_image_path, coords_json, encoding_json, confidence, 
              person_id, cluster_id))
        conn.commit()
        return cursor.lastrowid
    
    def face_exists(self, original_image_path: str, face_coordinates: Tuple[int, int, int, int]) -> bool:
        """
        Check if a face at specific coordinates already exists in the database.
        
        Args:
            original_image_path: Path to the original image
            face_coordinates: Face coordinates (x, y, width, height)
            
        Returns:
            True if face exists, False otherwise
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        coords_json = json.dumps(face_coordinates)
        
        cursor.execute("""
            SELECT COUNT(*) FROM faces 
            WHERE original_image_path = ? AND face_coordinates = ?
        """, (original_image_path, coords_json))
        
        count = cursor.fetchone()[0]
        return count > 0
    
    def get_face(self, face_id: int) -> Optional[dict]:
        """Get face information by ID."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM faces WHERE id = ?", (face_id,))
        row = cursor.fetchone()
        if row:
            face = dict(row)
            face['face_coordinates'] = json.loads(face['face_coordinates'])
            face['face_encoding'] = np.array(json.loads(face['face_encoding']))
            return face
        return None
    
    def get_faces_by_cluster(self, cluster_id: int) -> List[dict]:
        """Get all faces in a cluster."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM faces WHERE cluster_id = ?", (cluster_id,))
        faces = []
        for row in cursor.fetchall():
            face = dict(row)
            face['face_coordinates'] = json.loads(face['face_coordinates'])
            face['face_encoding'] = np.array(json.loads(face['face_encoding']))
            faces.append(face)
        return faces
    
    def get_faces_by_person(self, person_id: int) -> List[dict]:
        """Get all faces for a person."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM faces WHERE person_id = ?", (person_id,))
        faces = []
        for row in cursor.fetchall():
            face = dict(row)
            face['face_coordinates'] = json.loads(face['face_coordinates'])
            face['face_encoding'] = np.array(json.loads(face['face_encoding']))
            faces.append(face)
        return faces
    
    def get_faces_by_image(self, image_path: str) -> List[dict]:
        """Get all faces from a specific image."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM faces WHERE original_image_path = ?", (image_path,))
        faces = []
        for row in cursor.fetchall():
            face = dict(row)
            face['face_coordinates'] = json.loads(face['face_coordinates'])
            face['face_encoding'] = np.array(json.loads(face['face_encoding']))
            faces.append(face)
        return faces
    
    def get_unlabeled_faces(self) -> List[dict]:
        """Get all faces without a person_id."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM faces WHERE person_id IS NULL")
        faces = []
        for row in cursor.fetchall():
            face = dict(row)
            face['face_coordinates'] = json.loads(face['face_coordinates'])
            face['face_encoding'] = np.array(json.loads(face['face_encoding']))
            faces.append(face)
        return faces
    
    def get_all_faces(self) -> List[dict]:
        """Get all faces from the database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM faces")
        faces = []
        for row in cursor.fetchall():
            face = dict(row)
            face['face_coordinates'] = json.loads(face['face_coordinates'])
            face['face_encoding'] = np.array(json.loads(face['face_encoding']))
            faces.append(face)
        return faces
    
    def get_face_by_id(self, face_id: int) -> Optional[dict]:
        """Get a single face by its ID."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM faces WHERE id = ?", (face_id,))
        row = cursor.fetchone()
        if row:
            face = dict(row)
            face['face_coordinates'] = json.loads(face['face_coordinates'])
            face['face_encoding'] = np.array(json.loads(face['face_encoding']))
            return face
        return None
    
    def update_face_person(self, face_id: int, person_id: int):
        """Assign a person to a face."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE faces SET person_id = ? WHERE id = ?
        """, (person_id, face_id))
        conn.commit()
    
    def update_face_cluster(self, face_id: int, cluster_id: int):
        """Assign a cluster to a face."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE faces SET cluster_id = ? WHERE id = ?
        """, (cluster_id, face_id))
        conn.commit()
    
    def get_unique_clusters(self) -> List[int]:
        """Get list of unique cluster IDs."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT cluster_id FROM faces WHERE cluster_id IS NOT NULL")
        return [row[0] for row in cursor.fetchall()]
    
    def get_all_persons(self) -> List[dict]:
        """Get all persons from the database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM persons")
        return [dict(row) for row in cursor.fetchall()]
    
    def mark_face_removed(self, face_id: int, removed: bool = True):
        """Mark a face as removed from its cluster."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE faces SET is_removed = ? WHERE id = ?
        """, (1 if removed else 0, face_id))
        conn.commit()
    
    def get_active_faces_by_cluster(self, cluster_id: int) -> List[dict]:
        """Get non-removed faces in a cluster."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM faces 
            WHERE cluster_id = ? AND is_removed = 0
        """, (cluster_id,))
        faces = []
        for row in cursor.fetchall():
            face = dict(row)
            face['face_coordinates'] = json.loads(face['face_coordinates'])
            face['face_encoding'] = np.array(json.loads(face['face_encoding']))
            faces.append(face)
        return faces
    
    def confirm_person_cluster(self, person_id: int, cluster_center: np.ndarray):
        """Mark a person's cluster as confirmed and save cluster center."""
        conn = self.get_connection()
        cursor = conn.cursor()
        center_json = json.dumps(cluster_center.tolist())
        cursor.execute("""
            UPDATE persons 
            SET is_confirmed = 1, cluster_center = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (center_json, person_id))
        conn.commit()
    
    def add_image(self, file_path: str, person_id: Optional[int] = None, 
                  has_faces: bool = False, manually_assigned: bool = False) -> int:
        """Add an image to the database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO images (file_path, person_id, has_faces, manually_assigned)
                VALUES (?, ?, ?, ?)
            """, (file_path, person_id, 1 if has_faces else 0, 1 if manually_assigned else 0))
            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # Image already exists
            return None
    
    def assign_image_to_person(self, image_path: str, person_id: int):
        """Assign an image to a person."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE images 
            SET person_id = ?, manually_assigned = 1
            WHERE file_path = ?
        """, (person_id, image_path))
        
        if cursor.rowcount == 0:
            # Image doesn't exist, add it
            self.add_image(image_path, person_id, has_faces=False, manually_assigned=True)
        else:
            conn.commit()
    
    def get_images_by_person(self, person_id: int) -> List[dict]:
        """Get all images assigned to a person."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM images WHERE person_id = ?
        """, (person_id,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_unassigned_images(self) -> List[dict]:
        """Get all images not assigned to any person."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM images WHERE person_id IS NULL
        """)
        return [dict(row) for row in cursor.fetchall()]
    
    def close(self):
        """Close database connection."""
        if hasattr(self.local, 'conn') and self.local.conn:
            self.local.conn.close()
            self.local.conn = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
