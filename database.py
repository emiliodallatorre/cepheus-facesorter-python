"""Database manager for storing person and face information."""

import sqlite3
import json
from typing import List, Optional, Tuple
from datetime import datetime
import numpy as np


class DatabaseManager:
    """Manages SQLite database operations for face data."""
    
    def __init__(self, db_path: str):
        """Initialize database connection and create tables if needed."""
        self.db_path = db_path
        self.conn = None
        self.connect()
        self.create_tables()
    
    def connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
    
    def create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Persons table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                surname TEXT NOT NULL,
                instagram TEXT,
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
                extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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
        
        self.conn.commit()
    
    def add_person(self, name: str, surname: str, instagram: Optional[str] = None) -> int:
        """Add a new person to the database."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO persons (name, surname, instagram)
            VALUES (?, ?, ?)
        """, (name, surname, instagram))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_person(self, person_id: int) -> Optional[dict]:
        """Get person information by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM persons WHERE id = ?", (person_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_person_by_name(self, name: str, surname: str) -> Optional[dict]:
        """Get person by name and surname."""
        cursor = self.conn.cursor()
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
        
        cursor = self.conn.cursor()
        cursor.execute(f"""
            UPDATE persons SET {', '.join(updates)}
            WHERE id = ?
        """, params)
        self.conn.commit()
    
    def add_face(self, original_image_path: str, face_coordinates: Tuple[int, int, int, int],
                 face_encoding: np.ndarray, confidence: float, 
                 person_id: Optional[int] = None, cluster_id: Optional[int] = None) -> int:
        """Add a face detection to the database."""
        cursor = self.conn.cursor()
        
        # Convert coordinates and encoding to JSON
        coords_json = json.dumps(face_coordinates)
        encoding_json = json.dumps(face_encoding.tolist())
        
        cursor.execute("""
            INSERT INTO faces (original_image_path, face_coordinates, face_encoding,
                             confidence, person_id, cluster_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (original_image_path, coords_json, encoding_json, confidence, 
              person_id, cluster_id))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_face(self, face_id: int) -> Optional[dict]:
        """Get face information by ID."""
        cursor = self.conn.cursor()
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
        cursor = self.conn.cursor()
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
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM faces WHERE person_id = ?", (person_id,))
        faces = []
        for row in cursor.fetchall():
            face = dict(row)
            face['face_coordinates'] = json.loads(face['face_coordinates'])
            face['face_encoding'] = np.array(json.loads(face['face_encoding']))
            faces.append(face)
        return faces
    
    def get_unlabeled_faces(self) -> List[dict]:
        """Get all faces without a person_id."""
        cursor = self.conn.cursor()
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
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM faces")
        faces = []
        for row in cursor.fetchall():
            face = dict(row)
            face['face_coordinates'] = json.loads(face['face_coordinates'])
            face['face_encoding'] = np.array(json.loads(face['face_encoding']))
            faces.append(face)
        return faces
    
    def update_face_person(self, face_id: int, person_id: int):
        """Assign a person to a face."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE faces SET person_id = ? WHERE id = ?
        """, (person_id, face_id))
        self.conn.commit()
    
    def update_face_cluster(self, face_id: int, cluster_id: int):
        """Assign a cluster to a face."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE faces SET cluster_id = ? WHERE id = ?
        """, (cluster_id, face_id))
        self.conn.commit()
    
    def get_unique_clusters(self) -> List[int]:
        """Get list of unique cluster IDs."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT cluster_id FROM faces WHERE cluster_id IS NOT NULL")
        return [row[0] for row in cursor.fetchall()]
    
    def get_all_persons(self) -> List[dict]:
        """Get all persons from the database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM persons")
        return [dict(row) for row in cursor.fetchall()]
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
