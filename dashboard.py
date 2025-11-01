"""Web dashboard for managing face clusters."""

from flask import Flask, render_template, jsonify, request, send_file, Response, stream_with_context
from flask_cors import CORS
import os
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
import json
import time
import queue
import threading

import config
from database import DatabaseManager
from face_clusterer import FaceClusterer
from face_sorter import FaceSorter

app = Flask(__name__)
CORS(app)

# Global progress queue for SSE
progress_queues = {}
progress_lock = threading.Lock()

db_manager = None


def get_db():
    """Get database manager instance."""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager(config.DATABASE_PATH)
    return db_manager


def face_image_to_base64(face_data):
    """Convert face image to base64 for display."""
    try:
        image = cv2.imread(face_data['original_image_path'])
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x, y, w, h = face_data['face_coordinates']
        face_img = image_rgb[y:y+h, x:x+w]
        
        # Resize for thumbnail
        face_img = cv2.resize(face_img, (150, 150))
        
        # Convert to base64
        pil_img = Image.fromarray(face_img)
        buffer = BytesIO()
        pil_img.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/jpeg;base64,{img_str}"
    except:
        return None


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')


@app.route('/api/stats')
def get_stats():
    """Get database statistics."""
    db = get_db()
    
    all_faces = db.get_all_faces()
    unlabeled = db.get_unlabeled_faces()
    persons = db.get_all_persons()
    clusters = db.get_unique_clusters()
    
    confirmed_persons = [p for p in persons if p.get('is_confirmed')]
    
    return jsonify({
        'total_faces': int(len(all_faces)),
        'labeled_faces': int(len(all_faces) - len(unlabeled)),
        'unlabeled_faces': int(len(unlabeled)),
        'total_persons': int(len(persons)),
        'confirmed_persons': int(len(confirmed_persons)),
        'pending_persons': int(len(persons) - len(confirmed_persons)),
        'clusters': int(len([c for c in clusters if c != -1])),
        'outliers': int(sum(1 for f in all_faces if f.get('cluster_id') == -1))
    })


@app.route('/api/clusters')
def get_clusters():
    """Get all clusters with preview faces."""
    db = get_db()
    cluster_ids = db.get_unique_clusters()
    
    result = []
    for cluster_id in cluster_ids:
        if cluster_id == -1:
            continue  # Skip outliers for now
        
        faces = db.get_active_faces_by_cluster(cluster_id)
        
        if not faces:
            continue
        
        # Get preview images (first 4 faces)
        preview_faces = []
        for face in faces[:4]:
            img_data = face_image_to_base64(face)
            if img_data:
                preview_faces.append(img_data)
        
        # Check if cluster is labeled
        person_id = faces[0].get('person_id')
        person = None
        if person_id:
            person = db.get_person(person_id)
        
        result.append({
            'cluster_id': int(cluster_id),
            'face_count': int(len(faces)),
            'preview_faces': preview_faces,
            'is_labeled': person is not None,
            'person': person
        })
    
    return jsonify(result)


@app.route('/api/cluster/<int:cluster_id>')
def get_cluster_details(cluster_id):
    """Get detailed information about a cluster."""
    db = get_db()
    faces = db.get_active_faces_by_cluster(cluster_id)
    
    # Calculate distances
    embeddings = np.array([f['face_encoding'] for f in faces])
    
    distances = []
    if len(embeddings) > 1:
        for i, emb in enumerate(embeddings):
            other_distances = []
            for j, other_emb in enumerate(embeddings):
                if i != j:
                    dist = np.linalg.norm(emb - other_emb)
                    other_distances.append(dist)
            avg_dist = np.mean(other_distances)
            distances.append(avg_dist)
    else:
        distances = [0.0]
    
    # Prepare face data with images
    face_data = []
    for idx, face in enumerate(faces):
        img_data = face_image_to_base64(face)
        if img_data:
            face_data.append({
                'id': int(face['id']),
                'image': img_data,
                'distance': float(distances[idx]),
                'confidence': float(face['confidence']),
                'file_path': str(face['original_image_path'])
            })
    
    # Sort by distance descending - faces with highest distance (outliers) appear first
    face_data.sort(key=lambda x: x['distance'], reverse=True)
    
    return jsonify({
        'cluster_id': int(cluster_id),
        'faces': face_data,
        'total_count': int(len(faces))
    })


@app.route('/api/cluster/<int:cluster_id>/remove_face', methods=['POST'])
def remove_face_from_cluster(cluster_id):
    """Mark a face as removed from cluster."""
    data = request.json
    face_id = data.get('face_id')
    
    db = get_db()
    db.mark_face_removed(face_id, removed=True)
    
    return jsonify({'success': True})


@app.route('/api/cluster/<int:cluster_id>/label', methods=['POST'])
def label_cluster(cluster_id):
    """Label a cluster with person information."""
    data = request.json
    name = data.get('name')
    surname = data.get('surname')
    instagram = data.get('instagram')
    
    db = get_db()
    
    # Create or get person
    existing = db.get_person_by_name(name, surname)
    if existing:
        person_id = existing['id']
    else:
        person_id = db.add_person(name, surname, instagram)
    
    # Get active faces in cluster
    faces = db.get_active_faces_by_cluster(cluster_id)
    
    # Calculate refined cluster center
    embeddings = np.array([f['face_encoding'] for f in faces])
    cluster_center = np.mean(embeddings, axis=0)
    
    # Assign faces to person
    for face in faces:
        db.update_face_person(face['id'], person_id)
    
    # Confirm person cluster
    db.confirm_person_cluster(person_id, cluster_center)
    
    return jsonify({'success': True, 'person_id': person_id})


@app.route('/api/persons')
def get_persons():
    """Get all persons."""
    db = get_db()
    persons = db.get_all_persons()
    
    result = []
    for person in persons:
        faces = db.get_faces_by_person(person['id'])
        
        # Get preview image
        preview = None
        if faces:
            preview = face_image_to_base64(faces[0])
        
        result.append({
            'id': int(person['id']),
            'name': str(person['name']),
            'surname': str(person['surname']),
            'instagram': str(person.get('instagram')) if person.get('instagram') else None,
            'face_count': int(len(faces)),
            'is_confirmed': bool(person.get('is_confirmed', False)),
            'preview': preview
        })
    
    return jsonify(result)


@app.route('/api/person/<int:person_id>/faces')
def get_person_faces(person_id):
    """Get all faces for a specific person."""
    db = get_db()
    faces = db.get_faces_by_person(person_id)
    
    result = []
    for face in faces:
        img_data = face_image_to_base64(face)
        if img_data:
            result.append({
                'id': int(face['id']),
                'image': img_data,
                'confidence': float(face['confidence']),
                'file_path': str(face['original_image_path'])
            })
    
    return jsonify(result)


@app.route('/api/unlabeled_faces')


def send_progress_update(session_id, data):
    """Send progress update to specific session."""
    with progress_lock:
        if session_id in progress_queues:
            try:
                progress_queues[session_id].put_nowait(data)
            except:
                pass


@app.route('/api/scan/progress/<session_id>')
def scan_progress(session_id):
    """Server-Sent Events endpoint for scan progress."""
    def generate():
        # Create queue for this session
        with progress_lock:
            progress_queues[session_id] = queue.Queue()
        
        try:
            while True:
                try:
                    # Get progress data from queue (timeout after 30 seconds)
                    data = progress_queues[session_id].get(timeout=30)
                    
                    # Send SSE message
                    yield f"data: {json.dumps(data)}\n\n"
                    
                    # If complete, end stream
                    if data.get('status') == 'complete' or data.get('status') == 'error':
                        break
                        
                except queue.Empty:
                    # Send heartbeat to keep connection alive
                    yield f"data: {json.dumps({'status': 'heartbeat'})}\n\n"
        finally:
            # Clean up queue
            with progress_lock:
                if session_id in progress_queues:
                    del progress_queues[session_id]
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@app.route('/api/scan', methods=['POST'])
def scan_faces():
    """Scan directory for faces."""
    from face_detector import FaceDetector
    from face_recognizer import FaceRecognizer
    import glob
    
    data = request.get_json()
    directory = data.get('directory', './photos')
    force = data.get('force', False)
    session_id = data.get('session_id')  # Get session ID for progress tracking
    
    if not os.path.exists(directory):
        return jsonify({'success': False, 'error': f'Directory not found: {directory}'}), 404
    
    db = get_db()
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    
    # Find all images recursively (case-insensitive)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff',
                       '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.GIF', '*.TIFF']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
    
    stats = {
        'total_images': len(image_paths),
        'processed': 0,
        'faces_found': 0,
        'skipped': 0,
        'errors': 0
    }
    
    # Send initial progress
    if session_id:
        send_progress_update(session_id, {
            'status': 'scanning',
            'progress': 0,
            'total': len(image_paths),
            'current': 0,
            'message': f'Found {len(image_paths)} images to scan...',
            'stats': stats
        })
    
    print(f"\n{'='*60}")
    print(f"🔍 Starting face scan of {len(image_paths)} images")
    print(f"{'='*60}\n")
    
    for idx, img_path in enumerate(image_paths):
        try:
            filename = os.path.basename(img_path)
            
            # Terminal progress (every 10 images or at start/end)
            if idx % 10 == 0 or idx == len(image_paths) - 1:
                progress_percent = int((idx / len(image_paths)) * 100)
                bar_length = 40
                filled = int(bar_length * idx / len(image_paths))
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"\r[{bar}] {progress_percent}% | {idx}/{len(image_paths)} | 👤 {stats['faces_found']} faces | {filename[:30]:<30}", end='', flush=True)
            
            # Send progress update to frontend
            if session_id and idx % 5 == 0:  # Update every 5 images to avoid flooding
                progress_percent = int((idx / len(image_paths)) * 100)
                send_progress_update(session_id, {
                    'status': 'scanning',
                    'progress': progress_percent,
                    'total': len(image_paths),
                    'current': idx,
                    'message': f'Processing: {filename}',
                    'stats': stats
                })
            
            # Check if already processed
            if not force:
                existing_faces = db.get_faces_by_image(img_path)
                if existing_faces:
                    stats['skipped'] += 1
                    continue
            
            # Detect faces
            detections = detector.detect_faces(img_path)
            
            if not detections:
                stats['processed'] += 1
                continue
            
            # Process each face
            for detection in detections:
                box = detection['box']
                confidence = detection['confidence']
                face_image = detection['image']
                
                # Check if face already exists
                if db.face_exists(img_path, box):
                    continue
                
                # Prepare face for encoding
                face_normalized = detector.extract_face_for_encoding(face_image)
                
                # Generate embedding
                encoding = recognizer.get_embedding(face_normalized)
                
                if encoding is not None:
                    db.add_face(img_path, box, encoding, confidence)
                    stats['faces_found'] += 1
            
            stats['processed'] += 1
            
        except Exception as e:
            stats['errors'] += 1
            print(f"\n❌ Error processing {img_path}: {e}")
    
    # Final terminal output
    print(f"\n\n{'='*60}")
    print(f"✅ Scan Complete!")
    print(f"{'='*60}")
    print(f"📊 Total images:  {stats['total_images']}")
    print(f"✔️  Processed:     {stats['processed']}")
    print(f"👤 Faces found:   {stats['faces_found']}")
    print(f"⏭️  Skipped:       {stats['skipped']}")
    print(f"❌ Errors:        {stats['errors']}")
    print(f"{'='*60}\n")
    
    # Send completion
    if session_id:
        send_progress_update(session_id, {
            'status': 'complete',
            'progress': 100,
            'total': len(image_paths),
            'current': len(image_paths),
            'message': f'Scan complete! Found {stats["faces_found"]} faces',
            'stats': stats
        })
    
    return jsonify({'success': True, 'stats': stats})


@app.route('/api/cluster', methods=['POST'])
def cluster_faces():
    """Cluster all faces."""
    data = request.get_json()
    threshold = data.get('threshold', config.CLUSTERING_DISTANCE_THRESHOLD)
    min_size = data.get('min_size', 2)
    force = data.get('force', False)
    
    db = get_db()
    clusterer = FaceClusterer(distance_threshold=threshold)
    
    try:
        # Get all faces
        faces = db.get_all_faces()
        
        if not faces:
            return jsonify({'success': False, 'error': 'No faces found in database. Run scan first.'}), 400
        
        # If force, clear existing cluster assignments and removed flags
        if force:
            conn = db.get_connection()
            cursor = conn.cursor()
            # Reset all cluster IDs to NULL
            cursor.execute("UPDATE faces SET cluster_id = NULL")
            # Clear removed flags
            cursor.execute("UPDATE faces SET is_removed = 0")
            conn.commit()
            
            # Reload faces after clearing
            faces = db.get_all_faces()
        
        # Extract embeddings
        embeddings = np.array([face['face_encoding'] for face in faces])
        
        # Cluster faces
        cluster_labels = clusterer.auto_cluster_faces(embeddings, min_cluster_size=min_size)
        
        # Update database
        for face, cluster_id in zip(faces, cluster_labels):
            db.update_face_cluster(face['id'], int(cluster_id))
        
        # Get statistics
        stats = clusterer.get_cluster_statistics(cluster_labels)
        
        return jsonify({
            'success': True, 
            'stats': {
                'total_faces': int(len(faces)),
                'clusters_formed': int(stats['n_clusters']),
                'outliers': int(stats['n_outliers']),
                'avg_cluster_size': float(stats['avg_cluster_size'])
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/unlabeled_faces')
def get_unlabeled_faces():
    """Get all unlabeled faces (faces without a person_id)."""
    db = get_db()
    unlabeled = db.get_unlabeled_faces()
    
    # Group by original image for better display
    face_data = []
    for face in unlabeled:
        img_data = face_image_to_base64(face)
        if img_data:
            face_data.append({
                'id': int(face['id']),
                'image': img_data,
                'confidence': float(face['confidence']),
                'cluster_id': int(face['cluster_id']) if face.get('cluster_id') is not None else None,
                'file_path': str(face['original_image_path'])
            })
    
    return jsonify(face_data)


@app.route('/api/assign_face', methods=['POST'])
def assign_face_to_person():
    """Manually assign a face to a person."""
    data = request.json
    face_id = data.get('face_id')
    person_id = data.get('person_id')
    
    if not face_id or not person_id:
        return jsonify({'success': False, 'error': 'face_id and person_id required'}), 400
    
    db = get_db()
    
    # Verify person exists
    person = db.get_person(person_id)
    if not person:
        return jsonify({'success': False, 'error': 'Person not found'}), 404
    
    # Assign face to person
    db.update_face_person(face_id, person_id)
    
    return jsonify({'success': True})


@app.route('/api/search_persons')
def search_persons():
    """Search for persons by name."""
    query = request.args.get('q', '').lower().strip()
    
    if not query:
        return jsonify([])
    
    db = get_db()
    all_persons = db.get_all_persons()
    
    # Filter persons by name match
    results = []
    for person in all_persons:
        full_name = f"{person['name']} {person['surname']}".lower()
        if query in full_name:
            results.append({
                'id': int(person['id']),
                'name': str(person['name']),
                'surname': str(person['surname']),
                'full_name': f"{person['name']} {person['surname']}",
                'face_count': len(db.get_faces_by_person(person['id']))
            })
    
    return jsonify(results)


@app.route('/api/export', methods=['POST'])
def export_faces():
    """Export all confirmed persons to folders."""
    import shutil
    from datetime import datetime
    
    data = request.get_json() or {}
    output_dir = data.get('output_dir', './output')
    
    db = get_db()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Export faces to folders
    sorter = FaceSorter(db, output_dir=output_dir)
    results = sorter.sort_all_labeled_faces()
    
    # Copy database to output directory
    db_backup_path = os.path.join(output_dir, f'faces_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db')
    shutil.copy2(config.DATABASE_PATH, db_backup_path)
    
    return jsonify({
        'success': True,
        'results': results,
        'db_backup': db_backup_path,
        'output_dir': output_dir
    })


def run_dashboard(host='127.0.0.1', port=5000, debug=True):
    """Run the Flask dashboard."""
    # Disable reloader to allow code editing without restart
    app.run(host=host, port=port, debug=debug, use_reloader=False)


if __name__ == '__main__':
    run_dashboard()
