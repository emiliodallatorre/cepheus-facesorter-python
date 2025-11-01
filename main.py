"""Simplified CLI interface for the face sorter application - Dashboard-focused."""

import click
import os
from pathlib import Path
import numpy as np

import config
from database import DatabaseManager
from face_detector import FaceDetector, scan_directory_for_images
from face_recognizer import FaceRecognizer
from face_clusterer import FaceClusterer


@click.group()
def cli():
    """Face Sorter - Detect, cluster, and manage faces via web dashboard."""
    pass


@cli.command()
@click.option('--db', default=config.DATABASE_PATH, help='Database file path')
@click.option('--force', is_flag=True, help='Force re-scan even if images are already processed')
@click.argument('input_dir', type=click.Path(exists=True))
def scan(input_dir, db, force):
    """
    Scan a directory for images and detect faces.
    
    INPUT_DIR: Directory containing images to scan
    """
    click.echo(f"Scanning directory: {input_dir}")
    
    # Initialize components
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    db_manager = DatabaseManager(db)
    
    # Find images
    image_paths = scan_directory_for_images(input_dir)
    click.echo(f"Found {len(image_paths)} images")
    
    if not image_paths:
        click.echo("ERROR: No images found!")
        return
    
    # Check which images are already in database
    images_to_process = []
    if not force:
        for img_path in image_paths:
            existing = db_manager.get_faces_by_image(img_path)
            if not existing:
                images_to_process.append(img_path)
        click.echo(f"{len(image_paths) - len(images_to_process)} images already processed")
        if images_to_process:
            click.echo(f"{len(images_to_process)} new images to process")
    else:
        images_to_process = image_paths
        click.echo("Force mode: processing all images")
    
    if not images_to_process:
        click.echo("No new images to process!")
        db_manager.close()
        return
    
    # Process images
    total_faces = 0
    images_with_faces = 0
    images_without_faces = 0
    duplicate_faces = 0
    
    with click.progressbar(images_to_process, label='Processing images', 
                          show_eta=True, show_percent=True) as images:
        for image_path in images:
            try:
                # Detect faces
                faces = detector.detect_faces(image_path)
                
                if faces:
                    images_with_faces += 1
                    click.echo(f"\n  [OK] {Path(image_path).name}: {len(faces)} face(s) detected")
                else:
                    images_without_faces += 1
                
                for face in faces:
                    # Check if this face already exists
                    if db_manager.face_exists(image_path, face['box']):
                        click.echo(f"       Skipping duplicate face at {face['box']}")
                        duplicate_faces += 1
                        continue
                    
                    # Prepare face for encoding
                    face_normalized = detector.extract_face_for_encoding(face['image'])
                    
                    # Generate embedding
                    click.echo(f"       Generating embedding... ", nl=False)
                    embedding = recognizer.get_embedding(face_normalized)
                    click.echo("done")
                    
                    # Save to database
                    db_manager.add_face(
                        original_image_path=image_path,
                        face_coordinates=face['box'],
                        face_encoding=embedding,
                        confidence=face['confidence']
                    )
                    
                    total_faces += 1
                    
            except Exception as e:
                click.echo(f"\nWARNING: Error processing {Path(image_path).name}: {e}", err=True)
    
    db_manager.close()
    
    # Summary
    click.echo(f"\n{'='*50}")
    click.echo(f"Scan complete!")
    click.echo(f"  Images processed: {len(images_to_process)}")
    click.echo(f"  Images with faces: {images_with_faces}")
    click.echo(f"  Images without faces: {images_without_faces}")
    click.echo(f"  Total faces detected: {total_faces}")
    if duplicate_faces > 0:
        click.echo(f"  Duplicate faces skipped: {duplicate_faces}")
    click.echo(f"{'='*50}")


@cli.command()
@click.option('--db', default=config.DATABASE_PATH, help='Database file path')
@click.option('--threshold', default=config.CLUSTERING_DISTANCE_THRESHOLD, 
              help='Clustering distance threshold')
@click.option('--min-size', default=2, help='Minimum cluster size (faces with fewer are outliers)')
def cluster(db, threshold, min_size):
    """
    Cluster detected faces by similarity using hierarchical clustering.
    """
    click.echo("Clustering faces by similarity...")
    
    # Initialize components
    db_manager = DatabaseManager(db)
    clusterer = FaceClusterer(distance_threshold=threshold)
    
    # Get all faces
    faces = db_manager.get_all_faces()
    
    if not faces:
        click.echo("ERROR: No faces found in database. Run 'scan' first.")
        db_manager.close()
        return
    
    click.echo(f"Found {len(faces)} faces")
    
    # Extract embeddings
    click.echo("Extracting face embeddings...")
    embeddings = np.array([face['face_encoding'] for face in faces])
    click.echo(f"  Extracted {len(embeddings)} embeddings")
    
    # Cluster with automatic detection
    click.echo(f"Running hierarchical clustering (threshold={threshold}, min_size={min_size})...")
    cluster_labels = clusterer.auto_cluster_faces(embeddings, min_cluster_size=min_size)
    click.echo(f"  Clustering complete")
    
    # Update database
    click.echo("Updating database with cluster assignments...")
    updated_count = 0
    outlier_count = 0
    with click.progressbar(list(zip(faces, cluster_labels)), 
                          label='Updating faces',
                          show_percent=True) as items:
        for face, cluster_id in items:
            db_manager.update_face_cluster(face['id'], int(cluster_id))
            if cluster_id != -1:
                updated_count += 1
            else:
                outlier_count += 1
    click.echo(f"  Updated {updated_count} clustered faces, {outlier_count} outliers")
    
    # Get statistics
    stats = clusterer.get_cluster_statistics(cluster_labels)
    
    db_manager.close()
    
    click.echo(f"\n{'='*50}")
    click.echo(f"Clustering complete!")
    click.echo(f"  Clusters found: {stats['n_clusters']}")
    click.echo(f"  Outliers (ungrouped): {stats['n_outliers']}")
    click.echo(f"  Average cluster size: {stats['avg_cluster_size']:.1f} faces")
    click.echo(f"  Largest cluster: {max(stats['cluster_sizes'].values()) if stats['cluster_sizes'] else 0} faces")
    click.echo(f"  Smallest cluster: {min(stats['cluster_sizes'].values()) if stats['cluster_sizes'] else 0} faces")
    click.echo(f"{'='*50}")


@cli.command()
@click.option('--db', default=config.DATABASE_PATH, help='Database file path')
def clean(db):
    """
    Remove duplicate faces from the database.
    """
    click.echo("Checking for duplicate faces...")
    
    db_manager = DatabaseManager(db)
    all_faces = db_manager.get_all_faces()
    
    # Track seen faces by image path + coordinates
    seen = set()
    duplicates = []
    
    for face in all_faces:
        key = (face['original_image_path'], tuple(face['face_coordinates']))
        if key in seen:
            duplicates.append(face['id'])
        else:
            seen.add(key)
    
    if not duplicates:
        click.echo("No duplicates found!")
        db_manager.close()
        return
    
    click.echo(f"Found {len(duplicates)} duplicate faces")
    
    if click.confirm("Remove duplicates?"):
        conn = db_manager.get_connection()
        cursor = conn.cursor()
        for face_id in duplicates:
            cursor.execute("DELETE FROM faces WHERE id = ?", (face_id,))
        conn.commit()
        click.echo(f"[OK] Removed {len(duplicates)} duplicate faces")
    else:
        click.echo("Cancelled")
    
    db_manager.close()


@cli.command()
@click.option('--db', default=config.DATABASE_PATH, help='Database file path')
def stats(db):
    """
    Show database statistics.
    """
    db_manager = DatabaseManager(db)
    
    # Get statistics
    persons = db_manager.get_all_persons()
    all_faces = db_manager.get_all_faces()
    unlabeled_faces = db_manager.get_unlabeled_faces()
    clusters = db_manager.get_unique_clusters()
    
    click.echo("\nDatabase Statistics")
    click.echo("=" * 40)
    click.echo(f"Total persons: {len(persons)}")
    click.echo(f"Total faces: {len(all_faces)}")
    click.echo(f"Labeled faces: {len(all_faces) - len(unlabeled_faces)}")
    click.echo(f"Unlabeled faces: {len(unlabeled_faces)}")
    click.echo(f"Clusters: {len(clusters)}")
    
    if persons:
        click.echo("\nPersons:")
        for person in persons:
            instagram = f" (@{person['instagram']})" if person['instagram'] else ""
            face_count = len(db_manager.get_faces_by_person(person['id']))
            click.echo(f"  - {person['name']} {person['surname']}{instagram}: {face_count} faces")
    
    db_manager.close()


@cli.command()
@click.option('--host', default='127.0.0.1', help='Host to run dashboard on')
@click.option('--port', default=5000, help='Port to run dashboard on')
@click.option('--debug/--no-debug', default=True, help='Run in debug mode')
def dashboard(host, port, debug):
    """
    Launch the web dashboard for managing clusters.
    
    Open http://127.0.0.1:5000 in your browser to:
    - Scan for faces
    - Cluster similar faces
    - Review and refine clusters
    - Label persons
    - Export to folders
    """
    from dashboard import run_dashboard
    
    click.echo(f"Starting dashboard on http://{host}:{port}")
    click.echo("Press Ctrl+C to stop\n")
    click.echo("Dashboard Features:")
    click.echo("  - Scan photos for faces")
    click.echo("  - Cluster similar faces automatically")
    click.echo("  - Review clusters and remove incorrect matches")
    click.echo("  - Label clusters with person information")
    click.echo("  - Export organized photos to folders")
    click.echo()
    
    run_dashboard(host=host, port=port, debug=debug)


if __name__ == '__main__':
    cli()
