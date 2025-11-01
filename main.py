"""Main CLI interface for the face sorter application."""

import click
import os
import sys
from pathlib import Path
import numpy as np

import config
from database import DatabaseManager
from face_detector import FaceDetector, scan_directory_for_images
from face_recognizer import FaceRecognizer
from face_clusterer import FaceClusterer
from face_viewer import FaceViewer
from face_sorter import FaceSorter


@click.group()
def cli():
    """Face Sorter - Detect, cluster, label, and sort faces from images."""
    pass


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--db', default=config.DATABASE_PATH, help='Database file path')
def scan(input_dir, db):
    """
    Scan a directory for images and detect faces.
    
    INPUT_DIR: Directory containing images to scan
    """
    click.echo(f"🔍 Scanning directory: {input_dir}")
    
    # Initialize components
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    db_manager = DatabaseManager(db)
    
    # Find images
    image_paths = scan_directory_for_images(input_dir)
    click.echo(f"📸 Found {len(image_paths)} images")
    
    if not image_paths:
        click.echo("❌ No images found!")
        return
    
    # Process images
    total_faces = 0
    
    with click.progressbar(image_paths, label='Processing images') as images:
        for image_path in images:
            try:
                # Detect faces
                faces = detector.detect_faces(image_path)
                
                for face in faces:
                    # Prepare face for encoding
                    face_normalized = detector.extract_face_for_encoding(face['image'])
                    
                    # Generate embedding
                    embedding = recognizer.get_embedding(face_normalized)
                    
                    # Save to database
                    db_manager.add_face(
                        original_image_path=image_path,
                        face_coordinates=face['box'],
                        face_encoding=embedding,
                        confidence=face['confidence']
                    )
                    
                    total_faces += 1
                    
            except Exception as e:
                click.echo(f"\n⚠️  Error processing {image_path}: {e}", err=True)
    
    db_manager.close()
    
    click.echo(f"\n✅ Scan complete! Detected {total_faces} faces")
    click.echo(f"💾 Data saved to: {db}")


@cli.command()
@click.option('--db', default=config.DATABASE_PATH, help='Database file path')
@click.option('--threshold', default=config.CLUSTERING_DISTANCE_THRESHOLD, 
              help='Clustering distance threshold')
def cluster(db, threshold):
    """
    Cluster detected faces by similarity.
    """
    click.echo("🔗 Clustering faces by similarity...")
    
    # Initialize components
    db_manager = DatabaseManager(db)
    clusterer = FaceClusterer(distance_threshold=threshold)
    
    # Get all faces
    faces = db_manager.get_all_faces()
    
    if not faces:
        click.echo("❌ No faces found in database. Run 'scan' first.")
        db_manager.close()
        return
    
    click.echo(f"📊 Found {len(faces)} faces")
    
    # Extract embeddings
    embeddings = np.array([face['face_encoding'] for face in faces])
    
    # Cluster
    cluster_labels = clusterer.cluster_faces(embeddings)
    
    # Update database
    for face, cluster_id in zip(faces, cluster_labels):
        if cluster_id != -1:  # Skip outliers
            db_manager.update_face_cluster(face['id'], int(cluster_id))
    
    # Get statistics
    stats = clusterer.get_cluster_statistics(cluster_labels)
    
    db_manager.close()
    
    click.echo(f"\n✅ Clustering complete!")
    click.echo(f"   📦 Clusters: {stats['n_clusters']}")
    click.echo(f"   🔍 Outliers: {stats['n_outliers']}")
    click.echo(f"   📈 Avg cluster size: {stats['avg_cluster_size']:.1f}")


@cli.command()
@click.option('--db', default=config.DATABASE_PATH, help='Database file path')
def label(db):
    """
    Label faces by cluster - assign names to detected faces.
    """
    click.echo("🏷️  Starting face labeling process...")
    
    # Initialize components
    db_manager = DatabaseManager(db)
    viewer = FaceViewer("Face Labeling")
    
    # Get unique clusters
    cluster_ids = db_manager.get_unique_clusters()
    
    if not cluster_ids:
        click.echo("❌ No clusters found. Run 'cluster' first.")
        db_manager.close()
        return
    
    click.echo(f"📦 Found {len(cluster_ids)} clusters to label")
    
    try:
        with viewer:
            for cluster_id in cluster_ids:
                # Get faces in this cluster
                faces = db_manager.get_faces_by_cluster(cluster_id)
                
                if not faces:
                    continue
                
                click.echo(f"\n👤 Cluster {cluster_id} ({len(faces)} faces)")
                
                # Load face images
                face_images = []
                for face in faces:
                    import cv2
                    image = cv2.imread(face['original_image_path'])
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    x, y, w, h = face['face_coordinates']
                    face_img = image_rgb[y:y+h, x:x+w]
                    face_images.append(face_img)
                
                # Show faces and confirm they belong together
                result = viewer.show_face_grid(
                    face_images,
                    message=f"Cluster {cluster_id}: Are these all the same person?",
                    allow_individual_selection=False
                )
                
                if not result:
                    click.echo("   ⏭️  Skipped")
                    continue
                
                # Ask for person info
                person_info = viewer.ask_person_info()
                
                if not person_info:
                    click.echo("   ⏭️  Cancelled")
                    continue
                
                # Check if person already exists
                existing_person = db_manager.get_person_by_name(
                    person_info['name'], 
                    person_info['surname']
                )
                
                if existing_person:
                    person_id = existing_person['id']
                    click.echo(f"   ✓ Using existing person: {person_info['name']} {person_info['surname']}")
                else:
                    person_id = db_manager.add_person(
                        name=person_info['name'],
                        surname=person_info['surname'],
                        instagram=person_info['instagram']
                    )
                    click.echo(f"   ✓ Created new person: {person_info['name']} {person_info['surname']}")
                
                # Assign all faces in cluster to this person
                for face in faces:
                    db_manager.update_face_person(face['id'], person_id)
                
                click.echo(f"   ✅ Labeled {len(faces)} faces")
    
    except KeyboardInterrupt:
        click.echo("\n\n⏸️  Labeling interrupted")
    finally:
        db_manager.close()
    
    click.echo("\n✅ Labeling session complete!")


@cli.command()
@click.option('--db', default=config.DATABASE_PATH, help='Database file path')
@click.option('--output', default=config.OUTPUT_DIR, help='Output directory')
def sort(db, output):
    """
    Sort labeled faces into person folders.
    """
    click.echo("📁 Sorting faces into folders...")
    
    # Initialize components
    db_manager = DatabaseManager(db)
    sorter = FaceSorter(db_manager, output_dir=output)
    
    # Sort all faces
    results = sorter.sort_all_labeled_faces()
    
    db_manager.close()
    
    # Display results
    click.echo(f"\n✅ Sorting complete!")
    click.echo(f"   📸 Total faces: {results['total_faces_sorted']}")
    click.echo(f"   🖼️  Original images: {results['total_originals_copied']}")
    click.echo(f"   👥 Persons: {results['persons_processed']}")
    click.echo(f"\n📂 Output directory: {output}")
    
    # Show details
    if results['details']:
        click.echo("\n📋 Details:")
        for person_name, stats in results['details'].items():
            if 'error' in stats:
                click.echo(f"   ❌ {person_name}: {stats['error']}")
            else:
                click.echo(f"   ✓ {person_name}: {stats['faces_sorted']} faces")


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
    
    click.echo("\n📊 Database Statistics")
    click.echo("=" * 40)
    click.echo(f"👥 Total persons: {len(persons)}")
    click.echo(f"📸 Total faces: {len(all_faces)}")
    click.echo(f"🏷️  Labeled faces: {len(all_faces) - len(unlabeled_faces)}")
    click.echo(f"❓ Unlabeled faces: {len(unlabeled_faces)}")
    click.echo(f"📦 Clusters: {len(clusters)}")
    
    if persons:
        click.echo("\n👤 Persons:")
        for person in persons:
            instagram = f" (@{person['instagram']})" if person['instagram'] else ""
            face_count = len(db_manager.get_faces_by_person(person['id']))
            click.echo(f"   • {person['name']} {person['surname']}{instagram}: {face_count} faces")
    
    db_manager.close()


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--db', default=config.DATABASE_PATH, help='Database file path')
@click.option('--output', default=config.OUTPUT_DIR, help='Output directory')
def pipeline(input_dir, db, output):
    """
    Run the complete pipeline: scan -> cluster -> label -> sort.
    
    INPUT_DIR: Directory containing images to process
    """
    click.echo("🚀 Starting complete face sorting pipeline...\n")
    
    # Step 1: Scan
    ctx = click.get_current_context()
    ctx.invoke(scan, input_dir=input_dir, db=db)
    
    click.echo("\n" + "=" * 50 + "\n")
    
    # Step 2: Cluster
    ctx.invoke(cluster, db=db)
    
    click.echo("\n" + "=" * 50 + "\n")
    
    # Step 3: Label
    ctx.invoke(label, db=db)
    
    click.echo("\n" + "=" * 50 + "\n")
    
    # Step 4: Sort
    ctx.invoke(sort, db=db, output=output)
    
    click.echo("\n🎉 Pipeline complete!")


if __name__ == '__main__':
    cli()
