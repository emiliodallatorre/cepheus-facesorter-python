#!/usr/bin/env python3
"""
Example usage script showing the progress output.
This demonstrates what you'll see when running the face sorter.
"""

# Example output when running: python main.py scan ./images

"""
🔍 Scanning directory: ./images
📸 Found 25 images
Processing images  [####################################]  100%  00:00:35

  ✓ photo1.jpg: 2 face(s) detected
    → Generating embedding... ✓
    → Generating embedding... ✓
  ✓ photo2.jpg: 1 face(s) detected
    → Generating embedding... ✓
  ✓ photo3.jpg: 3 face(s) detected
    → Generating embedding... ✓
    → Generating embedding... ✓
    → Generating embedding... ✓

==================================================
✅ Scan complete!
   📸 Images processed: 25
   ✓ Images with faces: 23
   ✗ Images without faces: 2
   👤 Total faces detected: 47
💾 Data saved to: faces.db
==================================================
"""

# Example output when running: python main.py cluster

"""
🔗 Clustering faces by similarity...
📊 Found 47 faces
🔄 Extracting face embeddings...
   ✓ Extracted 47 embeddings
🔄 Running DBSCAN clustering algorithm...
   ✓ Clustering complete
💾 Updating database with cluster assignments...
Updating faces  [####################################]  100%
   ✓ Updated 45 faces

==================================================
✅ Clustering complete!
   📦 Clusters found: 8
   🔍 Outliers (ungrouped): 2
   📈 Average cluster size: 5.6 faces
   📊 Largest cluster: 12 faces
   📊 Smallest cluster: 2 faces
==================================================
"""

# Example output when running: python main.py label

"""
🏷️  Starting face labeling process...
📦 Found 8 clusters to label
💡 Tip: Press Ctrl+C to stop labeling at any time

==================================================
👤 Cluster 0 (12 faces) - Progress: 1/8
==================================================
🔄 Loading face images...
   ✓ Loaded 12 face images
🖼️  Opening face viewer window...
📝 Opening person info form...
   ✓ Created new person: John Doe
💾 Assigning 12 faces to person...
   ✅ Successfully labeled 12 faces

==================================================
👤 Cluster 1 (7 faces) - Progress: 2/8
==================================================
🔄 Loading face images...
   ✓ Loaded 7 face images
🖼️  Opening face viewer window...
   ⏭️  Skipped cluster

==================================================
✅ Labeling session complete!
   ✓ Clusters labeled: 5
   ⏭️  Clusters skipped: 3
   📊 Total clusters: 8
==================================================
"""

# Example output when running: python main.py sort

"""
📁 Sorting faces into folders...
👥 Found 5 persons to process

Sorting persons  [####################################]  100%  00:00:03

  📁 Processing: John Doe
     ✓ 12 faces sorted
  📁 Processing: Jane Smith
     ✓ 8 faces sorted
  📁 Processing: Bob Johnson
     ✓ 5 faces sorted

==================================================
✅ Sorting complete!
   📸 Total faces sorted: 35
   🖼️  Original images copied: 28
   👥 Persons processed: 5
📂 Output directory: output
==================================================

📋 Details by person:
   ✓ John Doe: 12 faces → output/John_Doe
   ✓ Jane Smith: 8 faces → output/Jane_Smith
   ✓ Bob Johnson: 5 faces → output/Bob_Johnson
   ✓ Alice Williams: 6 faces → output/Alice_Williams
   ✓ Charlie Brown: 4 faces → output/Charlie_Brown
"""

print(__doc__)
