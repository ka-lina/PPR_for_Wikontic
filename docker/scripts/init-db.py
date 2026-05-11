#!/usr/bin/env python3
"""Simple database initialization for development"""

import os
import sys
import time
from pymongo import MongoClient
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    mongo_uri = os.environ.get('MONGO_URI', 'mongodb://wikontic:wikontic123@mongodb:27017/?authSource=admin&directConnection=true')
    
    # Wait for MongoDB
    logger.info("Waiting for MongoDB...")
    for i in range(30):
        try:
            client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
            client.admin.command('ping')
            break
        except:
            time.sleep(1)
    else:
        logger.error("MongoDB not available")
        sys.exit(1)
    
    # Check if already initialized
    wikidata_db = os.environ.get('WIKIDATA_ONTOLOGY_DB', 'wikidata_ontology_dev')
    if wikidata_db in client.list_database_names():
        db = client[wikidata_db]
        if 'entity_types' in db.list_collection_names() and db.entity_types.count_documents({}) > 0:
            logger.info("Databases already initialized. Skipping...")
            return
    
    logger.info("Initializing databases (this may take a few minutes)...")
    
    # Run initialization scripts
    subprocess.run([
        "python", "/workspace/Wikontic/create_wikidata_ontology_db.py",
        "--mongo_uri", mongo_uri,
        "--database", wikidata_db,
        "--drop_collections", "true"
    ], check=True)
    
    subprocess.run([
        "python", "/workspace/Wikontic/create_ontological_triplets_db.py",
        "--mongo_uri", mongo_uri,
        "--db_name", os.environ.get('TRIPLETS_DB', 'triplets_db_dev'),
        "--drop_collections", "true"
    ], check=True)
    
    logger.info("✅ Database initialization complete!")

if __name__ == "__main__":
    main()