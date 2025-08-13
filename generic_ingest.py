import numpy as np
import chromadb
import uuid
import os
import pickle
import json
import logging
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ingest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GenericIngest:
    def __init__(self):
        self.chromadb_host = os.getenv('CHROMADB_HOST', 'localhost')
        self.chromadb_port = int(os.getenv('CHROMADB_PORT', 8000))
        self.embeddings_dir = Path('./data/embeddings')
        self.topic_domain = os.getenv('TOPICS_DOMAIN', 'general')
        
        try:
            logger.info(f"Connecting to ChromaDB at {self.chromadb_host}:{self.chromadb_port}")
            self.client = chromadb.HttpClient(
                host=self.chromadb_host,
                port=self.chromadb_port
            )
            logger.info(f"Connected to ChromaDB at {self.chromadb_host}:{self.chromadb_port}")
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise
        
        collection_name = self._create_collection_name()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            configuration={
                "hnsw": {
                    "space": "cosine"
                }
            }
        )
        logger.info(f"Using collection: {collection_name}")
    
    def _create_collection_name(self) -> str:
        """Create a safe collection name based on the topic domain"""
        # Clean topic domain to create valid collection name
        safe_name = "".join(c for c in self.topic_domain if c.isalnum() or c in ['-', '_']).lower()
        safe_name = safe_name.replace(' ', '_')
        
        # Ensure it starts with a letter
        if safe_name and not safe_name[0].isalpha():
            safe_name = f"topic_{safe_name}"
        
        # Default fallback
        if not safe_name:
            safe_name = "documents"
        
        return safe_name
    
    def load_embeddings_and_metadata(self) -> tuple[np.ndarray, List[Dict]]:
        """Load embeddings and metadata from files"""
        embeddings_file = self.embeddings_dir / 'embeddings.npy'
        metadata_file = self.embeddings_dir / 'metadata.pickle'
        
        if not embeddings_file.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        logger.info(f"Loading embeddings from {embeddings_file}")
        embeddings = np.load(embeddings_file)
        
        logger.info(f"Loading metadata from {metadata_file}")
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        logger.info(f"Loaded {len(embeddings)} embeddings and {len(metadata)} metadata entries")
        
        if len(embeddings) != len(metadata):
            raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs {len(metadata)} metadata entries")
        
        return embeddings, metadata
    
    def prepare_metadata_for_chromadb(self, metadata_list: List[Dict]) -> List[Dict]:
        """Prepare metadata for ChromaDB ingestion"""
        prepared_metadata = []
        
        for metadata in metadata_list:
            # ChromaDB has limitations on metadata fields and types
            # Only include essential fields and ensure they're strings or numbers
            chromadb_metadata = {
                'chunk_id': str(metadata.get('chunk_id', '')),
                'chunk_index': int(metadata.get('chunk_index', 0)),
                'total_chunks': int(metadata.get('total_chunks', 1)),
                'chunk_size': int(metadata.get('chunk_size', 0)),
                'source_file': str(metadata.get('source_file', '')),
                'title': str(metadata.get('title', ''))[:500],  # Limit title length
                'source': str(metadata.get('source', '')),
                'content_type': str(metadata.get('content_type', '')),
                'domain': str(self.topic_domain),
                'original_url': str(metadata.get('original_url', ''))[:500]  # Limit URL length
            }
            
            # Add processed_at if available
            if 'processed_at' in metadata:
                chromadb_metadata['processed_at'] = str(metadata['processed_at'])
            
            prepared_metadata.append(chromadb_metadata)
        
        return prepared_metadata
    
    def save_embeddings_to_chromadb(self, embeddings: np.ndarray, metadata_array: List[Dict]) -> bool:
        """Save embeddings and metadata to ChromaDB"""
        try:
            # Generate unique IDs
            ids = [str(uuid.uuid4()) for _ in range(len(embeddings))]
            embeddings_list = embeddings.tolist()
            
            # Prepare metadata for ChromaDB
            prepared_metadata = self.prepare_metadata_for_chromadb(metadata_array)
            
            # Add embeddings to collection in batches (ChromaDB has limits)
            batch_size = 1000
            total_batches = (len(embeddings) + batch_size - 1) // batch_size
            
            logger.info(f"Adding {len(embeddings)} embeddings to ChromaDB in {total_batches} batches")
            
            for i in range(0, len(embeddings), batch_size):
                batch_end = min(i + batch_size, len(embeddings))
                batch_ids = ids[i:batch_end]
                batch_embeddings = embeddings_list[i:batch_end]
                batch_metadata = prepared_metadata[i:batch_end]
                
                logger.info(f"Adding batch {i//batch_size + 1}/{total_batches} ({len(batch_ids)} items)")
                
                self.collection.add(
                    embeddings=batch_embeddings,
                    metadatas=batch_metadata,
                    ids=batch_ids
                )
                
                logger.info(f"Successfully added batch {i//batch_size + 1}")
            
            logger.info(f"Successfully added {len(embeddings)} embeddings to collection")
            
            # Test the collection with a query
            self._test_collection(embeddings_list[0])
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving embeddings to ChromaDB: {e}")
            return False
    
    def _test_collection(self, test_embedding: List[float]):
        """Test the collection with a sample query"""
        try:
            logger.info("Testing collection with sample query...")
            results = self.collection.query(
                query_embeddings=[test_embedding],
                n_results=5
            )
            
            logger.info(f"Test query returned {len(results['ids'][0])} results")
            
            # Log some sample results
            for i, (doc_id, metadata) in enumerate(zip(results['ids'][0], results['metadatas'][0])):
                logger.info(f"  Result {i+1}: {metadata.get('title', 'No title')[:50]}...")
                
        except Exception as e:
            logger.warning(f"Collection test failed: {e}")
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        try:
            collection_count = self.collection.count()
            
            # Get a sample of documents to analyze
            sample_results = self.collection.get(limit=100)
            
            sources = {}
            content_types = {}
            
            for metadata in sample_results['metadatas']:
                source = metadata.get('source', 'unknown')
                sources[source] = sources.get(source, 0) + 1
                
                content_type = metadata.get('content_type', 'unknown')
                content_types[content_type] = content_types.get(content_type, 0) + 1
            
            stats = {
                'collection_name': self.collection.name,
                'total_documents': collection_count,
                'domain': self.topic_domain,
                'sources': sources,
                'content_types': content_types
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def ingest_documents(self) -> bool:
        """Main method to ingest documents into ChromaDB"""
        try:
            logger.info("Starting document ingestion process")
            logger.info(f"Domain: {self.topic_domain}")
            
            # Load embeddings and metadata
            embeddings, metadata = self.load_embeddings_and_metadata()
            
            # Save to ChromaDB
            success = self.save_embeddings_to_chromadb(embeddings, metadata)
            
            if success:
                # Get and log collection statistics
                stats = self.get_collection_stats()
                
                logger.info("Ingestion completed successfully!")
                logger.info(f"Collection statistics:")
                for key, value in stats.items():
                    logger.info(f"  {key}: {value}")
                
                # Save ingestion summary
                summary = {
                    'ingestion_completed_at': np.datetime64('now').astype(str),
                    'domain': self.topic_domain,
                    'collection_name': self.collection.name,
                    'documents_ingested': len(embeddings),
                    'chromadb_host': self.chromadb_host,
                    'chromadb_port': self.chromadb_port,
                    'collection_stats': stats
                }
                
                summary_file = self.embeddings_dir / 'ingestion_summary.json'
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2, default=str)
                
                logger.info(f"Ingestion summary saved to: {summary_file}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error during ingestion: {e}")
            return False

def main():
    """Main function to run the ingestion process"""
    ingest = GenericIngest()
    success = ingest.ingest_documents()
    
    if success:
        logger.info("Document ingestion completed successfully!")
        return 0
    else:
        logger.error("Document ingestion failed!")
        return 1

if __name__ == "__main__":
    exit(main())
