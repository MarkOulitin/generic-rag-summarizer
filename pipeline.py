import argparse
import logging
import time
from pathlib import Path
from dotenv import load_dotenv
import os

from data_collector import DocumentCollector
from document_processor import DocumentProcessor
from embedding_generator import EmbeddingGenerator

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self):
        self.topic_domain = os.getenv('TOPICS_DOMAIN', 'artificial intelligence and machine learning')
        self.max_documents = int(os.getenv('MAX_DOCUMENTS', 2000))
        
        # Create data directories
        Path('./data').mkdir(exist_ok=True)
        Path('./data/collected_documents').mkdir(parents=True, exist_ok=True)
        Path('./data/processed_chunks').mkdir(parents=True, exist_ok=True)
        Path('./data/embeddings').mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized RAG Pipeline for domain: '{self.topic_domain}'")
        logger.info(f"Target documents: {self.max_documents}")
    
    def run_document_collection(self) -> bool:
        """Step 1: Collect documents from the web"""
        logger.info("=" * 50)
        logger.info("STEP 1: DOCUMENT COLLECTION")
        logger.info("=" * 50)
        
        try:
            collector = DocumentCollector()
            collector.collect_documents()
            
            # Check if documents were collected
            docs_dir = Path('./data/collected_documents')
            collected_files = [f for f in docs_dir.iterdir() if f.is_file() and f.suffix != '.json']
            
            if collected_files:
                logger.info(f"Document collection completed: {len(collected_files)} files collected")
                return True
            else:
                logger.error("No documents were collected")
                return False
                
        except Exception as e:
            logger.error(f"Document collection failed: {e}")
            return False
        finally:
            # Cleanup collector resources
            try:
                collector.cleanup()
            except:
                pass
    
    def run_document_processing(self) -> bool:
        """Step 2: Process and chunk documents"""
        logger.info("=" * 50)
        logger.info("STEP 2: DOCUMENT PROCESSING")
        logger.info("=" * 50)
        
        try:
            processor = DocumentProcessor()
            chunks = processor.process_all_documents()
            
            if chunks:
                logger.info(f"Document processing completed: {len(chunks)} chunks created")
                return True
            else:
                logger.error("No chunks were created during processing")
                return False
                
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return False
    
    def run_embedding_generation(self) -> bool:
        """Step 3: Generate embeddings for chunks"""
        logger.info("=" * 50)
        logger.info("STEP 3: EMBEDDING GENERATION")
        logger.info("=" * 50)
        
        try:
            generator = EmbeddingGenerator()
            success = generator.generate_embeddings_for_chunks()
            
            if success:
                logger.info("Embedding generation completed successfully")
                return True
            else:
                logger.error("Embedding generation failed")
                return False
                
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return False
        
    def run_complete_pipeline(self, skip_collection=False, skip_processing=False, 
                            skip_embeddings=False) -> bool:
        """Run the complete RAG pipeline"""
        logger.info("🚀 Starting Complete RAG Pipeline")
        logger.info(f"Domain: {self.topic_domain}")
        
        start_time = time.time()
        
        try:
            # Step 1: Document Collection
            if not skip_collection:
                if not self.run_document_collection():
                    logger.error("Pipeline failed at document collection step")
                    return False
            else:
                logger.info("Skipping document collection")
            
            # Step 2: Document Processing
            if not skip_processing:
                if not self.run_document_processing():
                    logger.error("Pipeline failed at document processing step")
                    return False
            else:
                logger.info("Skipping document processing")
            
            # Step 3: Embedding Generation
            if not skip_embeddings:
                if not self.run_embedding_generation():
                    logger.error("Pipeline failed at embedding generation step")
                    return False
            else:
                logger.info("Skipping embedding generation")
            
            elapsed_time = time.time() - start_time
            logger.info("=" * 50)
            logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(f"Total time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
            logger.info("=" * 50)
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            return False

def main():
    """
    Pipeline for Document Collection

    This script runs the complete pipeline:
    1. Collect documents from the web based on topic domain
    2. Process and chunk the documents
    3. Generate embeddings for the chunks
    
    Usage:
        python pipeline.py [--skip-collection] [--skip-processing] [--skip-embeddings]
    """

    parser = argparse.ArgumentParser(description='Run the complete RAG document pipeline')
    parser.add_argument('--skip-collection', action='store_true', 
                       help='Skip document collection step')
    parser.add_argument('--skip-processing', action='store_true', 
                       help='Skip document processing step')
    parser.add_argument('--skip-embeddings', action='store_true', 
                       help='Skip embedding generation step')
    parser.add_argument('--step', choices=['collection', 'processing', 'embeddings'],
                       help='Run only a specific step')
    
    args = parser.parse_args()
    
    pipeline = RAGPipeline()
    
    # Run specific step if requested
    if args.step:
        logger.info(f"Running only step: {args.step}")
        if args.step == 'collection':
            success = pipeline.run_document_collection()
        elif args.step == 'processing':
            success = pipeline.run_document_processing()
        elif args.step == 'embeddings':
            success = pipeline.run_embedding_generation()

    else:
        # Run complete pipeline
        success = pipeline.run_complete_pipeline(
            skip_collection=args.skip_collection,
            skip_processing=args.skip_processing,
            skip_embeddings=args.skip_embeddings,
        )
    
    if success:
        logger.info("Pipeline execution completed successfully!")
        return 0
    else:
        logger.error("Pipeline execution failed!")
        return 1

if __name__ == "__main__":
    exit(main())
