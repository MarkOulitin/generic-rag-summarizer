import torch
import time
import pickle
import numpy as np
import json
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embedding_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_gpu_memory():
    """Check available GPU memory"""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        allocated_memory = torch.cuda.memory_allocated(0) / 1024**3  # GB
        cached_memory = torch.cuda.memory_reserved(0) / 1024**3  # GB
        
        logger.info(f"GPU Memory - Total: {total_memory:.2f}GB, Allocated: {allocated_memory:.2f}GB, Cached: {cached_memory:.2f}GB")
        logger.info(f"Available: {total_memory - cached_memory:.2f}GB")
        return total_memory - cached_memory
    else:
        logger.info("CUDA not available")
        return 0

def print_model_vram(model):
    """Print model VRAM usage"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size_bytes = param_size + buffer_size
    total_size_mb = total_size_bytes / (1024 * 1024)

    logger.info(f"Model size: {total_size_mb:.2f} MB")

class EmbeddingGenerator:
    def __init__(self):
        self.input_dir = Path('./data/processed_chunks')
        self.output_dir = Path('./data/embeddings')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        logger.info("Loading embedding model...")
        check_gpu_memory()
        
        self.model = SentenceTransformer(
            "Qwen/Qwen3-Embedding-0.6B",
            model_kwargs={"device_map": "cuda" if torch.cuda.is_available() else "cpu"},
            tokenizer_kwargs={"padding_side": "left"},
        )
        
        logger.info("Model loaded successfully!")
        check_gpu_memory()
        print_model_vram(self.model)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
    
    def generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts with memory management"""
        if not texts:
            return np.array([])
        
        batch_size = 16
        
        logger.info(f"Processing {len(texts)} texts with batch size {batch_size}")
        
        embeddings_list = []
        start_time = time.time()
        
        try:
            for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
                batch_texts = texts[i:i + batch_size]
                
                batch_embeddings = self.model.encode(
                    batch_texts,
                    show_progress_bar=False,
                    convert_to_tensor=True,
                    device=self.device
                )
                
                embeddings_list.append(batch_embeddings.cpu().numpy())
                
                # Clear cache periodically
                if i % (batch_size * 15) == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            embeddings = np.vstack(embeddings_list)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Embedding generation completed in {elapsed_time:.1f} seconds")
            logger.info(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
            
            return embeddings
            
        except torch.cuda.OutOfMemoryError as e:
            logger.warning("GPU out of memory! Trying smaller batch size...")
            torch.cuda.empty_cache()
            
            # Retry with smaller batch size
            smaller_batch_size = max(1, batch_size // 2)
            logger.info(f"Retry with batch size {smaller_batch_size}")
            raise e
        
        except Exception as e:
            logger.error(f"Error during embedding generation: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
    
    def prepare_chunks_for_embedding(self, chunks: List[Dict]) -> tuple[List[str], List[Dict]]:
        """Prepare chunk texts and metadata for embedding generation"""
        texts = []
        chunk_contents = []
        metadata = []
        
        for chunk in chunks:
            if 'text' in chunk and 'metadata' in chunk:
                # Create embedding text combining title and content
                chunk_text = chunk['text']
                chunk_metadata = chunk['metadata']
                
                # Optionally prepend title for better context
                if chunk_metadata.get('title') and chunk_metadata['title'] != chunk_metadata.get('source_file', ''):
                    embedding_text = f"Title: {chunk_metadata['title']}\n\nContent: {chunk_text}"
                else:
                    embedding_text = chunk_text
                
                chunk_contents.append(chunk_text)
                texts.append(embedding_text)
                metadata.append(chunk_metadata)
            else:
                logger.warning(f"Skipping invalid chunk: {chunk}")
        
        logger.info(f"Prepared {len(texts)} chunks for embedding generation")
        return texts, metadata, chunk_contents
    
    def generate_embeddings_for_chunks(self) -> bool:
        """Generate embeddings for all processed chunks"""
        # Load processed chunks
        chunks_file = self.input_dir / 'processed_chunks.json'
        
        if not chunks_file.exists():
            logger.error(f"Processed chunks file not found: {chunks_file}")
            logger.error("Please run document_processor.py first to create processed chunks")
            return False
        
        logger.info(f"Loading chunks from {chunks_file}")
        
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
        except Exception as e:
            logger.error(f"Error loading chunks file: {e}")
            return False
        
        if not chunks:
            logger.error("No chunks found in the file")
            return False
        
        logger.info(f"Loaded {len(chunks)} chunks")
        
        texts, metadata, chunk_contents = self.prepare_chunks_for_embedding(chunks)
        
        if not texts:
            logger.error("No valid texts found for embedding generation")
            return False
        
        # Check initial GPU memory
        logger.info("Initial GPU memory status:")
        check_gpu_memory()
        
        # Generate embeddings
        try:
            embeddings = self.generate_embeddings_batch(texts)
            
            if embeddings.size == 0:
                logger.error("No embeddings generated")
                return False
            
            # Save embeddings and metadata
            embeddings_file = self.output_dir / 'embeddings.npy'
            metadata_file = self.output_dir / 'metadata.pickle'
            content_file = self.output_dir / 'content.pickle'
            
            logger.info(f"Saving embeddings to {embeddings_file}")
            np.save(embeddings_file, embeddings)
            
            logger.info(f"Saving metadata to {metadata_file}")
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            logger.info(f"Saving chunks content to {content_file}")
            with open(content_file, 'wb') as f:
                pickle.dump(chunk_contents, f)
            
            # Save generation summary
            summary = {
                'generation_completed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_chunks': len(chunks),
                'embeddings_generated': len(embeddings),
                'embedding_dimension': embeddings.shape[1],
                'model_used': "Qwen/Qwen3-Embedding-0.6B",
                'embeddings_file': str(embeddings_file),
                'metadata_file': str(metadata_file),
                'average_text_length': sum(len(text.split()) for text in texts) / len(texts)
            }
            
            summary_file = self.output_dir / 'generation_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Embedding generation completed successfully!")
            logger.info(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
            logger.info(f"Files saved:")
            logger.info(f"  - Embeddings: {embeddings_file}")
            logger.info(f"  - Metadata: {metadata_file}")
            logger.info(f"  - Summary: {summary_file}")
            
        except Exception as e:
            logger.error(f"Error during embedding generation: {e}")
            return False
        
        # Final GPU memory status
        logger.info("Final GPU memory status:")
        check_gpu_memory()
        
        return True

def main():
    generator = EmbeddingGenerator()
    success = generator.generate_embeddings_for_chunks()
    
    if success:
        logger.info("Embedding generation process completed successfully!")
        return 0
    else:
        logger.error("Embedding generation process failed!")
        return 1

if __name__ == "__main__":
    exit(main())
