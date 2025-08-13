import chromadb
import os
from sentence_transformers import SentenceTransformer
from utils import check_gpu_memory, print_model_vram
from logger import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GenericRetrieval:
    def __init__(self):
        self.chromadb_host = os.getenv('CHROMADB_HOST', 'localhost')
        self.chromadb_port = int(os.getenv('CHROMADB_PORT', 8000))
        self.topic_domain = os.getenv('TOPICS_DOMAIN', 'artificial intelligence and machine learning')
        
        # Initialize ChromaDB client
        self.client = chromadb.HttpClient(
            host=self.chromadb_host, 
            port=self.chromadb_port
        )
        
        # Create collection name from topic domain
        self.collection_name = self._create_collection_name()
        
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Connected to collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to connect to collection '{self.collection_name}': {e}")
            logger.info("Available collections:")
            try:
                collections = self.client.list_collections()
                for col in collections:
                    logger.info(f"  - {col.name}")
            except:
                logger.error("Could not list collections")
            raise
        
        # Load embedding model
        logger.info("Loading model...")
        check_gpu_memory()
        
        self.model = SentenceTransformer(
            "Qwen/Qwen3-Embedding-0.6B",
            model_kwargs={"device_map": "cuda"},
            tokenizer_kwargs={"padding_side": "left"},
        )
        
        logger.info("Model loaded successfully!")
        check_gpu_memory()
        print_model_vram(self.model)
    
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
    
    def retrieve(self, query: str, top_k: int):
        """Retrieve relevant documents based on query"""
        # Generate query embedding
        query_embedding = self.model.encode(
            query,
            show_progress_bar=False,
            convert_to_tensor=True,
            device='cuda'
        )
        query_embedding = query_embedding.cpu().tolist()
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        logger.info(f"Retrieved {len(results['metadatas'][0])} results for query: {query[:50]}...")
        
        return results['metadatas'][0]

# Create global instance for backward compatibility
client = chromadb.HttpClient(host=os.getenv('CHROMADB_HOST'), port=int(os.getenv('CHROMADB_PORT')))

# Try to use generic retrieval, fallback to original papers collection
try:
    generic_retrieval = GenericRetrieval()
    
    def retrieve(query: str, top_k):
        """Wrapper function for backward compatibility"""
        return generic_retrieval.retrieve(query, top_k)
        
except Exception as e:
    logger.warning(f"Failed to initialize generic retrieval: {e}")
    logger.info("Falling back to original papers collection")
    
    # Fallback to original implementation
    collection = client.get_or_create_collection(name="papers")
    
    logger.info("Loading model...")
    check_gpu_memory()
    
    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={"device_map": "cuda"},
        tokenizer_kwargs={"padding_side": "left"},
    )
    
    logger.info("Model loaded successfully!")
    check_gpu_memory()
    print_model_vram(model)
    
    def retrieve(query: str, top_k):
        query_embedding = model.encode(
            query,
            show_progress_bar=False,
            convert_to_tensor=True,
            device='cuda'
        )
        query_embedding = query_embedding.cpu().tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        logger.info("Query results:", results)
        return results['metadatas'][0]
