import chromadb
import os
from sentence_transformers import SentenceTransformer
from utils import check_gpu_memory, print_model_vram
from logger import logger

def create_collection_name(topic_domain) -> str:
    """Create a safe collection name based on the topic domain"""
    safe_name = "".join(c for c in topic_domain if c.isalnum() or c in ['-', '_']).lower()
    safe_name = safe_name.replace(' ', '_')
    
    if safe_name and not safe_name[0].isalpha():
        safe_name = f"topic_{safe_name}"
    
    if not safe_name:
        safe_name = "documents"
    
    return safe_name

client = chromadb.HttpClient(host=os.environ.get('CHROMADB_HOST'), port=int(os.environ.get('CHROMADB_PORT')))
topic_domain = os.environ['TOPICS_DOMAIN']
collection_name = create_collection_name(topic_domain)
collection = client.get_collection(name=collection_name)

logger.info("Loading model...")
check_gpu_memory()

embedding_model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={"device_map": "cuda"},
    tokenizer_kwargs={"padding_side": "left"},
)

logger.info("Model loaded successfully!")
check_gpu_memory()

print_model_vram(embedding_model)

def retrieve(query: str, top_k):
    query_embedding = embedding_model.encode(
        query,
        show_progress_bar=False,
        convert_to_tensor=True,
        device='cuda'
    )
    query_embedding = query_embedding.cpu().tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=['documents', 'metadatas', 'distances']
    )
    # Format retrieved chunks
    retrieved_chunks = []
    for i, (chunk_id, metadata, document) in enumerate(zip(
        results['ids'][0], 
        results['metadatas'][0], 
        results['documents'][0]
    )):
        chunk_data = {
            'chunk_id': chunk_id,
            'metadata': metadata,
            'content': document,  # The actual chunk text content
            'relevance_score': results['distances'][0][i] if 'distances' in results else None
        }
        retrieved_chunks.append(chunk_data)
    
    return retrieved_chunks
