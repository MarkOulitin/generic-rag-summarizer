import chromadb
import os
from sentence_transformers import SentenceTransformer
from utils import check_gpu_memory, print_model_vram
from logger import logger

client = chromadb.HttpClient(host=os.environ.get('CHROMADB_HOST'), port=int(os.environ.get('CHROMADB_PORT')))
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
