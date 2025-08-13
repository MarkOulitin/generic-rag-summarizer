import torch
import numpy as np
import xml.etree.ElementTree as ET
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import check_gpu_memory, print_model_vram
from logger import logger
logger.info("Loading model...")
check_gpu_memory()

reranker_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
reranker_model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3', device_map='cuda')
reranker_model.eval()

logger.info("Model loaded successfully!")
check_gpu_memory()

print_model_vram(reranker_model)

def rerank(query, chunks, top_k):
    # Prepare texts for reranking
    texts_to_rank = []
    for chunk in chunks:
        metadata = chunk['metadata']
        title = metadata.get('title', 'No Title')
        content = chunk.get('content', '')
        
        if content:
            chunk_text = f"Title: {title}\n\nContent: {content}"
        else:
            chunk_text = f"Title: {title}\n\nSource: {metadata.get('source_file', 'Unknown')}"
        
        texts_to_rank.append(chunk_text)
    
    scores = []
    with torch.no_grad():
        for text in texts_to_rank:
            pair = [query, text]
            inputs = reranker_tokenizer([pair], padding=True, truncation=True, return_tensors='pt', max_length=512)
            inputs = {k: v.to(reranker_model.device) for k, v in inputs.items()}
            score = reranker_model(**inputs, return_dict=True).logits.view(-1, ).float()
            scores.append(score)
    
    scores = torch.stack(scores).squeeze().cpu().numpy()
    
    top_k_indexes = np.argsort(scores)[-top_k:][::-1]
    
    reranked_chunks = []
    for idx in top_k_indexes:
        chunk = chunks[idx].copy()
        chunk['rerank_score'] = float(scores[idx])
        reranked_chunks.append(chunk)
    

    return reranked_chunks
