import numpy as np
import chromadb
import uuid
import os
from chromadb.config import Settings

client = chromadb.HttpClient(host=os.environ.get('CHROMADB_HOST'), port=int(os.environ.get('CHROMADB_PORT')))
collection = client.get_or_create_collection(
    name="papers",
    configuration={
        "hnsw": {
            "space": "cosine"
        }
    }
)

def save_embeddings_to_chromadb(embeddings, metadata_array):
    collection = client.get_collection(name='papers')
    ids = [str(uuid.uuid4()) for _ in range(len(embeddings))]
    embeddings_list = embeddings.tolist()
    
    # Add embeddings to collection in batches (ChromaDB has limits)
    batch_size = 1000
    total_batches = (len(embeddings) + batch_size - 1) // batch_size
    
    for i in range(0, len(embeddings), batch_size):
        batch_end = min(i + batch_size, len(embeddings))
        batch_ids = ids[i:batch_end]
        batch_embeddings = embeddings_list[i:batch_end]
        batch_metadata = metadata_array[i:batch_end]
        
        print(f"Adding batch {i//batch_size + 1}/{total_batches} ({len(batch_ids)} items)")
        
        collection.add(
            embeddings=batch_embeddings,
            metadatas=batch_metadata,
            ids=batch_ids
        )
    
    print(f"Successfully added {len(embeddings)} embeddings to collection")
    return collection

import pickle

for filename in ['PMC000xxxxxx', 'PMC001xxxxxx', 'PMC002xxxxxx']:
    print(f'Ingesting {filename}')
    with open(f'./data/{filename}.pickle', 'rb') as file:
        metadata = pickle.load(file)

    embeddings = np.load(f'./data/{filename}.npy')
    collection = save_embeddings_to_chromadb(embeddings, metadata)
    print(f'Ingesting {filename} completed')


print('Testing ingestion of last collection')
with open(f'./data/{filename}.pickle', 'rb') as file:
    metadata = pickle.load(file)

embeddings = np.load(f'./data/{filename}.npy')

query_embedding = embeddings[0].tolist()
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)
print("Query results:", results)