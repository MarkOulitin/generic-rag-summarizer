# RAG based Chatbot 🤖
# Design description
- For user query, I extract top (K) 70 similar chunks of documents from the ChromaDB
    - Documents are chunked and then embedded with some overlap between chunks
    - We get K nearest neigbors with cosine similarity
    - I chose ChromaDB due to it's simplicity
- After retrival, I rerank the 70 chunks to the user query. Because there is some query to chunk match (interaction) that can't be captured by cosine similary of k nearest neighbors.
- After reranking I take top (K) 10 chunks. Then I take all their body, user query to construct a prompt and then send them to ChatGPT 4o-mini for summarization with citations.
- After generation, I take the summary, user query and conversation history to generate folloup questions as recommended options.
- Due to VRAM constraints (6 GB) of my PC GPU, I had to pick the right models to fit in the VRAM. Exhausting api token to some 3rd party LLM for embedding the dataset isn't cost effective.
    - For embedding the user query and papers into ChromaDB, I used [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) because:
        - It was in the 4th place of [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard).
        - It's VRAM footprint was 2.2GB.
        - Context window of 32k - enough for embedding paper's title and abstract.
    - For reranking model I used cross encoder [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) because:
        - It's reported MRR peformace from author's paper good compared to the other models.
        - It takes only 2.1 GB VRAM.
        - Context window of 8k - enough for embedding paper's title and abstract.
    - Using these two models I could embed the dataset and make inference on my GTX 1060.
    - Of course it's slow not only because of the GPU (# of tensor cores) but also because I couldn't use flash attention 2 on GTX 1060. And it's slow because I didn't try to use quantized version, e.g. FP16.
- After narrowing our search to 10 chunks we can use LLM with larger context window, in this case GPT 4o-mini.
- Conversation is stored in-memory dictionary however it could be exported to Redis.
- User friendly web client. You can also copy the generated converation markdown to the clipboard and see the progress online!

# Acknowledgement
I started developing this on my PC with GTX 1060 using Qwen3 0.6B LLM. In the end I tested on L40s GPU machine on DataCrunch.

# Minimal requirements
- GTX 1060 with 6GB VRAM
- CUDA 12.8

# Setup
## Add your OpenAI key
In `./.env`:
```
OPENAI_API_KEY=[OPENAI_API_KEY]
```
We have contaier for server and container for chromadb
## (Optional) Download and index the documents
This step is optional because I will provide a pre-downloaded documents that also underwent indexing
```sh
conda create --name embed python=3.12 -y
conda activate embed
pip install -r ./server/requirements.txt
# optionally [--skip-collection] [--skip-processing] [--skip-embeddings]
python pipeline.py
```
## Start the server and vector database
```sh
docker build -t rag-server:latest ./server
docker compose up -d
# Important, populate ChromaDB
docker exec -it server python ingest.py
```
## Run client web application:
### Set server's host
In `./client/.env` set `SERVER_HOST`. Example:
```
SERVER_HOST=135.181.63.183
```
### Run client web application
```sh
conda create --name test-client python=3.12 -y
conda activate test-client
pip install -r ./client/requirements.txt
cd client && python ./client.py
```
