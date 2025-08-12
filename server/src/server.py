import asyncio
import json
import websockets
import retrival 
import rerenking
import generation
import traceback
import time
import os
from logger import logger

RETRIVAL_TOP_K = int(os.environ.get('RERANK_TOP_K'))
RERANK_TOP_K = int(os.environ.get('RETRIVAL_TOP_K'))

async def summarize_papers(websocket, query, user_role):
    logger.info(f"Parsed JSON - Query: {query}, User Role: {user_role}")
    try:
        stage_start_time = time.time()
        event = {"event": "extracting_candidates"}
        await websocket.send(json.dumps(event))
        logger.info(f"Sent event: extracting_candidates")
        
        papers_metadata = retrival.retrieve(query, top_k=RETRIVAL_TOP_K)
        
        event = {"event": "extracted_candidates", "time": time.time() - stage_start_time}
        await websocket.send(json.dumps(event))
        logger.info(f"Sent event: extracted_candidates, time: {event['time']:.3f}")
        
        stage_start_time = time.time()
        event = {"event": "reranking_candidates"}
        await websocket.send(json.dumps(event))
        logger.info(f"Sent event: reranking_candidates")
        
        papers = rerenking.get_papers_data(papers_metadata)
        reranked_papers = rerenking.rerank(query, papers, top_k=RERANK_TOP_K)
        
        event = {"event": "reranked_candidates", "time": time.time() - stage_start_time}
        await websocket.send(json.dumps(event))
        logger.info(f"Sent event: reranked_candidates, time: {event['time']:.3f}")
        
        event = {"event": "generating", "data": ''}
        await websocket.send(json.dumps(event))
        logger.info(f"Sent event: generating")
        
        summary = ""
        async for text_chunk in generation.generate(reranked_papers, query, user_role):
            summary += text_chunk
            event = {"event": "generating", "data": text_chunk}
            await websocket.send(json.dumps(event))
            logger.info(f"Sent event: generating")
        
        event = {"event": "evaluating"}
        await websocket.send(json.dumps(event))
        logger.info(f"Sent event: evaluating")
        
        evaluation = await generation.evaluate(query, user_role, summary, reranked_papers)
        
        event = {"event": "evaluation_completed", "data": evaluation}
        await websocket.send(json.dumps(event))
        logger.info(f"Sent event: evaluation_completed" + json.dumps(event))
    except Exception as e:
        event = {"event": "error", "data": str(e)}
        await websocket.send(json.dumps(event))
        trace = traceback.format_exc()
        logger.error(f"Sent event: error, Error: {e}\n{trace}")

async def websocket_handler(websocket):
    logger.info(f"Client connected")
    try:
        async for message in websocket:
            logger.info(f"Received message: {message}")
            try:
                data = json.loads(message)
                if "query" in data and "user_role" in data:
                    query = data["query"]
                    user_role = data["user_role"]
                    await summarize_papers(websocket, query, user_role)
                else:
                    error_response = {
                        "event": "error",
                        "data": "Missing 'query' or 'user_role' in JSON body."
                    }
                    await websocket.send(json.dumps(error_response))
                    logger.error(f"Sent error response: {json.dumps(error_response)}")
            except json.JSONDecodeError:
                error_response = {
                    "event": "error",
                    "data": "Invalid JSON format received."
                }
                await websocket.send(json.dumps(error_response))
                logger.error(f"Sent error response: {json.dumps(error_response)}")
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}")
                error_response = {
                    "event": "error",
                    "data": f"Server error: {str(e)}"
                }
                await websocket.send(json.dumps(error_response))
                logger.error(f"Sent error response: {json.dumps(error_response)}")
    except websockets.exceptions.ConnectionClosedOK:
        logger.error(f"Client disconnected gracefully")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.error(f"Client disconnected with error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket handler: {e}")
    finally:
        logger.info(f"Client connection closed")

async def main():
    """
    Starts the WebSocket server.
    """
    host = "0.0.0.0"
    port = int(os.environ.get('SERVER_PORT', 9090))
    async with websockets.serve(websocket_handler, host, port):
        logger.info(f"WebSocket server started on ws://{host}:{port}")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.error("\nServer stopped by user.")
