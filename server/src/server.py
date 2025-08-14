import asyncio
import json
import websockets
import traceback
import os
from dotenv import load_dotenv
load_dotenv()

from langgraph_workflow import process_rag_query, get_workflow
from logger import logger

RETRIVAL_TOP_K = int(os.environ.get('RERANK_TOP_K'))
RERANK_TOP_K = int(os.environ.get('RETRIVAL_TOP_K'))


async def handle_chat_message(websocket, message_data):
    """Handle chat message with conversation context"""
    query = message_data.get("message", "")
    conversation_id = message_data.get("conversation_id", "default")
    
    logger.info(f"Chat message - Query: {query}, Conversation ID: {conversation_id}")
    
    try:
        async def generate_callback(text_chunk):
            event = {"event": "generating", "data": text_chunk}
            await websocket.send(json.dumps(event))
            
        summary = await process_rag_query(
            query, 
            conversation_id=conversation_id, 
            generate_callback=generate_callback
        )
            
        if summary.get('error') is None:
            event = {"event": "completed", "data": summary}
            await websocket.send(json.dumps(event))    
        else:
            event = {"event": "error", "data": str(summary['error'])}
            await websocket.send(json.dumps(event))
            logger.error(f"Workflow returned error. Sent event: error, Error: {summary['error']}")
    except Exception as e:
        event = {"event": "error", "data": str(e)}
        await websocket.send(json.dumps(event))
        trace = traceback.format_exc()
        logger.error(f"Sent event: error, Error: {e}\n{trace}")

async def handle_clear_conversation(websocket, message_data):
    """Handle clearing conversation history"""
    conversation_id = message_data.get("conversation_id", "default")
    logger.info(f"Clearing conversation: {conversation_id}")
    
    try:
        workflow = get_workflow()
        workflow.clear_conversation(conversation_id)
        
        event = {"event": "conversation_cleared", "data": {"conversation_id": conversation_id}}
        await websocket.send(json.dumps(event))
        logger.info(f"Conversation {conversation_id} cleared successfully")
    except Exception as e:
        event = {"event": "error", "data": str(e)}
        await websocket.send(json.dumps(event))
        logger.error(f"Error clearing conversation: {e}")

async def websocket_handler(websocket):
    logger.info(f"Client connected")
    try:
        async for message in websocket:
            logger.info(f"Received message: {message}")
            try:
                data = json.loads(message)
                action = data.get("action", "chat")
                
                if action == "chat":
                    # Handle chat message
                    if "message" in data:
                        await handle_chat_message(websocket, data)
                    elif "query" in data:  # Backward compatibility
                        message_data = {
                            "message": data["query"],
                            "conversation_id": data.get("conversation_id", "default")
                        }
                        await handle_chat_message(websocket, message_data)
                    else:
                        error_response = {
                            "event": "error",
                            "data": "Missing 'message' or 'query' in JSON body."
                        }
                        await websocket.send(json.dumps(error_response))
                        logger.error(f"Sent error response: {json.dumps(error_response)}")
                        
                elif action == "clear_conversation":
                    await handle_clear_conversation(websocket, data)
                    
                else:
                    error_response = {
                        "event": "error",
                        "data": f"Unknown action: {action}"
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
    get_workflow()
    async with websockets.serve(websocket_handler, host, port):
        logger.info(f"WebSocket server started on ws://{host}:{port}")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.error("\nServer stopped by user.")
