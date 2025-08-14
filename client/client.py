import gradio as gr
import json
import time
from typing import Dict, Any
import asyncio
import websockets
import os
import time
from logger import logger
from dotenv import load_dotenv
load_dotenv()

class RAGClient:
    def __init__(self):
        self.current_summary = ""
        self.stage_times = {}
        self.conversation_id = "default"
        self.chat_history = []  # List of {"role": "user/assistant", "content": str, "timestamp": str}
        
    def format_time(self, seconds: float) -> str:
        """Format time in seconds to readable format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        else:
            minutes = int(seconds // 60)
            seconds = seconds % 60
            return f"{minutes}m {seconds:.1f}s"
    
    def create_status_report(self) -> str:
        """Create a status report of all completed stages"""
        report = "## Processing Status\n\n"
        
        stage_names = {
            'extraction': 'Extracting candidates',
            'reranking': 'Reranking candidates', 
            'generation': 'Generating summary',
        }
        
        for stage, name in stage_names.items():
            if stage in self.stage_times:
                time_str = self.format_time(self.stage_times[stage])
                report += f"✅ **{name}** - {time_str}\n\n"
        
        return report
    
    def add_message_to_history(self, role: str, content: str):
        """Add a message to chat history"""
        timestamp = time.strftime("%H:%M", time.localtime())
        self.chat_history.append({
            "role": role,
            "content": content,
            "timestamp": timestamp
        })
    
    def clear_chat_history(self):
        """Clear the chat history"""
        self.chat_history = []
    
    def format_chat_history(self) -> str:
        """Format chat history for display"""
        if not self.chat_history:
            return "No messages yet. Start the conversation!"
        
        formatted_history = ""
        for message in self.chat_history:
            role_emoji = "👤" if message["role"] == "user" else "🤖"
            role_name = "You" if message["role"] == "user" else "Assistant"
            formatted_history += f"\n### {role_emoji} {role_name} *({message['timestamp']})*\n\n{message['content']}\n\n---\n"
        
        return formatted_history
        
    async def process_query(self, query: str, summary_callback=None, status_callback=None, history_callback=None):
        """Process the query and handle server-sent events"""

        self.current_summary = ""
        self.stage_times = {}

        # Add user message to history
        self.add_message_to_history("user", query)
        
        # Update history display
        if history_callback:
            history_callback(self.format_chat_history())

        # Clear previous results
        if status_callback:
            status_callback("Connecting to server...")
        if summary_callback:
            summary_callback("")

        try:
            uri = f"ws://{os.environ.get('SERVER_HOST', 'localhost')}:{os.environ.get('SERVER_PORT', 9090)}/"
            logger.info(f"Connecting to {uri}")
            async with websockets.connect(uri) as websocket:
                logger.info(f"Connected to {uri}")
                message = {
                    "action": "chat",
                    "message": query,
                    "conversation_id": self.conversation_id
                }
                logger.info(f"Sending message: {message}")

                await websocket.send(json.dumps(message))

                try:
                    while True:
                        response_json = await websocket.recv()
                        response_data = json.loads(response_json)
                        event_type = response_data.get('event')
                        data = response_data.get('data')

                        if event_type == 'extracting_candidates':
                            pass
                            # logger.info(f"Received {response_json}")
                            # logger.info(f"If status_callback: {bool(status_callback)}, {self.stage_times}")
                            # if status_callback:
                            #     status_callback(self.create_status_report() + "\n⏳ Extracting candidates...")

                        elif event_type == 'extracted_candidates':
                            pass
                            # logger.info(f"Received {response_json}")
                            # self.stage_times['extraction'] = response_data.get("time")
                            # logger.info(f"If status_callback: {bool(status_callback)}, {self.stage_times}")
                            # if status_callback:
                            #     status_callback(self.create_status_report())
                        
                        elif event_type == 'reranking_candidates':
                            pass
                            # logger.info(f"Received {response_json}")
                            # logger.info(f"If status_callback: {bool(status_callback)}, {self.stage_times}")
                            # if status_callback:
                            #     status_callback(self.create_status_report() + "\n⏳ Reranking candidates...")

                        elif event_type == 'reranked_candidates':
                            pass
                            # logger.info(f"Received {response_json}")
                            # logger.info(f"If status_callback: {bool(status_callback)}, {self.stage_times}")
                            # self.stage_times['reranking'] = response_data.get("time")
                            # if status_callback:
                            #     status_callback(self.create_status_report())

                        elif event_type == 'generating':
                            if 'generation' not in self.stage_times:
                                if status_callback:
                                    status_callback(self.create_status_report() + "\n⏳ Generating summary...")
                                self.stage_times['generation'] = time.time()
                                logger.info(f"If generation: {self.stage_times}")

                            if data and summary_callback:
                                self.current_summary += data
                                summary_callback(self.current_summary)

                        elif event_type == 'completed':
                            stage_time = time.time()
                            self.stage_times['generation'] = stage_time - self.stage_times['generation']
                            
                            # Add assistant response to history
                            if self.current_summary:
                                self.add_message_to_history("assistant", self.current_summary)
                                if history_callback:
                                    history_callback(self.format_chat_history())
                            
                            if status_callback:
                                status_callback(self.create_status_report() + "\n✅ **Completed!**")
                            break
                        
                        elif event_type == 'error':
                            if status_callback:
                                status_callback(f"An error occurred: {data}")
                            break
                except websockets.exceptions.ConnectionClosed:
                    logger.error("Connection closed.")
                except json.JSONDecodeError:
                    logger.error(f"Received non-JSON response: {response_json}")
                except Exception as e:
                    logger.error(f"An error occurred while receiving: {e}")
                finally:
                    await websocket.close()

        except Exception as e:
            error_message = f"Failed to connect or process: {e}"
            logger.error(error_message)
            if status_callback:
                status_callback(error_message)

    async def clear_conversation(self):
        """Clear the conversation history both locally and on server"""
        try:
            uri = f"ws://{os.environ.get('SERVER_HOST', 'localhost')}:{os.environ.get('SERVER_PORT', 9090)}/"
            async with websockets.connect(uri) as websocket:
                message = {
                    "action": "clear_conversation",
                    "conversation_id": self.conversation_id
                }
                await websocket.send(json.dumps(message))
                
                # Wait for confirmation
                response_json = await websocket.recv()
                response_data = json.loads(response_json)
                
                if response_data.get("event") == "conversation_cleared":
                    self.clear_chat_history()
                    logger.info("Conversation cleared successfully")
                    return True
                else:
                    logger.error(f"Failed to clear conversation: {response_data}")
                    return False
        except Exception as e:
            logger.error(f"Error clearing conversation: {e}")
            return False

client = RAGClient()
def create_gradio_app():
    """Create the Gradio chat interface"""

    async def send_message(message, history_display, status_display):
        """Handle sending a chat message"""
        if not message.strip():
            yield ("", history_display, status_display, "")
        else:
            summary_output = ""
            status_output = ""
            history_output = ""
            
            def update_summary(content):
                nonlocal summary_output
                summary_output = content
            
            def update_status(content):
                nonlocal status_output
                status_output = content
                
            def update_history(content):
                nonlocal history_output
                history_output = content

            processing_task = asyncio.create_task(client.process_query(
                query=message,
                summary_callback=update_summary,
                status_callback=update_status,
                history_callback=update_history,
            ))
            
            while not processing_task.done():
                yield (
                    "",  # Clear input
                    history_output if history_output else history_display,
                    status_output,
                    summary_output
                )
                await asyncio.sleep(0.1)

            await processing_task
            yield (
                "",  # Clear input
                history_output,
                status_output,
                summary_output
            )

    async def clear_chat():
        """Clear the conversation history"""
        success = await client.clear_conversation()
        if success:
            return (
                client.format_chat_history(),
                "Conversation cleared successfully!",
                ""
            )
        else:
            return (
                client.format_chat_history(),
                "Failed to clear conversation",
                ""
            )

    with gr.Blocks(title="AI Powered Chatbot", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🤖 AI Powered Chatbot")
        gr.Markdown("Have a conversation with an AI assistant that can access and cite relevant information.")
        
        with gr.Row():
            with gr.Column(scale=2):
                # Chat history display
                chat_history = gr.Markdown(
                    value=client.format_chat_history(),
                    label="Conversation History",
                    elem_id="chat_history",
                    container=True
                )
                
                # Input and controls
                with gr.Row():
                    message_input = gr.Textbox(
                        placeholder="Type your message here...",
                        label="Your Message",
                        scale=4,
                        lines=2
                    )
                    send_btn = gr.Button("📤 Send", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                    copy_conversation_btn = gr.Button("📋 Copy Conversation", variant="secondary")
                
                # Hidden textbox to hold current response for copying
                current_response_holder = gr.Textbox(visible=False, label="current_response_holder")
            
            with gr.Column(scale=1):
                status_output_md = gr.Markdown(
                    "### Status\nReady to chat..."
                )
        
        # Event handlers
        send_btn.click(
            fn=send_message,
            inputs=[message_input, chat_history, status_output_md],
            outputs=[message_input, chat_history, status_output_md, current_response_holder]
        )
        
        # Allow Enter key to send message
        message_input.submit(
            fn=send_message,
            inputs=[message_input, chat_history, status_output_md],
            outputs=[message_input, chat_history, status_output_md, current_response_holder]
        )
        
        clear_btn.click(
            fn=clear_chat,
            inputs=[],
            outputs=[chat_history, status_output_md, current_response_holder]
        )
        
        copy_conversation_btn.click(
            fn=None,
            inputs=chat_history,
            outputs=None,
            js="""
                (history) => {
                    if (history === null || history === undefined || history === "") {
                        window.alert("No conversation to copy.");
                        return;
                    }
                    navigator.clipboard.writeText(history);
                    window.alert("Conversation copied to clipboard!");
                }
            """
        )
        
        gr.Examples(
            examples=[
                ["What are the latest trends in artificial intelligence?"],
                ["Can you explain how transformer models work?"],
                ["What are the security implications of large language models?"],
                ["How is AI being used in healthcare today?"],
                ["Tell me about recent developments in computer vision."]
            ],
            inputs=[message_input],
        )
    
    return app

if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(server_name="localhost", server_port=int(os.environ.get('GRADIO_PORT', 7860)))