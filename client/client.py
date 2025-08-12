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
        self.evaluation_result = None
        
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
            'evaluation': 'Evaluating result'
        }
        
        for stage, name in stage_names.items():
            if stage in self.stage_times:
                time_str = self.format_time(self.stage_times[stage])
                report += f"✅ **{name}** - {time_str}\n\n"
        
        return report
    
    def create_evaluation_report(self, evaluation: Dict[str, Any]) -> str:
        """Create a formatted evaluation report"""
        report = "## Evaluation Results\n\n"
        
        metrics = ['faithfulness', 'relevancy', 'coverage', 'clarity']
        
        for metric in metrics:
            if metric in evaluation:
                score = evaluation[metric]['score']
                justification = evaluation[metric]['justification']
                report += f"**{metric.capitalize()}**: {score}/5\n"
                report += f"*{justification}*\n\n"
        
        return report
        
    async def process_query(self, query: str, user_role: str,
                            summary_callback=None,
                            status_callback=None,
                            evaluation_callback=None):
        """Process the query and handle server-sent events"""

        self.current_summary = ""
        self.stage_times = {}
        self.evaluation_result = None

        # Clear previous results
        if status_callback:
            status_callback("Connecting to server...")
        if summary_callback:
            summary_callback("")
        if evaluation_callback:
            evaluation_callback("")

        try:
            uri = f"ws://{os.environ.get('SERVER_HOST', 'localhost')}:{os.environ.get('SERVER_PORT', 9090)}/"
            logger.info(f"Connecting to {uri}")
            async with websockets.connect(uri) as websocket:
                logger.info(f"Connected to {uri}")
                message = {"query": query, "user_role": user_role}

                await websocket.send(json.dumps(message))

                try:
                    while True:
                        response_json = await websocket.recv()
                        response_data = json.loads(response_json)
                        event_type = response_data.get('event')
                        data = response_data.get('data')

                        if event_type == 'extracting_candidates':
                            logger.info(f"Received {response_json}")
                            logger.info(f"If status_callback: {bool(status_callback)}, {self.stage_times}")
                            if status_callback:
                                status_callback(self.create_status_report() + "\n⏳ Extracting candidates...")

                        elif event_type == 'extracted_candidates':
                            logger.info(f"Received {response_json}")
                            self.stage_times['extraction'] = response_data.get("time")
                            logger.info(f"If status_callback: {bool(status_callback)}, {self.stage_times}")
                            if status_callback:
                                status_callback(self.create_status_report())
                        
                        elif event_type == 'reranking_candidates':
                            logger.info(f"Received {response_json}")
                            logger.info(f"If status_callback: {bool(status_callback)}, {self.stage_times}")
                            if status_callback:
                                status_callback(self.create_status_report() + "\n⏳ Reranking candidates...")

                        elif event_type == 'reranked_candidates':
                            logger.info(f"Received {response_json}")
                            logger.info(f"If status_callback: {bool(status_callback)}, {self.stage_times}")
                            self.stage_times['reranking'] = response_data.get("time")
                            if status_callback:
                                status_callback(self.create_status_report())

                        elif event_type == 'generating':
                            if 'generation' not in self.stage_times:
                                if status_callback:
                                    status_callback(self.create_status_report() + "\n⏳ Generating summary...")
                                self.stage_times['generation'] = time.time()
                                logger.info(f"If generation: {self.stage_times}")

                            if data and summary_callback:
                                self.current_summary += data
                                summary_callback(self.current_summary)

                        elif event_type == 'evaluating':
                            stage_time = time.time()
                            self.stage_times['generation'] = stage_time - self.stage_times['generation']
                            if status_callback:
                                status_callback(self.create_status_report() + "\n⏳ Evaluating result...")
                            logger.info(f"If evaluate: {self.stage_times}")
                            self.stage_times['evaluation'] = stage_time
                        
                        elif event_type == 'evaluation_completed':
                            self.stage_times['evaluation'] = time.time() - self.stage_times['evaluation']
                            if data and evaluation_callback:
                                self.evaluation_result = data
                                evaluation_callback(self.create_evaluation_report(self.evaluation_result))
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

client = RAGClient()
def create_gradio_app():
    """Create the Gradio interface"""

    async def process_request(query, user_role):
        """Handle the request processing as an async generator"""
        
        summary_output = ""
        status_output = ""
        evaluation_output = ""
        
        def update_summary(content):
            nonlocal summary_output
            summary_output = content
        
        def update_status(content):
            nonlocal status_output
            status_output = content
            
        def update_evaluation(content):
            nonlocal evaluation_output
            evaluation_output = content

        processing_task = asyncio.create_task(client.process_query(
            query=query,
            user_role=user_role,
            summary_callback=update_summary,
            status_callback=update_status,
            evaluation_callback=update_evaluation
        ))
        
        while not processing_task.done():
            yield (
                summary_output,
                status_output,
                evaluation_output,
                gr.update(visible=bool(client.current_summary)),
                summary_output 
            )
            await asyncio.sleep(0.1)

        await processing_task
        yield (
            summary_output,
            status_output,
            evaluation_output,
            gr.update(visible=bool(client.current_summary)),
            summary_output
        )

    with gr.Blocks(title="RAG Scientific Paper Summarizer", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🔬 RAG Scientific Paper Summarizer")
        gr.Markdown("Enter your query and role to get AI-powered summaries of scientific papers.")
        
        with gr.Row():
            with gr.Column(scale=2):
                user_role = gr.Textbox(
                    label="Your Role",
                    placeholder="e.g., PhD student, researcher, industry professional",
                    value="biomedical researcher"
                )
                
                query = gr.Textbox(
                    label="Your Query",
                    placeholder="e.g., What are the latest advances in transformer models?",
                    value="What are the latest treatment options for juvenile arthritis?",
                    lines=3
                )
                
                submit_btn = gr.Button("🚀 Generate Summary", variant="primary", size="lg")
                
                summary_output_md = gr.Markdown(
                    label="Generated Summary",
                    value=""
                )
                
                summary_holder = gr.Textbox(visible=False, label="summary_holder")
                
                copy_btn = gr.Button(
                    "📋 Copy to clipboard",
                    visible=False,
                )
            
            with gr.Column(scale=1):
                status_output_md = gr.Markdown(
                    "### Processing Status\nReady to process your query..."
                )
                
                evaluation_output_md = gr.Markdown(
                    "### Evaluation Results\nEvaluation will appear here after processing..."
                )
        
        submit_btn.click(
            fn=process_request,
            inputs=[query, user_role],
            outputs=[summary_output_md, status_output_md, evaluation_output_md, copy_btn, summary_holder]
        )
        
        copy_btn.click(
            fn=None,                  
            inputs=summary_holder,
            outputs=None,             
            js="""
                (text) => {
                    if (text === null || text === undefined || text === "") {
                        window.alert("Error: Nothing to copy. The summary is empty.");
                        return;
                    }
                    navigator.clipboard.writeText(text);
                    window.alert("Summary copied to clipboard!");
                }
            """
        )
        
        gr.Examples(
            examples=[
                ["What are the key innovations in transformer architectures?", "AI researcher"],
                ["How has natural language processing evolved in recent years?", "PhD student"],
                ["What are the practical applications of large language models?", "industry professional"],
                ["What are the latest developments in retrieval-augmented generation?", "research scientist"]
            ],
            inputs=[query, user_role],
        )
    
    return app

if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(server_name="localhost", server_port=int(os.environ.get('GRADIO_PORT', 7860)))