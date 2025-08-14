import json
import os
import retrival
import time
import rerenking
import generation
import traceback
from typing import List, Dict, Any, TypedDict, Optional, Callable, Awaitable
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from logger import logger
from dotenv import load_dotenv

load_dotenv()

class ConversationMemory:
    """Manages conversation history for different conversation IDs"""
    
    def __init__(self):
        self.conversations = {}  # {conversation_id: List[Dict]}
        
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a specific conversation ID"""
        return self.conversations.get(conversation_id, [])
    
    def add_message(self, conversation_id: str, role: str, content: str):
        """Add a message to the conversation history"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        self.conversations[conversation_id].append({
            "role": role,
            "content": content
        })
    
    def clear_conversation(self, conversation_id: str):
        """Clear conversation history for a specific conversation ID"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]

class WorkflowState(TypedDict):
    """State that gets passed between workflow nodes"""
    query: str
    conversation_id: str
    conversation_history: List[Dict[str, Any]]  # List of {"role": "user/assistant", "content": str}
    
    # Retrieval stage
    retrieved_chunks: List[Dict[str, Any]]
    
    # Reranking stage  
    reranked_chunks: List[Dict[str, Any]]
    
    # Generation stage
    generated_summary: str
    formatted_sources: List[Dict[str, Any]]
    
    # Progress tracking
    current_stage: str
    stage_times: Dict[str, float]
    error: Optional[str] 

@dataclass
class WorkflowConfig:
    """Configuration for the RAG workflow"""
    retrieval_top_k: int = 50
    rerank_top_k: int = 10
    max_summary_length: int = 5000

class RAGWorkflow:
    """LangGraph-based RAG workflow implementation"""
    
    def __init__(self, config: Optional[WorkflowConfig] = None):
        self.config = config or WorkflowConfig()
        self.config.retrieval_top_k = int(os.getenv('RETRIVAL_TOP_K', self.config.retrieval_top_k))
        self.config.rerank_top_k = int(os.getenv('RERANK_TOP_K', self.config.rerank_top_k))
        
        # Store the callback separately to avoid serialization issues
        self._generate_callback = None
        
        # Initialize conversation memory
        self.conversation_memory = ConversationMemory()
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
    def clear_conversation(self, conversation_id: str):
        """Clear conversation history for a specific conversation ID"""
        self.conversation_memory.clear_conversation(conversation_id)
        
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(WorkflowState)
        
        # Define workflow nodes
        workflow.add_node("retrieve", self._retrieve_chunks)
        workflow.add_node("rerank", self._rerank_chunks)
        workflow.add_node("generate", self._generate_summary)
        
        # Define workflow edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "generate")
        workflow.add_edge("generate", END)
        
        # Add memory checkpointing
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    async def _retrieve_chunks(self, state: WorkflowState) -> WorkflowState:
        start_time = time.time()
        
        try:
            logger.info(f"Retrieving chunks for query: {state['query'][:50]}...")
            retrieved_chunks = retrival.retrieve(state['query'], self.config.retrieval_top_k)
            elapsed_time = time.time() - start_time
    
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks in {elapsed_time:.2f}s")
            
            # Update state
            state['retrieved_chunks'] = retrieved_chunks
            state['current_stage'] = 'retrieved'
            state['stage_times']['retrieval'] = elapsed_time
            return state
        except Exception as e:
            logger.error(f"Error in retrieve_chunks: {e}")
            state['error'] = f"Retrieval failed: {str(e)}"
            return state
    
    async def _rerank_chunks(self, state: WorkflowState) -> WorkflowState:
        start_time = time.time()
        
        try:
            query = state['query']
            chunks = state['retrieved_chunks']
            
            logger.info(f"Reranking {len(chunks)} chunks...")
            
            if not chunks:
                logger.warning("No chunks to rerank")
                state['reranked_chunks'] = []
                return state
            
            reranked_chunks = rerenking.rerank(query, chunks, self.config.rerank_top_k)
            elapsed_time = time.time() - start_time
            
            logger.info(f"Reranked to top {len(reranked_chunks)} chunks in {elapsed_time:.2f}s")
            
            # Update state
            state['reranked_chunks'] = reranked_chunks
            state['current_stage'] = 'reranked'
            state['stage_times']['reranking'] = elapsed_time
            
            return state
            
        except Exception as e:
            logger.error(f"Error in rerank_chunks: {e}")
            state['error'] = f"Reranking failed: {str(e)}"
            return state
    
    async def _generate_summary(self, state: WorkflowState) -> WorkflowState:
        start_time = time.time()
        
        try:
            query = state['query']
            chunks = state['reranked_chunks']
            
            logger.info(f"Generating summary from {len(chunks)} chunks...")
            
            if not chunks:
                logger.warning("No chunks available for summary generation")
                state['generated_summary'] = "No relevant information found for your query."
                state['formatted_sources'] = []
                return state
            
            generated_summary, formatted_sources = await generation.generate(
                query, chunks, state['conversation_history'], self._generate_callback
            )
            elapsed_time = time.time() - start_time
            
            logger.info(f"Generated summary ({len(generated_summary)} chars) in {elapsed_time:.2f}s")
            
            # Update state
            state['generated_summary'] = generated_summary
            state['formatted_sources'] = formatted_sources
            state['current_stage'] = 'generated'
            state['stage_times']['generation'] = elapsed_time
            
            return state
            
        except Exception as e:
            logger.error(f"Error in generate_summary: {e}")
            state['error'] = f"Generation failed: {str(e)}"
            return state
        
    async def run_workflow(self, query: str, conversation_id: str = "default", generate_callback=None) -> Dict[str, Any]:
        try:
            # Store the callback in the instance to avoid serialization issues
            self._generate_callback = generate_callback
            
            # Get conversation history
            conversation_history = self.conversation_memory.get_conversation_history(conversation_id)
            
            # Add user query to conversation history
            self.conversation_memory.add_message(conversation_id, "user", query)
            
            # Initialize state
            initial_state = WorkflowState(
                query=query,
                conversation_id=conversation_id,
                conversation_history=conversation_history,
                retrieved_chunks=[],
                reranked_chunks=[],
                generated_summary="",
                formatted_sources=[],
                current_stage="starting",
                stage_times={},
                error=None
            )
            
            # Run the workflow
            config = {"configurable": {"thread_id": conversation_id}}
            final_state = await self.workflow.ainvoke(initial_state, config=config)
            
            # Add assistant response to conversation history
            if final_state.get("generated_summary") and not final_state.get("error"):
                self.conversation_memory.add_message(conversation_id, "assistant", final_state["generated_summary"])
            
            # Return results
            return {
                "summary": final_state.get("generated_summary", ""),
                "sources": final_state.get("formatted_sources", []),
                "stage_times": final_state.get("stage_times", {}),
                "error": final_state.get("error"),
                "conversation_id": conversation_id,
                "metadata": {
                    "retrieval_count": len(final_state.get("retrieved_chunks", [])),
                    "rerank_count": len(final_state.get("reranked_chunks", [])),
                }
            }
            
        except Exception as e:
            trace = traceback.format_exc()
            logger.error(f"Workflow execution failed, Error: {e}\n{trace}")
            return {
                "summary": "",
                "sources": [],
                "stage_times": {},
                "error": str(e),
                "metadata": {}
            }

workflow_instance = None

def get_workflow() -> RAGWorkflow:
    """Get or create the global workflow instance"""
    global workflow_instance
    if workflow_instance is None:
        workflow_instance = RAGWorkflow()
    return workflow_instance

async def process_rag_query(query: str, conversation_id: str = "default", generate_callback=None) -> Dict[str, Any]:
    """Process a RAG query using the LangGraph workflow"""
    workflow = get_workflow()
    return await workflow.run_workflow(query, conversation_id, generate_callback)
