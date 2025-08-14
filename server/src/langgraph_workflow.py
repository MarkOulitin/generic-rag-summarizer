import json
import os
import retrival
import time
import rerenking
import generation
import answer_evaluator
import traceback
from typing import List, Dict, Any, TypedDict, Optional, Callable, Awaitable
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from logger import logger
from dotenv import load_dotenv

load_dotenv()

class WorkflowState(TypedDict):
    """State that gets passed between workflow nodes"""
    query: str
    original_query: str  # Track the original user query
    
    # Retrieval stage
    retrieved_chunks: List[Dict[str, Any]]
    
    # Reranking stage  
    reranked_chunks: List[Dict[str, Any]]
    
    # Generation stage
    generated_summary: str
    formatted_sources: List[Dict[str, Any]]
    
    # Agentic behavior tracking
    iteration_count: int
    max_iterations: int
    is_answer_adequate: bool
    evaluation_details: Dict[str, Any]
    query_history: List[str]  # Track query refinements
    
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
    max_iterations: int = 3  # Maximum number of query refinement iterations

class RAGWorkflow:
    """LangGraph-based RAG workflow implementation"""
    
    def __init__(self, config: Optional[WorkflowConfig] = None):
        self.config = config or WorkflowConfig()
        self.config.retrieval_top_k = int(os.getenv('RETRIVAL_TOP_K', self.config.retrieval_top_k))
        self.config.rerank_top_k = int(os.getenv('RERANK_TOP_K', self.config.rerank_top_k))
        self.config.max_iterations = int(os.getenv('MAX_ITERATIONS', self.config.max_iterations))
        
        # Store the callback separately to avoid serialization issues
        self._generate_callback = None
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with agentic behavior"""
        workflow = StateGraph(WorkflowState)
        
        # Define workflow nodes
        workflow.add_node("retrieve", self._retrieve_chunks)
        workflow.add_node("rerank", self._rerank_chunks)
        workflow.add_node("generate", self._generate_summary)
        workflow.add_node("evaluate", self._evaluate_answer)
        workflow.add_node("rewrite_query", self._rewrite_query)
        
        # Define workflow edges with conditional routing
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "generate")
        workflow.add_edge("generate", "evaluate")
        
        # Conditional routing after evaluation
        workflow.add_conditional_edges(
            "evaluate",
            self._should_continue_or_end,
            {
                "continue": "rewrite_query",
                "end": END
            }
        )
        
        # After rewriting query, go back to retrieval
        workflow.add_edge("rewrite_query", "retrieve")
        
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
                query, chunks, self._generate_callback
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
    
    async def _evaluate_answer(self, state: WorkflowState) -> WorkflowState:
        """Evaluate if the generated answer adequately addresses the query"""
        start_time = time.time()
        
        try:
            query = state['original_query']  # Use original query for evaluation
            generated_summary = state['generated_summary']
            
            logger.info(f"Evaluating answer adequacy for iteration {state['iteration_count']}")
            
            is_adequate, evaluation_details = await answer_evaluator.evaluate_answer(query, generated_summary)
            elapsed_time = time.time() - start_time
            
            logger.info(f"Answer evaluation: adequate={is_adequate}, confidence={evaluation_details.get('confidence', 0):.2f}")
            
            # Update state
            state['is_answer_adequate'] = is_adequate
            state['evaluation_details'] = evaluation_details
            state['current_stage'] = 'evaluated'
            state['stage_times']['evaluation'] = elapsed_time
            
            return state
            
        except Exception as e:
            logger.error(f"Error in evaluate_answer: {e}")
            # Default to adequate to prevent infinite loops on evaluation errors
            state['is_answer_adequate'] = True
            state['evaluation_details'] = {"error": str(e)}
            return state
    
    async def _rewrite_query(self, state: WorkflowState) -> WorkflowState:
        """Rewrite the query to improve retrieval results"""
        start_time = time.time()
        
        try:
            original_query = state['original_query']
            current_query = state['query']
            generated_summary = state['generated_summary']
            evaluation_details = state['evaluation_details']
            
            logger.info(f"Rewriting query for iteration {state['iteration_count'] + 1}")
            
            new_query = await answer_evaluator.rewrite_query(
                current_query, generated_summary, evaluation_details
            )
            elapsed_time = time.time() - start_time
            
            logger.info(f"Query rewritten: '{current_query}' -> '{new_query}'")
            
            # Update state
            state['query'] = new_query
            state['query_history'].append(new_query)
            state['iteration_count'] += 1
            state['current_stage'] = 'query_rewritten'
            state['stage_times']['query_rewriting'] = elapsed_time
            
            # Reset retrieval and generation state for next iteration
            state['retrieved_chunks'] = []
            state['reranked_chunks'] = []
            state['generated_summary'] = ""
            state['formatted_sources'] = []
            
            return state
            
        except Exception as e:
            logger.error(f"Error in rewrite_query: {e}")
            state['error'] = f"Query rewriting failed: {str(e)}"
            return state
    
    def _should_continue_or_end(self, state: WorkflowState) -> str:
        """Determine whether to continue with query refinement or end the workflow"""
        
        # End if answer is adequate
        if state['is_answer_adequate']:
            logger.info("Answer is adequate, ending workflow")
            return "end"
        
        # End if maximum iterations reached
        if state['iteration_count'] >= state['max_iterations']:
            logger.info(f"Maximum iterations ({state['max_iterations']}) reached, ending workflow")
            return "end"
        
        # End if there's an error
        if state.get('error'):
            logger.info("Error detected, ending workflow")
            return "end"
        
        # Continue with query refinement
        logger.info(f"Answer inadequate, continuing to iteration {state['iteration_count'] + 1}")
        return "continue"
        
    async def run_workflow(self, query: str, generate_callback=None) -> Dict[str, Any]:
        try:
            # Store the callback in the instance to avoid serialization issues
            self._generate_callback = generate_callback
            
            # Initialize state
            initial_state = WorkflowState(
                query=query,
                original_query=query,  # Keep track of the original query
                retrieved_chunks=[],
                reranked_chunks=[],
                generated_summary="",
                formatted_sources=[],
                iteration_count=0,
                max_iterations=self.config.max_iterations,
                is_answer_adequate=False,
                evaluation_details={},
                query_history=[query],
                current_stage="starting",
                stage_times={},
                error=None
            )
            
            # Run the workflow
            config = {"configurable": {"thread_id": "rag_workflow"}}
            final_state = await self.workflow.ainvoke(initial_state, config=config)
            
            # Return results
            return {
                "summary": final_state.get("generated_summary", ""),
                "sources": final_state.get("formatted_sources", []),
                "stage_times": final_state.get("stage_times", {}),
                "error": final_state.get("error"),
                "metadata": {
                    "retrieval_count": len(final_state.get("retrieved_chunks", [])),
                    "rerank_count": len(final_state.get("reranked_chunks", [])),
                    "iterations": final_state.get("iteration_count", 0),
                    "is_answer_adequate": final_state.get("is_answer_adequate", False),
                    "evaluation_details": final_state.get("evaluation_details", {}),
                    "query_history": final_state.get("query_history", []),
                    "original_query": final_state.get("original_query", ""),
                    "final_query": final_state.get("query", "")
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

async def process_rag_query(query: str, generate_callback=None) -> Dict[str, Any]:
    """Process a RAG query using the LangGraph workflow"""
    workflow = get_workflow()
    return await workflow.run_workflow(query, generate_callback)
