import os
from openai import AsyncOpenAI
from typing import Dict, Any, Tuple
from logger import logger

api_key = os.environ.get('OPENAI_API_KEY')
client = AsyncOpenAI(api_key=api_key)

evaluation_system_prompt = """You are an expert evaluator that determines if a generated answer adequately addresses a user's query. Your role is to assess answer quality and completeness objectively."""

evaluation_prompt_template = """
Evaluate whether the provided answer adequately addresses the user's query.

User Query:
{user_query}

Generated Answer:
{generated_answer}

--- Evaluation Criteria ---
1. Completeness: Does the answer fully address all aspects of the query?
2. Relevance: Is the answer directly relevant to what was asked?
3. Informativeness: Does the answer provide sufficient detail and insight?
4. Accuracy: Based on the content, does the answer appear factually sound?

--- Instructions ---
- Consider the query's complexity and scope
- An answer should be marked as adequate if it reasonably addresses the main question, even if not exhaustively comprehensive
- Be objective and focus on whether a typical user would find their question answered
- If the answer says "no relevant information found" or is clearly insufficient, mark as inadequate

Respond with a JSON object containing:
{{
    "is_adequate": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of your assessment",
    "missing_aspects": ["list", "of", "missing", "elements"] // only if is_adequate is false
}}
"""

query_rewriter_system_prompt = """You are an expert query rewriter. Your goal is to rephrase user queries to improve information retrieval when the initial query didn't yield adequate results."""

query_rewriter_prompt_template = """
The original query did not retrieve adequate information. Rewrite the query to improve retrieval results.

Original Query:
{original_query}

Previous Answer (inadequate):
{previous_answer}

Evaluation Feedback:
{evaluation_feedback}

--- Rewriting Guidelines ---
1. Expand scope: Add related terms, synonyms, or broader concepts
2. Specify context: Add domain-specific terminology if missing
3. Break down complex queries: If multi-part, focus on the most important aspect
4. Add keywords: Include terms likely to appear in relevant documents
5. Maintain intent: Keep the core question unchanged

--- Instructions ---
- Generate 1-3 alternative query formulations
- Prioritize the most promising reformulation
- Keep queries concise but more specific than the original
- Consider different ways experts might discuss the topic

Respond with a JSON object containing:
{{
    "primary_query": "The best reformulated query",
    "reasoning": "Why this reformulation should work better",
    "alternatives": ["alternative 1", "alternative 2"] // optional additional formulations
}}
"""

async def evaluate_answer(user_query: str, generated_answer: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Evaluate if the generated answer adequately addresses the user's query.
    
    Returns:
        Tuple of (is_adequate: bool, evaluation_details: dict)
    """
    try:
        evaluation_prompt = evaluation_prompt_template.format(
            user_query=user_query,
            generated_answer=generated_answer
        )
        
        response = await client.chat.completions.create(
            model=os.environ.get('OPENAI_MODEL', 'gpt-4'),
            messages=[
                {"role": "system", "content": evaluation_system_prompt},
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0.1  # Low temperature for consistent evaluation
        )
        
        result_text = response.choices[0].message.content
        logger.info(f"Answer evaluation result: {result_text}")
        
        # Parse the JSON response
        import json
        evaluation_result = json.loads(result_text)
        
        is_adequate = evaluation_result.get('is_adequate', False)
        
        return is_adequate, evaluation_result
        
    except Exception as e:
        logger.error(f"Error in answer evaluation: {e}")
        # Default to adequate to avoid infinite loops on evaluation errors
        return True, {
            "is_adequate": True,
            "confidence": 0.0,
            "reasoning": f"Evaluation failed: {str(e)}",
            "missing_aspects": []
        }

async def rewrite_query(original_query: str, previous_answer: str, evaluation_feedback: Dict[str, Any]) -> str:
    """
    Rewrite the query to improve retrieval when the previous answer was inadequate.
    
    Returns:
        Rewritten query string
    """
    try:
        feedback_text = evaluation_feedback.get('reasoning', '') + ' Missing: ' + ', '.join(evaluation_feedback.get('missing_aspects', []))
        
        rewriter_prompt = query_rewriter_prompt_template.format(
            original_query=original_query,
            previous_answer=previous_answer,
            evaluation_feedback=feedback_text
        )
        
        response = await client.chat.completions.create(
            model=os.environ.get('OPENAI_MODEL', 'gpt-4'),
            messages=[
                {"role": "system", "content": query_rewriter_system_prompt},
                {"role": "user", "content": rewriter_prompt}
            ],
            temperature=0.3  # Slightly higher temperature for creative reformulation
        )
        
        result_text = response.choices[0].message.content
        logger.info(f"Query rewrite result: {result_text}")
        
        # Parse the JSON response
        import json
        rewrite_result = json.loads(result_text)
        
        new_query = rewrite_result.get('primary_query', original_query)
        logger.info(f"Rewritten query: '{original_query}' -> '{new_query}'")
        
        return new_query
        
    except Exception as e:
        logger.error(f"Error in query rewriting: {e}")
        # Return original query if rewriting fails
        return original_query
