import os
from openai import AsyncOpenAI
from typing import List, Dict, Any
from logger import logger

api_key = os.environ.get('OPENAI_API_KEY')
client = AsyncOpenAI(api_key=api_key)
topic_domain = os.environ['TOPICS_DOMAIN']

recommendation_system_prompt = f"""You are an AI assistant expert in the topic "{topic_domain}". Your task is to generate relevant follow-up questions based on a conversation history. These questions should help users explore the topic deeper or from different angles."""

recommendation_prompt_template = """
Based on the following conversation history, generate 3-5 relevant follow-up questions that would help the user explore the topic further. The questions should be:

1. Directly related to the conversation context
2. Progressively deeper or exploring different aspects of the topic
3. Naturally flowing from what has been discussed
4. Engaging and thought-provoking
5. Specific enough to be actionable but broad enough to be interesting

Conversation History:
{conversation_history}

Generate ONLY the questions, one per line, without any numbering, bullets, or additional text. Each question should be on its own line.

Example format:
What are the security implications of this approach?
How does this compare to traditional methods?
What are the latest developments in this area?
"""

async def generate_recommendations(conversation_history: List[Dict[str, Any]]) -> List[str]:
    """Generate follow-up question recommendations based on conversation history"""
    
    if len(conversation_history) == 0:
        # Return default questions for new conversations
        return [
            f"What are the current trends in {topic_domain}?",
            f"What are the key challenges in {topic_domain}?",
            f"How has {topic_domain} evolved recently?",
            f"What are the practical applications of {topic_domain}?",
            f"What should beginners know about {topic_domain}?"
        ]
    
    try:
        # Format conversation history for the prompt
        formatted_history = ""
        for msg in conversation_history[-6:]:  # Use last 6 messages for context
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted_history += f"{role}: {msg['content']}\n\n"
        
        recommendation_prompt = recommendation_prompt_template.format(
            conversation_history=formatted_history
        )
        # Generate recommendations using OpenAI
        response = await client.chat.completions.create(
            model=os.environ.get('OPENAI_MODEL'),
            messages=[
                {
                    "role": "system", 
                    "content": recommendation_system_prompt
                },
                {
                    "role": "user",
                    "content": recommendation_prompt
                }
            ]
        )
        
        # Parse the response into individual questions
        recommendations_text = response.choices[0].message.content.strip()
        recommendations = [
            q.strip() for q in recommendations_text.split('\n') 
            if q.strip() and not q.strip().startswith('#')
        ]
        
        # Ensure we have 3-5 recommendations
        if len(recommendations) < 3:
            # Add some generic follow-ups if we don't have enough
            generic_questions = [
                "Can you explain this in more detail?",
                "What are the implications of this?",
                "How does this relate to current practices?",
                "What are the potential challenges?",
                "What should I know next about this topic?"
            ]
            recommendations.extend(generic_questions[:5-len(recommendations)])
        elif len(recommendations) > 5:
            recommendations = recommendations[:5]
            
        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        # Return fallback questions
        return [
            "Can you provide more details about this?",
            "What are the practical applications?",
            "How does this compare to alternatives?",
            "What are the current challenges?",
            "What should I explore next?"
        ]
