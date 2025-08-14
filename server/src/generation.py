import json
import os
from openai import AsyncOpenAI
api_key = os.environ.get('OPENAI_API_KEY')
client = AsyncOpenAI(api_key=api_key)
topic_domain = os.environ['TOPICS_DOMAIN']
generation_system_prompt = f"""You are an assistant expert in the topic \"{topic_domain}\". Your purpose is to synthesize a single, cohesive and comprehensive summary with proper citations. The summary should include correctly formatted citations and a complete reference list."""

generation_prompt_template = """
Your task is to generate a comprehensive summary based on the provided chunks of information.

User Query:
{user_query}

--- Instructions ---
1. Synthesize findings: Integrate information from all provided chunks to create a cohesive summary that directly answers the user's query.
2. Mandatory in-text citations: You MUST cite all information from the source articles using only numerical in-text citations in square brackets. Do not use any other format.
    - Correct Format: `[1]`, `[2]`, or `[1, 3]`
    - Incorrect Format: `(Source 1)`, `(Article 1)`, `[Source 1]`
    - Example: "SGLT2 inhibitors were shown to reduce cardiovascular events [1, 4]."
3. Generate a Reference List: After the summary, add a section titled "References". In this section, create a numbered list corresponding to the in-text citations. Each entry must include the source number and URL.
5. Concise and focused: Only include information that is directly relevant to the user's query.
6. Source material only: Base your summary exclusively on the content of provided below. Do not introduce any external information or knowledge.

Document Chunks:
{formatted_chunks_string}
"""

async def generate(query, chunks, stream_callback=None):
    formatted_sources = []
    formatted_chunks_list = []
    
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk['metadata']
        
        source_info = {
            'index': i,
            'title': metadata.get('title', 'No Title'),
            'source_file': metadata.get('source_file', 'Unknown'),
            'source': metadata.get('source', 'Unknown'),
            'url': metadata.get('original_url', ''),
            'chunk_id': metadata.get('chunk_id', ''),
            'content': chunk.get('content', '')
        }
        formatted_sources.append(source_info)
        
        content = chunk.get('content', 'No content available')
        chunk_text = f"""
--- Source [{i+1}] ---
Title: {source_info['title']}
URL: {source_info['url'] if source_info['url'] else 'N/A'}
Content: {content}
"""
        formatted_chunks_list.append(chunk_text)
    
    formatted_chunks_string = '\n'.join(formatted_chunks_list)
    generation_prompt = generation_prompt_template.format(
        user_query=query,
        formatted_chunks_string=formatted_chunks_string
    )
    # Choose streaming or non-streaming based on callback presence
    if stream_callback:
        # Streaming mode
        response = await client.chat.completions.create(
            model=os.environ.get('OPENAI_MODEL'),
            messages=[
                {
                    "role": "system",
                    "content": generation_system_prompt
                },
                {
                    "role": "user",
                    "content": generation_prompt
                }
            ],
            stream=True
        )
        
        generated_summary = ""
        async for chunk in response:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                generated_summary += content
                # Call the callback with the text chunk
                await stream_callback(content)
        
        return generated_summary, formatted_sources
    else:
        # Non-streaming mode (original behavior)
        response = await client.chat.completions.create(
            model=os.environ.get('OPENAI_MODEL'),
            messages=[
                {
                    "role": "system",
                    "content": generation_system_prompt
                },
                {
                    "role": "user",
                    "content": generation_prompt
                }
            ]
        )
        
        generated_summary = response.choices[0].message.content
        return generated_summary, formatted_sources
