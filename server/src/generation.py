import json
import os
from openai import AsyncOpenAI
api_key = os.environ.get('OPENAI_API_KEY')
client = AsyncOpenAI(api_key=api_key)

generation_system_prompt = """You are an expert scientific research assistant. Your purpose is to synthesize key findings from multiple scientific articles into a single, cohesive summary that is tailored to a specific user's role and query. The summary should include correctly formatted citations and a complete reference list."""

generation_prompt_template = """
Your task is to generate a comprehensive summary based on the provided scientific articles.

User Role:
{user_role}

User Query:
{user_query}

--- Instructions ---
1. Synthesize findings: Do not just summarize each paper individually. Integrate the information from all provided articles to create a single, narrative-style summary that directly answers the user's query. Identify the main themes, supporting evidence, contrasting points, and overall conclusions.
2. Mandatory in-text citations: You MUST cite all information from the source articles using only numerical in-text citations in square brackets. Do not use any other format.
    - Correct Format: `[1]`, `[2]`, or `[1, 3]`
    - Incorrect Format: `(Source 1)`, `(Article 1)`, `[Source 1]`
    - Example: "SGLT2 inhibitors were shown to reduce cardiovascular events [1, 4]."
3. Generate a Reference List: After the summary, add a section titled "References". In this section, create a numbered list corresponding to the in-text citations. Each entry must include the source number, title and authors.
4. Target audience: The summary must be written for a {user_role}. Use appropriate terminology and level of detail for someone with this background.
5. Concise and focused: The final summary must not exceed 5,000 characters. Only include information that is directly relevant to the user's query.
6. Source material only: Base your summary exclusively on the content of the articles provided below. Do not introduce any external information or knowledge.

Provided Scientific Articles:

{formatted_papers_string}
"""

async def generate(papers, query: str, user_role: str):
    formatted_papers_list = []
    for i, paper in enumerate(papers):
        paper_string = f"""
--- Source [{i+1}] ---
Title: {paper['title']}
Authors: {', '.join(paper['authors'])}
Year: {paper['publication_year']}
Abstract: {paper['abstract']}
Content: {paper['body_text']}
"""
        formatted_papers_list.append(paper_string)

    formatted_papers_string = "\n".join(formatted_papers_list)

    final_prompt = generation_prompt_template.format(
        user_role=user_role,
        user_query=query,
        formatted_papers_string=formatted_papers_string
    )

    response = await client.responses.create(
        model=os.environ.get('OPENAI_MODEL'),
        instructions=generation_system_prompt,
        input=[
            {
                "role": "user",
                "content": final_prompt,
            },
        ],
        stream=True,
    )

    async for event in response:
        if 'response.output_text.delta' == event.type:
            yield event.delta

evaluation_prompt_template = """
You are an expert AI assistant acting as a meticulous evaluator of summaries generated from scientific papers.
Your task is to evaluate the provided 'Generated Summary' based on a 'User Query', 'User Role', and the 'Source Documents'.

Evaluate the summary based on the following four criteria, providing a score from 1 to 5 (1=Poor, 5=Excellent) and a concise justification for each.
Your final output MUST be a valid JSON object.

## Evaluation Criteria:
1.  **Faithfulness**: Does the summary only contain information that is directly supported by the source documents? It should not invent, distort, or misrepresent information.
2.  **Relevancy**: Is the summary directly relevant to the 'User Query' and tailored to the 'User Role'? (e.g., an expert requires more technical depth than a student).
3.  **Information Coverage**: Does the summary cover the most important and relevant points from the source documents needed to answer the query? Does it miss any critical information?
4.  **Clarity**: Is the summary well-written, clear, and easy to understand for the target 'User Role'? Is the language concise and the structure logical?

## Input Data:
### User Query:
{query}

### User Role:
{user_role}

### Source Documents:
{context}

### Generated Summary:
{summary}

## Required Output Format:
Provide your evaluation as a single JSON object with keys "faithfulness", "relevancy", "coverage", and "clarity". Each key should map to an object containing "score" and "justification".

Example:
{{
  "faithfulness": {{
    "score": 5,
    "justification": "The summary is fully supported by the provided abstracts and contains no new or contradictory information."
  }},
  "relevancy": {{ ... }},
  "coverage": {{ ... }},
  "clarity": {{ ... }}
}}
"""

async def evaluate(query, user_role, summary, source_papers):
    context = ""
    for i, paper in enumerate(source_papers):
        context += f"--- Source Document {i+1}: {paper.get('title', 'N/A')} ---\n"
        context += f"Authors: {', '.join(paper.get('authors', ['N/A']))}\n"
        context += f"Abstract: {paper.get('abstract', 'N/A')}\n\n"
    prompt = evaluation_prompt_template.format(
        query=query,
        user_role=user_role,
        context=context,
        summary=summary,
    )
    response = await client.chat.completions.create(
        model=os.environ.get('OPENAI_MODEL'),
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        response_format={ "type": "json_object" }
    )
    feedback = json.loads(response.choices[0].message.content)
    return feedback
