import torch
import numpy as np
import xml.etree.ElementTree as ET
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import check_gpu_memory, print_model_vram
from logger import logger
logger.info("Loading model...")
check_gpu_memory()

reranker_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
reranker_model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3', device_map='cuda')
reranker_model.eval()

logger.info("Model loaded successfully!")
check_gpu_memory()

print_model_vram(reranker_model)

def rerank(query, papers, top_k):
    template = "Title: {title}\n\nAbstract: {abstract}"
    scores = []
    with torch.no_grad():
        for paper in papers:
            paper_info = template.format(title=paper['title'], abstract=paper['abstract'])
            pair = [query, paper_info]
            inputs = reranker_tokenizer([pair], padding=True, truncation=True, return_tensors='pt', max_length=512)
            inputs = {k: v.to(reranker_model.device) for k, v in inputs.items()}
            score = reranker_model(**inputs, return_dict=True).logits.view(-1, ).float()
            scores.append(score)
    scores = torch.stack(scores).squeeze().cpu().numpy()
    top_k_indexes = np.argsort(scores)[-top_k:]

    papers_for_summary = [papers[i] for i in top_k_indexes]
    return papers_for_summary

def get_papers_data(papers_meta):
    papers = []
    for paper_metadata in papers_meta:
        file_path = paper_metadata['file_path']
        paper = parse_full_pubmed_xml(file_path)
        if paper is not None:
            papers.append(paper)
    return papers

def parse_full_pubmed_xml(file_path):
    """
    Parses a PubMed XML file to extract title, abstract, authors, 
    publication year, and the full body text.
    
    Args:
        file_path (str): The path to the XML file.

    Returns:
        dict: A dictionary containing the extracted data, or None on failure.
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        # --- Extract Title ---
        title_node = root.find('.//article-title')
        title = ''.join(title_node.itertext()).strip() if title_node is not None else "No Title"

        # Extract Abstract
        abstract_node = root.find('.//abstract')
        if abstract_node is not None:
            # Join all paragraphs and text within the abstract
            abstract = '\n'.join([''.join(p.itertext()).strip() for p in abstract_node.findall('.//p')])
        else:
            abstract = "No Abstract"
        
        # --- Extract Authors ---
        authors = []
        author_nodes = root.findall('.//contrib-group/contrib[@contrib-type="author"]')
        for author_node in author_nodes:
            surname = author_node.find('.//surname')
            given_names = author_node.find('.//given-names')
            if surname is not None and given_names is not None:
                authors.append(f"{given_names.text} {surname.text}")
        
        # --- Extract Publication Year ---
        pub_date_node = root.find('.//pub-date[@pub-type="ppub"]') # Prioritize print publication date
        if pub_date_node is None:
            pub_date_node = root.find('.//pub-date[@pub-type="epub"]') # Fallback to electronic
        
        pub_year = pub_date_node.find('year').text if pub_date_node is not None and pub_date_node.find('year') is not None else "No Year"

        # --- Extract Body Text ---
        body_node = root.find('.//body')
        if body_node is not None:
            # Find all paragraph tags within the body
            body_paragraphs = body_node.findall('.//p')
            body_text = '\n\n'.join([''.join(p.itertext()).strip() for p in body_paragraphs])
        else:
            body_text = "No Body Text"
            
        return {
            'title': title,
            'abstract': abstract,
            'authors': authors,
            'publication_year': pub_year,
            'body_text': body_text
        }

    except ET.ParseError:
        logger.warn(f"Warning: Could not parse {file_path}")
        return None
