import os
import time
import requests
import json
from pathlib import Path
from urllib.parse import urlparse
from typing import List, Dict, Set
import logging
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from duckduckgo_search import DDGS

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DocumentCollector:
    def __init__(self):
        self.topic = os.environ['TOPICS_DOMAIN']
        self.max_documents = int(os.getenv('MAX_DOCUMENTS', 2000))
        self.data_dir = Path('./data/collected_documents')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Track collected URLs to avoid duplicates
        self.collected_urls: Set[str] = set()
        self.documents_collected = 0
        
        
    def search_google_scholar(self, query: str, num_pages: int = 10) -> List[Dict]:
        results = []
        base_url = "https://scholar.google.com/scholar"
        
        # Better headers to appear more like a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        for page in range(num_pages):
            params = {
                'q': query,
                'start': page * 10,
                'hl': 'en'
            }
            
            try:
                response = requests.get(base_url, params=params, headers=headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                page_results = 0
                # Parse search results
                for result in soup.find_all('div', class_='gs_r'):
                    title_elem = result.find('h3', class_='gs_rt')
                    if title_elem:
                        link_elem = title_elem.find('a')
                        if link_elem and link_elem.get('href'):
                            url = link_elem['href']
                            title = link_elem.get_text()
                            
                            # Look for PDF links
                            pdf_link = result.find('a', string='[PDF]')
                            if pdf_link:
                                url = pdf_link['href']
                            
                            results.append({
                                'title': title.strip(),
                                'url': url,
                                'source': 'google_scholar'
                            })
                            page_results += 1
                
                logger.info(f"Found {page_results} results on Google Scholar page {page + 1}")
                
                time.sleep(20)
                
            except Exception as e:
                logger.error(f"Error searching Google Scholar page {page}: {e}")
                continue
                
        return results
    
    def search_arxiv(self, query: str, max_results: int = 500) -> List[Dict]:
        """Search arXiv for research papers"""
        results = []
        base_url = "http://export.arxiv.org/api/query"
        
        # ArXiv API allows max 2000 results per query
        batch_size = min(100, max_results)
        
        for start in range(0, max_results, batch_size):
            params = {
                'search_query': f'all:{query}',
                'start': start,
                'max_results': batch_size,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            try:
                response = requests.get(base_url, params=params, timeout=15)
                response.raise_for_status()
                
                # Parse XML response
                from xml.etree import ElementTree as ET
                root = ET.fromstring(response.content)
                
                for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                    title_elem = entry.find('{http://www.w3.org/2005/Atom}title')
                    id_elem = entry.find('{http://www.w3.org/2005/Atom}id')
                    
                    if title_elem is not None and id_elem is not None:
                        title = title_elem.text.strip()
                        arxiv_id = id_elem.text.split('/')[-1]
                        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                        
                        results.append({
                            'title': title,
                            'url': pdf_url,
                            'source': 'arxiv',
                            'arxiv_id': arxiv_id
                        })
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error searching arXiv batch starting at {start}: {e}")
                continue
                
        return results
    
    def search_web_general(self, query: str) -> List[Dict]:
        """General web search for documents using DuckDuckGo"""
        results = []
        
        # Search for different file types and content types
        search_queries = [
            f"{query} filetype:pdf",
            f"{query} filetype:md",
            f"{query} filetype:doc",
            f"{query} documentation",
            f"{query} tutorial",
            f"{query} guide",
            f"{query} research paper",
            f"{query} whitepaper"
        ]
        
        # Initialize DuckDuckGo search
        ddgs = DDGS()
        
        for search_query in search_queries:
            try:
                logger.info(f"Searching DuckDuckGo for: '{search_query}'")
                
                search_results = ddgs.text(
                    keywords=search_query,
                    max_results=25,
                    safesearch='moderate'
                )
                
                # Process results
                for result in search_results:
                    try:
                        title = result.get('title', '').strip()
                        url = result.get('href', '').strip()
                        body = result.get('body', '').strip()
                        
                        if url and title:
                            results.append({
                                'title': title,
                                'url': url,
                                'source': 'web_search',
                                'description': body[:200] if body else ''
                            })
                    except Exception as e:
                        logger.warning(f"Error processing search result: {e}")
                        continue
                
                logger.info(f"Found {len(search_results)} results for '{search_query}'")
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in web search for '{search_query}': {e}")
                continue
                
        logger.info(f"Total web search results collected: {len(results)}")
        return results
    
    def download_document(self, doc_info: Dict) -> bool:
        """Download a document and save it to the data directory"""
        try:
            url = doc_info['url']
            if url in self.collected_urls:
                return False
                
            # Determine file extension
            parsed_url = urlparse(url)
            file_ext = Path(parsed_url.path).suffix.lower()
            
            if not file_ext:
                # Try to determine from content-type
                head_response = requests.head(url, timeout=10, allow_redirects=True)
                content_type = head_response.headers.get('content-type', '').lower()
                if 'pdf' in content_type:
                    file_ext = '.pdf'
                elif 'html' in content_type:
                    file_ext = '.html'
                else:
                    file_ext = '.txt'
            
            # Create safe filename
            safe_title = "".join(c for c in doc_info['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_title = safe_title[:100]  # Limit filename length
            filename = f"{self.documents_collected:04d}_{safe_title}{file_ext}"
            filepath = self.data_dir / filename
            
            # Download the document
            response = requests.get(url, timeout=30, allow_redirects=True)
            response.raise_for_status()
            
            # Save to file
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            # Save metadata
            metadata = {
                'original_url': url,
                'title': doc_info['title'],
                'source': doc_info['source'],
                'filename': filename,
                'file_size': len(response.content),
                'content_type': response.headers.get('content-type', 'unknown')
            }
            
            metadata_file = filepath.with_suffix('.json')
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.collected_urls.add(url)
            self.documents_collected += 1
            
            logger.info(f"Downloaded: {filename} ({len(response.content)} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {doc_info.get('url', 'unknown')}: {e}")
            return False
    
    def collect_documents(self):
        """Main method to collect documents from various sources"""
        logger.info(f"Starting document collection for topic: '{self.topic}'")
        logger.info(f"Target: {self.max_documents} documents")
        
        all_results = []
        
        arxiv_results = []
        # Search different sources
        logger.info("Searching arXiv...")
        arxiv_results = self.search_arxiv(self.topic, max_results=800)
        all_results.extend(arxiv_results)
        logger.info(f"Found {len(arxiv_results)} arXiv papers")
        
        logger.info("Searching Google Scholar...")
        scholar_results = self.search_google_scholar(self.topic, num_pages=20)
        all_results.extend(scholar_results)
        logger.info(f"Found {len(scholar_results)} Google Scholar results")
        
        logger.info("Searching web...")
        web_results = self.search_web_general(self.topic)
        all_results.extend(web_results)
        logger.info(f"Found {len(web_results)} web results")
        
        logger.info(f"Total results found: {len(all_results)}")
        
        # Remove duplicates based on URL
        unique_results = []
        seen_urls = set()
        for result in all_results:
            if result['url'] not in seen_urls:
                unique_results.append(result)
                seen_urls.add(result['url'])
        
        logger.info(f"Unique results after deduplication: {len(unique_results)}")
        
        # Download documents
        logger.info("Starting document downloads...")
        successful_downloads = 0
        
        for i, doc_info in enumerate(unique_results):
            if self.documents_collected >= self.max_documents:
                break
                
            logger.info(f"Downloading {i+1}/{len(unique_results)}: {doc_info['title'][:50]}...")
            
            if self.download_document(doc_info):
                successful_downloads += 1
            
            # Rate limiting
            time.sleep(1)
            
            # Progress update every 50 downloads
            if successful_downloads % 50 == 0:
                logger.info(f"Progress: {successful_downloads}/{self.max_documents} documents downloaded")
        
        logger.info(f"Collection completed. Downloaded {successful_downloads} documents.")
        
        # Save collection summary
        summary = {
            'topic': self.topic,
            'total_found': len(unique_results),
            'total_downloaded': successful_downloads,
            'sources': {
                'arxiv': len(arxiv_results),
                'google_scholar': len(scholar_results),
                'web': len(web_results)
            }
        }
        
        with open(self.data_dir / 'collection_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
    
def main():
    collector = DocumentCollector()
    collector.collect_documents()
    
if __name__ == "__main__":
    main()
