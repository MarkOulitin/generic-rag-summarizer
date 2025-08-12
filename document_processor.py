import os
import json
import logging
from pathlib import Path
from typing import List, Dict
import re
from datetime import datetime

import PyPDF2
import pypdf
from bs4 import BeautifulSoup
import markdown
from docx import Document as DocxDocument

from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('document_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.chunk_size = int(os.getenv('CHUNK_SIZE', 512))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 50))
        self.input_dir = Path('./data/collected_documents')
        self.output_dir = Path('./data/processed_chunks')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Track processing statistics
        self.processed_files = 0
        self.total_chunks = 0
        self.processing_errors = 0
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
            
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\'"/@#$%&+=<>\[\]{}|\\]', ' ', text)
        
        # Remove repeated punctuation
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        
        # Clean up multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        
        # Strip and ensure proper spacing
        text = text.strip()
        
        return text
    
    def extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF files"""
        text = ""
        
        try:
            # Try with pypdf first (more modern)
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num} from {file_path}: {e}")
                        continue
                        
        except Exception as e:
            logger.warning(f"pypdf failed for {file_path}, trying PyPDF2: {e}")
            
            # Fallback to PyPDF2
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num, page in enumerate(pdf_reader.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n\n"
                        except Exception as e:
                            logger.warning(f"Error extracting page {page_num} from {file_path}: {e}")
                            continue
                            
            except Exception as e:
                logger.error(f"Both PDF extraction methods failed for {file_path}: {e}")
                return ""
        
        return self.clean_text(text)
    
    def extract_text_from_html(self, file_path: Path) -> str:
        """Extract text from HTML files"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract text from specific elements if they exist
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup
            
            text = main_content.get_text()
            return self.clean_text(text)
            
        except Exception as e:
            logger.error(f"Error extracting text from HTML {file_path}: {e}")
            return ""
    
    def extract_text_from_markdown(self, file_path: Path) -> str:
        """Extract text from Markdown files"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
            
            # Convert markdown to HTML then extract text
            html = markdown.markdown(content)
            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text()
            
            return self.clean_text(text)
            
        except Exception as e:
            logger.error(f"Error extracting text from Markdown {file_path}: {e}")
            return ""
    
    def extract_text_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX files"""
        try:
            doc = DocxDocument(file_path)
            text = ""
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return self.clean_text(text)
            
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {e}")
            return ""
    
    def extract_text_from_txt(self, file_path: Path) -> str:
        """Extract text from plain text files"""
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        content = file.read()
                    return self.clean_text(content)
                except UnicodeDecodeError:
                    continue
            
            logger.warning(f"Could not decode text file {file_path} with any encoding")
            return ""
            
        except Exception as e:
            logger.error(f"Error extracting text from TXT {file_path}: {e}")
            return ""
    
    def extract_text_from_file(self, file_path: Path) -> str:
        """Extract text from a file based on its extension"""
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif extension in ['.html', '.htm']:
            return self.extract_text_from_html(file_path)
        elif extension in ['.md', '.markdown']:
            return self.extract_text_from_markdown(file_path)
        elif extension in ['.docx', '.doc']:
            return self.extract_text_from_docx(file_path)
        elif extension == '.txt':
            return self.extract_text_from_txt(file_path)
        else:
            logger.warning(f"Unsupported file type: {extension} for {file_path}")
            return ""
    
    def create_chunks(self, text: str, metadata: Dict) -> List[Dict]:
        """Split text into chunks and create chunk metadata"""
        if not text or len(text.strip()) < 50:  # Skip very short texts
            return []
        
        try:
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            chunk_data = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    'chunk_id': f"{metadata['filename']}_{i:04d}",
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_size': len(chunk),
                    'source_file': metadata['filename'],
                    'original_url': metadata.get('original_url', ''),
                    'title': metadata.get('title', ''),
                    'source': metadata.get('source', ''),
                    'content_type': metadata.get('content_type', ''),
                    'processed_at': datetime.now().isoformat()
                }
                
                chunk_data.append({
                    'text': chunk.strip(),
                    'metadata': chunk_metadata
                })
            
            return chunk_data
            
        except Exception as e:
            logger.error(f"Error creating chunks for {metadata.get('filename', 'unknown')}: {e}")
            return []
    
    def process_single_document(self, file_path: Path) -> List[Dict]:
        """Process a single document and return its chunks"""
        # Skip metadata files
        if file_path.suffix == '.json':
            return []
        
        logger.info(f"Processing: {file_path.name}")
        
        # Load metadata if available
        metadata_file = file_path.with_suffix('.json')
        metadata = {}
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata for {file_path}: {e}")
        
        # Set default metadata if not available
        metadata.setdefault('filename', file_path.name)
        metadata.setdefault('title', file_path.stem)
        metadata.setdefault('source', 'unknown')
        
        try:
            # Extract text from the document
            text = self.extract_text_from_file(file_path)
            
            if not text:
                logger.warning(f"No text extracted from {file_path}")
                self.processing_errors += 1
                return []
            
            # Create chunks
            chunks = self.create_chunks(text, metadata)
            
            if chunks:
                self.processed_files += 1
                self.total_chunks += len(chunks)
                logger.info(f"Created {len(chunks)} chunks from {file_path.name}")
            else:
                logger.warning(f"No chunks created from {file_path}")
                self.processing_errors += 1
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            self.processing_errors += 1
            return []
    
    def process_all_documents(self) -> List[Dict]:
        """Process all documents in the input directory"""
        logger.info(f"Starting document processing from {self.input_dir}")
        logger.info(f"Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}")
        
        if not self.input_dir.exists():
            logger.error(f"Input directory {self.input_dir} does not exist")
            return []
        
        # Get all document files (exclude metadata files)
        document_files = [
            f for f in self.input_dir.iterdir() 
            if f.is_file() and f.suffix != '.json' and not f.name.startswith('.')
        ]
        
        logger.info(f"Found {len(document_files)} documents to process")
        
        all_chunks = []
        
        for i, file_path in enumerate(document_files, 1):
            logger.info(f"Processing document {i}/{len(document_files)}: {file_path.name}")
            
            chunks = self.process_single_document(file_path)
            all_chunks.extend(chunks)
            
            # Progress update every 50 files
            if i % 50 == 0:
                logger.info(f"Progress: {i}/{len(document_files)} files processed, {len(all_chunks)} chunks created")
        
        # Save all chunks to file
        output_file = self.output_dir / 'processed_chunks.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
        # Save processing summary
        summary = {
            'processing_completed_at': datetime.now().isoformat(),
            'input_directory': str(self.input_dir),
            'output_directory': str(self.output_dir),
            'total_documents_found': len(document_files),
            'documents_processed_successfully': self.processed_files,
            'processing_errors': self.processing_errors,
            'total_chunks_created': len(all_chunks),
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'average_chunks_per_document': len(all_chunks) / max(self.processed_files, 1)
        }
        
        summary_file = self.output_dir / 'processing_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Processing completed!")
        logger.info(f"Documents processed: {self.processed_files}/{len(document_files)}")
        logger.info(f"Total chunks created: {len(all_chunks)}")
        logger.info(f"Processing errors: {self.processing_errors}")
        logger.info(f"Results saved to: {output_file}")
        
        return all_chunks

def main():
    processor = DocumentProcessor()
    chunks = processor.process_all_documents()
    return chunks

if __name__ == "__main__":
    main()
