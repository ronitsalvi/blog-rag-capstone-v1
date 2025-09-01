import re
import trafilatura
from bs4 import BeautifulSoup
from typing import List
import nltk
from nltk.tokenize import sent_tokenize
import logging

logger = logging.getLogger(__name__)

def clean_html_to_text(html: str) -> str:
    """
    Convert HTML content to clean plain text with URLs in brackets.
    
    Args:
        html: Raw HTML content from blog_description
        
    Returns:
        Clean plain text with URLs preserved in brackets
    """
    if not html or not html.strip():
        return ""
    
    try:
        # First attempt with trafilatura for main content extraction
        clean_text = trafilatura.extract(html, include_links=True)
        
        if clean_text and len(clean_text.strip()) > 50:
            # Trafilatura succeeded, clean up further
            return _post_process_text(clean_text)
        else:
            # Fallback to BeautifulSoup approach
            return _fallback_html_cleaning(html)
            
    except Exception as e:
        logger.warning(f"HTML cleaning failed, using fallback: {e}")
        return _fallback_html_cleaning(html)

def _fallback_html_cleaning(html: str) -> str:
    """Fallback HTML cleaning using BeautifulSoup."""
    try:
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract links and replace with [url] format
        for link in soup.find_all('a', href=True):
            url = link.get('href')
            text = link.get_text().strip()
            if url and url.startswith(('http', 'https')):
                link.replace_with(f"{text} [{url}]" if text else f"[{url}]")
        
        # Remove style and script tags
        for tag in soup(['style', 'script', 'meta', 'link']):
            tag.decompose()
        
        # Get text and clean
        text = soup.get_text()
        return _post_process_text(text)
        
    except Exception as e:
        logger.error(f"Fallback HTML cleaning failed: {e}")
        # Last resort: regex-based cleaning
        text = re.sub(r'<[^>]+>', ' ', html)
        return _post_process_text(text)

def _post_process_text(text: str) -> str:
    """Post-process extracted text for consistency."""
    if not text:
        return ""
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Clean up bracket formatting
    text = re.sub(r'\s*\[\s*([^\]]+)\s*\]\s*', r' [\1] ', text)
    
    # Remove excessive punctuation
    text = re.sub(r'[.]{3,}', '...', text)
    
    # Clean and strip
    text = text.strip()
    
    return text

def extract_keywords_from_meta(meta_title: str, meta_description: str) -> List[str]:
    """
    Extract keywords from meta title and description using simple heuristics.
    
    Args:
        meta_title: The meta title field
        meta_description: The meta description field
        
    Returns:
        List of extracted keywords/tags
    """
    # Common stop words to filter out
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could',
        'can', 'may', 'might', 'must', 'this', 'that', 'these', 'those'
    }
    
    # Technical keywords that should be preserved
    technical_terms = {
        'python', 'sql', 'data', 'science', 'machine', 'learning', 'deep',
        'analytics', 'statistics', 'pandas', 'numpy', 'matplotlib', 'seaborn',
        'tableau', 'power', 'bi', 'spark', 'apache', 'algorithm', 'model',
        'regression', 'classification', 'clustering', 'visualization', 'database'
    }
    
    keywords = set()
    
    # Process both meta fields
    combined_text = f"{meta_title} {meta_description}".lower()
    
    # Extract words and filter
    words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_text)
    
    for word in words:
        word = word.lower().strip()
        if word in technical_terms:
            keywords.add(word)
        elif len(word) >= 4 and word not in stop_words:
            keywords.add(word)
    
    # Look for common data science phrases
    phrases = [
        'data science', 'machine learning', 'deep learning', 'business analytics',
        'power bi', 'apache spark', 'data visualization', 'predictive analytics'
    ]
    
    for phrase in phrases:
        if phrase in combined_text:
            keywords.add(phrase)
    
    return sorted(list(keywords))

def chunk_text_sentences(text: str, target_size: int = 600, overlap: int = 120):
    """
    Chunk text into sentence-aware segments.
    
    Args:
        text: Input text to chunk
        target_size: Target character count per chunk
        overlap: Overlap characters between chunks
        
    Returns:
        List of (chunk_text, start_char, end_char) tuples
    """
    try:
        nltk.download('punkt', quiet=True)
        sentences = sent_tokenize(text)
    except:
        # Fallback sentence splitting
        sentences = re.split(r'[.!?]+\s+', text)
    
    chunks = []
    current_chunk = ""
    current_start = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Check if adding this sentence would exceed target size
        potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
        
        if len(potential_chunk) > target_size and current_chunk:
            # Save current chunk
            end_pos = current_start + len(current_chunk)
            chunks.append((current_chunk.strip(), current_start, end_pos))
            
            # Start new chunk with overlap
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + " " + sentence
            current_start = end_pos - len(overlap_text)
        else:
            current_chunk = potential_chunk
    
    # Add final chunk if exists
    if current_chunk.strip():
        end_pos = current_start + len(current_chunk)
        chunks.append((current_chunk.strip(), current_start, end_pos))
    
    return chunks