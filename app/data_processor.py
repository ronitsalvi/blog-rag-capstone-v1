import os
import json
import pandas as pd
import logging
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import google.generativeai as genai
import google.ai.generativelanguage as glm
from .models import BlogDoc
from .utils_text import clean_html_to_text, extract_keywords_from_meta
from .config import get_storage_path

logger = logging.getLogger(__name__)

class SQLProcessor:
    """Processes blog.sql using Gemini API to extract clean blog data."""
    
    def __init__(self):
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        genai.configure(api_key=api_key)
        self.model = genai.models.GenerativeModel('gemini-pro')
        
    def process_sql_file(self, sql_file_path: str) -> List[BlogDoc]:
        """
        Process blog.sql file using Gemini to extract blog data.
        
        Args:
            sql_file_path: Path to the blog.sql file
            
        Returns:
            List of BlogDoc objects
        """
        logger.info(f"Processing SQL file: {sql_file_path}")
        
        # Read SQL file in chunks due to size
        with open(sql_file_path, 'r', encoding='utf-8') as f:
            sql_content = f.read()
        
        # Process with Gemini
        blog_data = self._extract_with_gemini(sql_content)
        
        # Clean and validate
        cleaned_blogs = []
        for raw_blog in blog_data:
            try:
                cleaned_blog = self._clean_blog_data(raw_blog)
                cleaned_blogs.append(cleaned_blog)
            except Exception as e:
                logger.warning(f"Failed to clean blog {raw_blog.get('id', 'unknown')}: {e}")
                
        logger.info(f"Successfully processed {len(cleaned_blogs)} blogs")
        return cleaned_blogs
    
    def _extract_with_gemini(self, sql_content: str) -> List[Dict[str, Any]]:
        """Extract blog data using Gemini API with chunked processing."""
        
        # Find all INSERT statement boundaries
        insert_chunks = self._chunk_sql_inserts(sql_content)
        logger.info(f"Found {len(insert_chunks)} INSERT statement chunks to process")
        
        all_blog_data = []
        failed_chunks = []
        start_time = time.time()
        
        for i, chunk in enumerate(insert_chunks):
            chunk_start = time.time()
            logger.info(f"Processing chunk {i+1}/{len(insert_chunks)} ({len(chunk)} statements)")
            
            # Try processing chunk with retry logic
            chunk_data = self._process_chunk_with_retry(chunk, i+1, max_retries=2)
            
            if chunk_data:
                all_blog_data.extend(chunk_data)
                chunk_time = time.time() - chunk_start
                total_time = time.time() - start_time
                estimated_remaining = (chunk_time * (len(insert_chunks) - i - 1))
                
                logger.info(f"✓ Chunk {i+1} extracted {len(chunk_data)} records. "
                          f"Total: {len(all_blog_data)} | "
                          f"Time: {chunk_time:.1f}s | "
                          f"ETA: {estimated_remaining/60:.1f}m")
            else:
                failed_chunks.append(i+1)
                logger.error(f"✗ Chunk {i+1} failed after retries")
        
        # Report final results
        total_time = time.time() - start_time
        logger.info(f"🎯 Extraction complete! {len(all_blog_data)} records extracted in {total_time/60:.1f}m")
        
        if failed_chunks:
            logger.warning(f"Failed chunks: {failed_chunks} ({len(failed_chunks)}/{len(insert_chunks)})")
        
        return all_blog_data
    
    def _process_chunk_with_retry(self, chunk: List[str], chunk_num: int, max_retries: int = 2) -> List[Dict[str, Any]]:
        """Process a chunk with retry logic for failed attempts."""
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"Retrying chunk {chunk_num}, attempt {attempt + 1}")
                    time.sleep(1)  # Brief delay before retry
                
                chunk_data = self._process_sql_chunk(chunk)
                return chunk_data
                
            except json.JSONDecodeError as e:
                logger.error(f"Chunk {chunk_num} attempt {attempt + 1}: Invalid JSON - {e}")
                if attempt == max_retries:
                    return []
                    
            except Exception as e:
                logger.error(f"Chunk {chunk_num} attempt {attempt + 1}: {e}")
                if attempt == max_retries:
                    return []
        
        return []
    
    def _chunk_sql_inserts(self, sql_content: str) -> List[List[str]]:
        """
        Split SQL content into chunks of complete INSERT statements.
        
        Returns:
            List of chunks, where each chunk contains 10-15 complete INSERT statements
        """
        lines = sql_content.split('\n')
        insert_statements = []
        current_statement = []
        in_insert = False
        
        for line in lines:
            line = line.strip()
            
            # Start of INSERT statement
            if line.startswith('INSERT INTO `blog`'):
                if current_statement and in_insert:
                    # Save previous statement
                    insert_statements.append('\n'.join(current_statement))
                current_statement = [line]
                in_insert = True
                
            elif in_insert and line:
                current_statement.append(line)
                
                # End of INSERT statement - look for closing );
                if line.endswith(');'):
                    insert_statements.append('\n'.join(current_statement))
                    current_statement = []
                    in_insert = False
        
        # Handle last statement if exists
        if current_statement and in_insert:
            insert_statements.append('\n'.join(current_statement))
        
        # Group into chunks of 10-15 statements each
        chunk_size = 12  # Conservative size for Gemini's context window
        chunks = []
        for i in range(0, len(insert_statements), chunk_size):
            chunk = insert_statements[i:i + chunk_size]
            chunks.append(chunk)
            
        logger.info(f"Split {len(insert_statements)} INSERT statements into {len(chunks)} chunks of ~{chunk_size} each")
        return chunks
    
    def _process_sql_chunk(self, insert_statements: List[str]) -> List[Dict[str, Any]]:
        """Process a chunk of INSERT statements with Gemini."""
        
        chunk_content = '\n\n'.join(insert_statements)
        
        prompt = """
        Parse these MySQL INSERT statements for the blog table and extract blog data.

        For each blog record, extract these fields and convert to JSON:
        - id: the blog ID (integer)
        - title: blog_title field
        - url: base_url field  
        - short_desc: blog_short_description field
        - body: blog_description field (keep as-is, will clean HTML separately)
        - author: author field
        - meta_title: meta_title field
        - meta_description: meta_description field
        - blog_date: blog_date field
        
        Handle these edge cases carefully:
        - Escaped quotes in string values
        - NULL values (convert to null in JSON)
        - Multi-line text content
        - Embedded quotes and special characters
        
        Return ONLY a valid JSON array of blog objects. Only include records where blog_status appears to be active (typically 1).
        
        SQL INSERT Statements:
        """
        
        response = self.model.generate_content(
            prompt + chunk_content,
            generation_config=genai.GenerationConfig(
                max_output_tokens=16384,  # Increased for larger chunks
                temperature=0.1
            )
        )
        
        # Extract JSON from response
        content = response.text
        
        # Find JSON array in response
        start_idx = content.find('[')
        end_idx = content.rfind(']') + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON array found in Gemini response")
            
        json_str = content[start_idx:end_idx]
        blog_data = json.loads(json_str)
        
        return blog_data
    
    def _clean_blog_data(self, raw_blog: Dict[str, Any]) -> BlogDoc:
        """Clean and validate a single blog record."""
        
        # Parse date
        published_at = None
        if raw_blog.get('blog_date'):
            try:
                date_str = str(raw_blog['blog_date']).strip()
                # Try common date formats
                for fmt in ['%b %d, %Y', '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']:
                    try:
                        published_at = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
            except Exception as e:
                logger.warning(f"Could not parse date '{raw_blog.get('blog_date')}': {e}")
        
        # Clean HTML content
        body = ""
        if raw_blog.get('body'):
            body = clean_html_to_text(str(raw_blog['body']))
        
        # Extract tags
        tags = extract_keywords_from_meta(
            raw_blog.get('meta_title', ''),
            raw_blog.get('meta_description', '')
        )
        
        return BlogDoc(
            id=str(raw_blog['id']),
            title=str(raw_blog.get('title', '')).strip(),
            url=str(raw_blog.get('url', '')).strip(),
            short_desc=str(raw_blog.get('short_desc', '')).strip(),
            body=body,
            author=str(raw_blog['author']).strip() if raw_blog.get('author') else None,
            tags=tags,
            published_at=published_at
        )

def validate_blog_data(blogs: List[BlogDoc]) -> Dict[str, Any]:
    """
    Validate processed blog data and return validation report.
    
    Args:
        blogs: List of processed BlogDoc objects
        
    Returns:
        Validation report with statistics and issues
    """
    report = {
        'total_blogs': len(blogs),
        'validation_issues': [],
        'statistics': {}
    }
    
    # Check required fields
    missing_titles = sum(1 for blog in blogs if not blog['title'])
    missing_urls = sum(1 for blog in blogs if not blog['url'])
    missing_body = sum(1 for blog in blogs if not blog['body'])
    
    if missing_titles > 0:
        report['validation_issues'].append(f"{missing_titles} blogs missing titles")
    if missing_urls > 0:
        report['validation_issues'].append(f"{missing_urls} blogs missing URLs")
    if missing_body > 0:
        report['validation_issues'].append(f"{missing_body} blogs missing body content")
    
    # Statistics
    avg_body_length = sum(len(blog['body']) for blog in blogs) / len(blogs) if blogs else 0
    total_tags = sum(len(blog['tags']) for blog in blogs)
    
    report['statistics'] = {
        'avg_body_length': int(avg_body_length),
        'total_unique_tags': len(set(tag for blog in blogs for tag in blog['tags'])),
        'blogs_with_dates': sum(1 for blog in blogs if blog['published_at']),
        'blogs_with_authors': sum(1 for blog in blogs if blog['author'])
    }
    
    return report

def export_data(blogs: List[BlogDoc], base_filename: str = "blog_data"):
    """
    Export blog data to both parquet and xlsx formats.
    
    Args:
        blogs: List of BlogDoc objects
        base_filename: Base name for output files
    """
    storage_path = get_storage_path()
    
    # Convert to DataFrame
    df_data = []
    for blog in blogs:
        df_data.append({
            'id': blog['id'],
            'title': blog['title'],
            'url': blog['url'],
            'short_desc': blog['short_desc'],
            'body': blog['body'],
            'author': blog['author'],
            'tags': ','.join(blog['tags']),  # Join tags as comma-separated string
            'published_at': blog['published_at']
        })
    
    df = pd.DataFrame(df_data)
    
    # Export to parquet
    parquet_path = storage_path / f"{base_filename}.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info(f"Exported to parquet: {parquet_path}")
    
    # Export to xlsx
    xlsx_path = storage_path / f"{base_filename}.xlsx"
    df.to_excel(xlsx_path, index=False)
    logger.info(f"Exported to xlsx: {xlsx_path}")
    
    return parquet_path, xlsx_path