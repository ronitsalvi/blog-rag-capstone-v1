#!/usr/bin/env python3
"""
Complete SQL processor using older Gemini API to extract all blog records.
"""

import os
import json
import pandas as pd
import re
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import google.generativeai as genai
from bs4 import BeautifulSoup
import trafilatura

# Configure Gemini
GEMINI_API_KEY = "AIzaSyDlwtCyXrlhQzMCAvK8QEEbh46D2AFf-xc"
genai.configure(api_key=GEMINI_API_KEY)

def clean_html_to_text(html: str) -> str:
    """Convert HTML to clean text with URLs in brackets."""
    if not html or not html.strip():
        return ""
    
    try:
        # Use trafilatura for main content
        clean_text = trafilatura.extract(html, include_links=False)
        if clean_text and len(clean_text.strip()) > 50:
            # Add URLs in brackets
            soup = BeautifulSoup(html, 'html.parser')
            for link in soup.find_all('a', href=True):
                url = link.get('href')
                text = link.get_text().strip()
                if url and url.startswith(('http', 'https')):
                    if text and text in clean_text:
                        clean_text = clean_text.replace(text, f"{text} [{url}]")
            return post_process_text(clean_text)
    except:
        pass
    
    # Fallback to BeautifulSoup
    try:
        soup = BeautifulSoup(html, 'html.parser')
        
        # Replace links with [url] format
        for link in soup.find_all('a', href=True):
            url = link.get('href')
            text = link.get_text().strip()
            if url and url.startswith(('http', 'https')):
                link.replace_with(f"{text} [{url}]" if text else f"[{url}]")
        
        # Remove style and script tags
        for tag in soup(['style', 'script', 'meta', 'link']):
            tag.decompose()
        
        text = soup.get_text()
        return post_process_text(text)
    except:
        return post_process_text(re.sub(r'<[^>]+>', ' ', html))

def post_process_text(text: str) -> str:
    """Clean up extracted text."""
    if not text:
        return ""
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Clean bracket formatting
    text = re.sub(r'\s*\[\s*([^\]]+)\s*\]\s*', r' [\1] ', text)
    
    return text.strip()

def extract_keywords(meta_title: str, meta_description: str) -> List[str]:
    """Extract keywords from meta fields."""
    technical_terms = {
        'python', 'sql', 'data', 'science', 'machine', 'learning', 'deep',
        'analytics', 'statistics', 'pandas', 'numpy', 'matplotlib', 'seaborn',
        'tableau', 'power', 'bi', 'spark', 'apache', 'algorithm', 'model',
        'regression', 'classification', 'clustering', 'visualization', 'database',
        'programming', 'analysis', 'business', 'intelligence'
    }
    
    keywords = set()
    combined = f"{meta_title} {meta_description}".lower()
    
    words = re.findall(r'\b[a-zA-Z]{3,}\b', combined)
    for word in words:
        word = word.lower()
        if word in technical_terms:
            keywords.add(word)
    
    phrases = ['data science', 'machine learning', 'deep learning', 'business analytics']
    for phrase in phrases:
        if phrase in combined:
            keywords.add(phrase)
    
    return sorted(list(keywords))

def process_sql_chunk(sql_chunk: str, chunk_num: int) -> List[Dict]:
    """Process SQL chunk with older Gemini API."""
    
    prompt = f"""
Parse this MySQL dump and extract ALL blog data from INSERT statements.

Extract these exact fields for each record:
- id: blog ID (integer)
- title: blog_title field  
- url: base_url field
- short_desc: blog_short_description field
- body: blog_description field (keep HTML as-is)
- author: author field
- meta_title: meta_title field  
- meta_description: meta_description field
- blog_date: blog_date field

Handle escaped quotes, NULL values, multi-line content.
Only include records where blog_status = 1.
Return ONLY valid JSON array format.

SQL Content:
{sql_chunk[:25000]}
"""
    
    try:
        print(f"Calling Gemini API for chunk {chunk_num}...")
        
        model = genai.GenerativeModel('gemini-2.5-pro')
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=8192,
                temperature=0.1
            )
        )
        
        if not response.text:
            print(f"Empty response for chunk {chunk_num}")
            return []
        
        content = response.text
        print(f"Response length: {len(content)} chars")
        
        # Extract JSON
        start_idx = content.find('[')
        end_idx = content.rfind(']') + 1
        
        if start_idx == -1 or end_idx == 0:
            print(f"No JSON found in chunk {chunk_num}")
            return []
        
        json_str = content[start_idx:end_idx]
        blog_data = json.loads(json_str)
        
        print(f"Extracted {len(blog_data)} records from chunk {chunk_num}")
        return blog_data
        
    except Exception as e:
        print(f"ERROR processing chunk {chunk_num}: {e}")
        return []

def main():
    """Process complete SQL file."""
    
    sql_file = "database and model/blog.sql"
    
    with open(sql_file, 'r', encoding='utf-8') as f:
        sql_content = f.read()
    
    print(f"SQL file: {len(sql_content):,} characters")
    
    # Find INSERT statements and process in chunks
    insert_pattern = r'INSERT INTO `blog`[^;]*?VALUES[^;]*;'
    inserts = re.findall(insert_pattern, sql_content, re.DOTALL)
    
    print(f"Found {len(inserts)} INSERT statements")
    
    # Process in batches of 20 INSERT statements
    batch_size = 20
    all_blogs = []
    
    for i in range(0, len(inserts), batch_size):
        batch = inserts[i:i+batch_size]
        batch_content = '\n'.join(batch)
        
        chunk_num = i // batch_size + 1
        total_chunks = (len(inserts) + batch_size - 1) // batch_size
        
        print(f"\n--- Batch {chunk_num}/{total_chunks} ({len(batch)} INSERT statements) ---")
        
        batch_blogs = process_sql_chunk(batch_content, chunk_num)
        
        if batch_blogs:
            all_blogs.extend(batch_blogs)
            print(f"Total extracted: {len(all_blogs)} blogs")
        
        # Rate limiting
        if i + batch_size < len(inserts):
            print("Waiting 10 seconds...")
            time.sleep(10)
    
    print(f"\n=== Extraction Complete ===")
    print(f"Total blogs: {len(all_blogs)}")
    
    if all_blogs:
        # Clean and export
        cleaned_blogs = []
        
        for raw_blog in all_blogs:
            try:
                # Clean data
                body = clean_html_to_text(str(raw_blog.get('body', '')))
                tags = extract_keywords(
                    raw_blog.get('meta_title', ''),
                    raw_blog.get('meta_description', '')
                )
                
                cleaned = {
                    'id': str(raw_blog['id']),
                    'title': str(raw_blog.get('title', '')).strip(),
                    'url': str(raw_blog.get('url', '')).strip(),
                    'short_desc': str(raw_blog.get('short_desc', '')).strip(),
                    'body': body,
                    'author': str(raw_blog.get('author', '')).strip() or None,
                    'tags': ','.join(tags),
                    'published_at': str(raw_blog.get('blog_date', ''))
                }
                
                cleaned_blogs.append(cleaned)
                
            except Exception as e:
                print(f"Cleaning failed for blog {raw_blog.get('id')}: {e}")
        
        # Export
        df = pd.DataFrame(cleaned_blogs)
        
        storage_path = Path("app/storage")
        storage_path.mkdir(exist_ok=True)
        
        df.to_parquet(storage_path / "blog_data_full.parquet", index=False)
        df.to_excel(storage_path / "blog_data_full.xlsx", index=False)
        
        print(f"\n🎉 SUCCESS: {len(df)} blogs exported!")
        print(f"Files: blog_data_full.parquet, blog_data_full.xlsx")

if __name__ == "__main__":
    main()