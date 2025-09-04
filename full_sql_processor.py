#!/usr/bin/env python3
"""
Complete SQL processor using Gemini 2.5 Pro to extract all blog records.
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

# Configure Gemini with your API key
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
            # Now add URLs in brackets using BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            for link in soup.find_all('a', href=True):
                url = link.get('href')
                text = link.get_text().strip()
                if url and url.startswith(('http', 'https')):
                    link_text = f"{text} [{url}]" if text else f"[{url}]"
                    # Replace in clean text if the text exists
                    if text and text in clean_text:
                        clean_text = clean_text.replace(text, link_text)
                    else:
                        clean_text += f" {link_text}"
            return post_process_text(clean_text)
    except Exception as e:
        print(f"Trafilatura failed: {e}")
    
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
    except Exception as e:
        print(f"BeautifulSoup failed: {e}")
        # Last resort: basic regex
        text = re.sub(r'<[^>]+>', ' ', html)
        return post_process_text(text)

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
    
    # Extract technical terms
    words = re.findall(r'\b[a-zA-Z]{3,}\b', combined)
    for word in words:
        word = word.lower()
        if word in technical_terms:
            keywords.add(word)
    
    # Look for phrases
    phrases = ['data science', 'machine learning', 'deep learning', 'business analytics', 'power bi']
    for phrase in phrases:
        if phrase in combined:
            keywords.add(phrase)
    
    return sorted(list(keywords))

def process_sql_chunk_with_gemini(sql_chunk: str, chunk_num: int, total_chunks: int) -> List[Dict]:
    """Process a chunk of SQL with Gemini 2.5 Pro."""
    
    prompt = f"""
Parse this MySQL dump chunk ({chunk_num}/{total_chunks}) and extract blog data from INSERT statements.

Extract these fields for each blog record and return as JSON array:
- id: blog ID (integer)  
- title: blog_title field
- url: base_url field
- short_desc: blog_short_description field
- body: blog_description field (keep HTML as-is for now)
- author: author field  
- meta_title: meta_title field
- meta_description: meta_description field
- blog_date: blog_date field

IMPORTANT RULES:
1. Handle escaped quotes (\'), NULL values, and multi-line content carefully
2. Only include records where blog_status = 1 (active blogs)
3. Return ONLY a valid JSON array, no other text
4. Process ALL INSERT statements in this chunk

SQL Content:
{sql_chunk}
"""
    
    try:
        print(f"Processing chunk {chunk_num}/{total_chunks} with Gemini 2.5 Pro...")
        
        # Use the newer API format for gemini-2.5-pro
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=8192,
                temperature=0.1,
                candidate_count=1
            ),
            request_options={"timeout": 600}  # 10 minute timeout
        )
        
        content = response.text
        print(f"Gemini response length: {len(content)} characters")
        
        # Extract JSON from response
        start_idx = content.find('[')
        end_idx = content.rfind(']') + 1
        
        if start_idx == -1 or end_idx == 0:
            print(f"WARNING: No JSON array found in response for chunk {chunk_num}")
            print(f"Response preview: {content[:200]}...")
            return []
        
        json_str = content[start_idx:end_idx]
        blog_data = json.loads(json_str)
        
        print(f"Successfully extracted {len(blog_data)} records from chunk {chunk_num}")
        return blog_data
        
    except Exception as e:
        print(f"ERROR processing chunk {chunk_num}: {e}")
        return []

def split_sql_into_chunks(sql_content: str, max_chunk_size: int = 200000) -> List[str]:
    """Split SQL content into processable chunks."""
    
    # Find all INSERT statements
    insert_positions = []
    for match in re.finditer(r'INSERT INTO `blog`', sql_content):
        insert_positions.append(match.start())
    
    print(f"Found {len(insert_positions)} INSERT statements")
    
    chunks = []
    current_pos = 0
    
    for i, insert_pos in enumerate(insert_positions):
        # Find end of this INSERT statement
        if i + 1 < len(insert_positions):
            next_insert = insert_positions[i + 1]
            chunk_end = next_insert
        else:
            chunk_end = len(sql_content)
        
        # Extract INSERT statement
        insert_statement = sql_content[insert_pos:chunk_end].strip()
        
        # If this chunk would be too large, process what we have so far
        if len(insert_statement) > max_chunk_size and chunks:
            break
        
        chunks.append(insert_statement)
    
    print(f"Created {len(chunks)} chunks for processing")
    return chunks

def process_full_sql():
    """Process the complete blog.sql file."""
    
    sql_file = "database and model/blog.sql"
    print(f"Processing complete SQL file: {sql_file}")
    
    # Read full SQL file
    with open(sql_file, 'r', encoding='utf-8') as f:
        sql_content = f.read()
    
    print(f"SQL file size: {len(sql_content):,} characters")
    
    # Split into chunks
    chunks = split_sql_into_chunks(sql_content, max_chunk_size=150000)
    
    all_blogs = []
    
    # Process each chunk
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Processing Chunk {i}/{len(chunks)} ---")
        chunk_blogs = process_sql_chunk_with_gemini(chunk, i, len(chunks))
        
        if chunk_blogs:
            all_blogs.extend(chunk_blogs)
            print(f"Total blogs extracted so far: {len(all_blogs)}")
        
        # Rate limiting between API calls
        if i < len(chunks):
            print("Waiting 5 seconds between chunks...")
            time.sleep(5)
    
    print(f"\n=== Processing Complete ===")
    print(f"Total blog records extracted: {len(all_blogs)}")
    
    return all_blogs

def clean_and_export_all_blogs(raw_blogs: List[Dict]):
    """Clean all blog data and export."""
    
    print(f"Cleaning and processing {len(raw_blogs)} blogs...")
    
    cleaned_blogs = []
    
    for i, raw_blog in enumerate(raw_blogs, 1):
        try:
            if i % 50 == 0:
                print(f"Processed {i}/{len(raw_blogs)} blogs...")
            
            # Parse date
            published_at = None
            if raw_blog.get('blog_date'):
                date_str = str(raw_blog['blog_date']).strip()
                for fmt in ['%b %d, %Y', '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']:
                    try:
                        published_at = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
            
            # Clean HTML body
            body = ""
            if raw_blog.get('body'):
                body = clean_html_to_text(str(raw_blog['body']))
            
            # Extract tags
            tags = extract_keywords(
                raw_blog.get('meta_title', ''),
                raw_blog.get('meta_description', '')
            )
            
            cleaned_blog = {
                'id': str(raw_blog['id']),
                'title': str(raw_blog.get('title', '')).strip(),
                'url': str(raw_blog.get('url', '')).strip(),
                'short_desc': str(raw_blog.get('short_desc', '')).strip(),
                'body': body,
                'author': str(raw_blog['author']).strip() if raw_blog.get('author') and raw_blog['author'] != 'NULL' else None,
                'tags': ','.join(tags),
                'published_at': published_at.strftime('%Y-%m-%d') if published_at else None
            }
            
            cleaned_blogs.append(cleaned_blog)
            
        except Exception as e:
            print(f"Failed to clean blog {raw_blog.get('id', 'unknown')}: {e}")
    
    print(f"Successfully cleaned {len(cleaned_blogs)} blogs")
    
    # Create DataFrame
    df = pd.DataFrame(cleaned_blogs)
    
    # Create storage directory
    storage_path = Path("app/storage")
    storage_path.mkdir(parents=True, exist_ok=True)
    
    # Export files
    parquet_path = storage_path / "blog_data_full.parquet"
    xlsx_path = storage_path / "blog_data_full.xlsx"
    
    df.to_parquet(parquet_path, index=False)
    df.to_excel(xlsx_path, index=False)
    
    print(f"\n=== Full Dataset Export Complete ===")
    print(f"Total blogs: {len(df)}")
    print(f"Parquet: {parquet_path}")
    print(f"XLSX: {xlsx_path}")
    
    # Detailed validation
    print(f"\n=== Validation Report ===")
    print(f"Blogs with titles: {df['title'].notna().sum()}")
    print(f"Blogs with URLs: {df['url'].notna().sum()}")
    print(f"Blogs with body content: {df['body'].str.len().gt(0).sum()}")
    print(f"Blogs with authors: {df['author'].notna().sum()}")
    print(f"Blogs with dates: {df['published_at'].notna().sum()}")
    print(f"Average body length: {df['body'].str.len().mean():.0f} characters")
    print(f"Longest body: {df['body'].str.len().max()} characters")
    print(f"Total unique tags: {len(set(tag.strip() for tags in df['tags'] for tag in str(tags).split(',') if tag.strip()))}")
    
    # Show top tags
    all_tags = []
    for tags in df['tags']:
        if pd.notna(tags):
            all_tags.extend([tag.strip() for tag in str(tags).split(',') if tag.strip()])
    
    from collections import Counter
    top_tags = Counter(all_tags).most_common(10)
    print(f"Top 10 tags: {[f'{tag}({count})' for tag, count in top_tags]}")
    
    # Show sample blogs
    print(f"\n=== Sample Blogs ===")
    for i in range(min(3, len(df))):
        blog = df.iloc[i]
        print(f"{i+1}. {blog['title']}")
        print(f"   URL: {blog['url']}")
        print(f"   Author: {blog['author']}")
        print(f"   Tags: {blog['tags']}")
        print(f"   Body preview: {blog['body'][:100]}...")
        print()
    
    return df

def main():
    """Main processing function."""
    
    print("=== Full Blog SQL Processing with Gemini 2.5 Pro ===")
    print("This will process all ~914 blog records from blog.sql")
    print("Estimated time: 15-30 minutes with API rate limiting")
    
    try:
        # Process all SQL data
        raw_blogs = process_full_sql()
        
        if not raw_blogs:
            print("ERROR: No blog data extracted")
            return
        
        # Clean and export
        df = clean_and_export_all_blogs(raw_blogs)
        
        print(f"\n🎉 SUCCESS: Processed {len(df)} blogs from SQL file!")
        print(f"Files ready for Phase 2 vector indexing:")
        print(f"  - app/storage/blog_data_full.parquet")
        print(f"  - app/storage/blog_data_full.xlsx")
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()