#!/usr/bin/env python3
"""
Direct regex-based SQL parser to extract all 914 blog records.
No API dependency - uses pure Python regex parsing.
"""

import pandas as pd
import re
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import trafilatura

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

def unescape_sql_string(value: str) -> str:
    """Unescape SQL string values."""
    if not value or value == 'NULL':
        return ""
    
    # Remove quotes
    if value.startswith("'") and value.endswith("'"):
        value = value[1:-1]
    
    # Unescape common patterns
    value = value.replace("\\'", "'")
    value = value.replace('\\"', '"')
    value = value.replace('\\\\', '\\')
    value = value.replace('\\n', '\n')
    value = value.replace('\\r', '\r')
    value = value.replace('\\t', '\t')
    
    return value

def parse_insert_statement(insert_sql: str) -> List[Dict]:
    """Parse a single INSERT statement and extract blog records."""
    
    # Column order from the CREATE TABLE structure:
    # id, category, base_url, blog_title, blog_image, blog_app_image, 
    # blog_short_description, blog_description, author, author_description, 
    # author_image, author_linkedin, view_count, list_count, blog_date, 
    # meta_title, meta_description, page_schema, country, state, city, 
    # sitemap_status, blog_status, created_at, created_by, created_ipaddress, 
    # updated_at, updated_by, updated_ipaddress
    
    try:
        # Find the VALUES clause and extract the tuple
        values_match = re.search(r'VALUES\s*\((.+)\);?\s*$', insert_sql, re.DOTALL)
        if not values_match:
            return []
        
        # Get the full tuple content
        tuple_content = values_match.group(1)
        
        # Parse fields using a more robust approach
        fields = []
        current_field = ""
        in_quotes = False
        quote_char = None
        paren_depth = 0
        
        i = 0
        while i < len(tuple_content):
            char = tuple_content[i]
            
            if char == '(' and not in_quotes:
                paren_depth += 1
                current_field += char
            elif char == ')' and not in_quotes:
                paren_depth -= 1
                current_field += char
            elif not in_quotes and char in ("'", '"'):
                in_quotes = True
                quote_char = char
                current_field += char
            elif in_quotes and char == quote_char:
                # Check if escaped
                if i > 0 and tuple_content[i-1] == '\\':
                    current_field += char
                else:
                    in_quotes = False
                    quote_char = None
                    current_field += char
            elif not in_quotes and char == ',' and paren_depth == 0:
                fields.append(current_field.strip())
                current_field = ""
            else:
                current_field += char
            
            i += 1
        
        # Add last field
        if current_field.strip():
            fields.append(current_field.strip())
        
        # Map to our needed fields
        if len(fields) >= 23:  # blog_status is at index 22
            blog_status = fields[22].strip()
            
            # Only process active blogs
            if blog_status == '1':
                record = {
                    'id': fields[0].strip(),
                    'title': unescape_sql_string(fields[3]),  # blog_title
                    'url': unescape_sql_string(fields[2]),   # base_url
                    'short_desc': unescape_sql_string(fields[6]),  # blog_short_description
                    'body': unescape_sql_string(fields[7]),  # blog_description
                    'author': unescape_sql_string(fields[8]),  # author
                    'meta_title': unescape_sql_string(fields[15]),  # meta_title
                    'meta_description': unescape_sql_string(fields[16]),  # meta_description
                    'blog_date': unescape_sql_string(fields[14])  # blog_date
                }
                return [record]
        
        return []
        
    except Exception as e:
        print(f"Parse error: {e}")
        return []

def process_complete_sql_file(sql_file: str) -> List[Dict]:
    """Process the complete SQL file using direct regex parsing."""
    
    print(f"Reading SQL file: {sql_file}")
    
    with open(sql_file, 'r', encoding='utf-8') as f:
        sql_content = f.read()
    
    print(f"SQL file size: {len(sql_content):,} characters")
    
    # Find all INSERT statements for blog table
    insert_pattern = r'INSERT INTO `blog`[^;]*?VALUES[^;]*;'
    inserts = re.findall(insert_pattern, sql_content, re.DOTALL | re.IGNORECASE)
    
    print(f"Found {len(inserts)} INSERT statements")
    
    all_blogs = []
    
    for i, insert_sql in enumerate(inserts, 1):
        if i % 50 == 0:
            print(f"Processing INSERT {i}/{len(inserts)}...")
        
        try:
            records = parse_insert_statement(insert_sql)
            all_blogs.extend(records)
        except Exception as e:
            print(f"Failed to parse INSERT {i}: {e}")
    
    print(f"\nExtraction complete: {len(all_blogs)} blog records")
    return all_blogs

def clean_and_export_blogs(raw_blogs: List[Dict]):
    """Clean blog data and export to files."""
    
    print(f"Cleaning {len(raw_blogs)} blog records...")
    
    cleaned_blogs = []
    
    for i, blog in enumerate(raw_blogs, 1):
        if i % 100 == 0:
            print(f"Cleaned {i}/{len(raw_blogs)} blogs...")
        
        try:
            # Clean HTML body
            body = clean_html_to_text(blog.get('body', ''))
            
            # Extract keywords
            tags = extract_keywords(
                blog.get('meta_title', ''),
                blog.get('meta_description', '')
            )
            
            # Parse date
            published_at = None
            if blog.get('blog_date'):
                date_str = str(blog['blog_date']).strip()
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%b %d, %Y']:
                    try:
                        published_at = datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
                        break
                    except ValueError:
                        continue
            
            cleaned = {
                'id': str(blog['id']),
                'title': str(blog.get('title', '')).strip(),
                'url': str(blog.get('url', '')).strip(),
                'short_desc': str(blog.get('short_desc', '')).strip(),
                'body': body,
                'author': str(blog.get('author', '')).strip() or None,
                'tags': ','.join(tags),
                'published_at': published_at
            }
            
            # Only include blogs with meaningful content
            if cleaned['title'] and cleaned['body'] and len(cleaned['body']) > 50:
                cleaned_blogs.append(cleaned)
        
        except Exception as e:
            print(f"Failed to clean blog {blog.get('id')}: {e}")
    
    print(f"Successfully cleaned {len(cleaned_blogs)} blogs")
    
    # Create DataFrame and export
    df = pd.DataFrame(cleaned_blogs)
    
    storage_path = Path("app/storage")
    storage_path.mkdir(parents=True, exist_ok=True)
    
    # Export files
    parquet_path = storage_path / "blog_data_full.parquet"
    xlsx_path = storage_path / "blog_data_full.xlsx"
    
    df.to_parquet(parquet_path, index=False)
    df.to_excel(xlsx_path, index=False)
    
    print(f"\n=== Export Complete ===")
    print(f"Total blogs: {len(df)}")
    print(f"Parquet: {parquet_path}")
    print(f"XLSX: {xlsx_path}")
    
    # Validation report
    print(f"\n=== Validation Report ===")
    print(f"Blogs with titles: {df['title'].notna().sum()}")
    print(f"Blogs with URLs: {df['url'].notna().sum()}")
    print(f"Blogs with body content: {df['body'].str.len().gt(50).sum()}")
    print(f"Blogs with authors: {df['author'].notna().sum()}")
    print(f"Average body length: {df['body'].str.len().mean():.0f} characters")
    
    # Show sample
    print(f"\n=== Sample Records ===")
    for i in range(min(3, len(df))):
        blog = df.iloc[i]
        print(f"{i+1}. {blog['title']}")
        print(f"   URL: {blog['url']}")
        print(f"   Body: {blog['body'][:100]}...")
        print()
    
    return df

def main():
    """Main processing function."""
    
    print("=== Direct SQL Parser - Processing All Blog Records ===")
    print("Using regex-based parsing (no API dependency)")
    
    sql_file = "database and model/blog.sql"
    
    try:
        # Extract all blog records
        raw_blogs = process_complete_sql_file(sql_file)
        
        if not raw_blogs:
            print("ERROR: No blog records extracted")
            return
        
        # Clean and export
        df = clean_and_export_blogs(raw_blogs)
        
        print(f"\n🎉 SUCCESS: Processed {len(df)} blogs!")
        print(f"Files ready for vector indexing:")
        print(f"  - app/storage/blog_data_full.parquet")
        print(f"  - app/storage/blog_data_full.xlsx")
        
        return df
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()