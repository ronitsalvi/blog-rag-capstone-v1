#!/usr/bin/env python3
"""
CSV-style SQL parser - treat SQL as CSV data.
"""

import pandas as pd
import re
import csv
from pathlib import Path
from io import StringIO
from bs4 import BeautifulSoup

def clean_html_to_text(html: str) -> str:
    """Clean HTML content."""
    if not html or html == 'NULL':
        return ""
    
    try:
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(['style', 'script']):
            tag.decompose()
        text = soup.get_text()
        return re.sub(r'\s+', ' ', text).strip()[:1500]
    except:
        text = re.sub(r'<[^>]+>', ' ', str(html))
        return re.sub(r'\s+', ' ', text).strip()[:1500]

def main():
    """Extract using CSV parsing approach."""
    
    print("=== CSV-Style SQL Parser ===")
    
    sql_file = "database and model/blog.sql"
    
    with open(sql_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract all INSERT statements
    insert_pattern = r"INSERT INTO `blog`.*?VALUES\s*\((.*?)\);"
    matches = re.findall(insert_pattern, content, re.DOTALL)
    
    print(f"Found {len(matches)} INSERT statements")
    
    # Column mapping (0-indexed)
    columns = [
        'id', 'category', 'base_url', 'blog_title', 'blog_image', 'blog_app_image',
        'blog_short_description', 'blog_description', 'author', 'author_description',
        'author_image', 'author_linkedin', 'view_count', 'list_count', 'blog_date',
        'meta_title', 'meta_description', 'page_schema', 'country', 'state', 'city',
        'sitemap_status', 'blog_status', 'created_at', 'created_by', 'created_ipaddress',
        'updated_at', 'updated_by', 'updated_ipaddress'
    ]
    
    blogs = []
    
    for i, match in enumerate(matches):
        if i % 100 == 0:
            print(f"Processing {i}/{len(matches)}...")
        
        try:
            # Convert SQL VALUES to CSV format
            csv_line = match
            
            # Handle newlines in the data
            csv_line = csv_line.replace('\n', '\\n').replace('\r', '\\r')
            
            # Basic field splitting using CSV reader
            reader = csv.reader([csv_line], quotechar="'")
            row = next(reader)
            
            # Check if we have enough fields and blog_status = 1
            if len(row) >= 23 and row[22].strip() == '1':
                blog = {
                    'id': row[0].strip(),
                    'title': row[3].strip(),  # blog_title
                    'url': row[2].strip(),    # base_url
                    'short_desc': row[6].strip(),  # blog_short_description
                    'body': clean_html_to_text(row[7]),  # blog_description
                    'author': row[8].strip() if row[8] != 'NULL' else None,
                    'published_at': row[14].strip() if row[14] != 'NULL' else None
                }
                
                # Only include if has title and body
                if blog['title'] and blog['body'] and len(blog['body']) > 50:
                    blogs.append(blog)
                    
        except Exception as e:
            print(f"Parse error for record {i}: {e}")
    
    print(f"Successfully parsed {len(blogs)} blog records")
    
    if blogs:
        # Export
        df = pd.DataFrame(blogs)
        
        storage_path = Path("app/storage")
        storage_path.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(storage_path / "blog_data_full.parquet", index=False)
        df.to_excel(storage_path / "blog_data_full.xlsx", index=False)
        
        print(f"\n✅ SUCCESS: {len(df)} blogs exported!")
        print(f"Files: blog_data_full.parquet, blog_data_full.xlsx")
        
        # Validation
        print(f"\nValidation:")
        print(f"- Total blogs: {len(df)}")
        print(f"- With titles: {df['title'].notna().sum()}")
        print(f"- With body: {df['body'].str.len().gt(50).sum()}")
        print(f"- Avg body length: {df['body'].str.len().mean():.0f}")
        
        # Sample
        print(f"\nSample:")
        blog = df.iloc[0]
        print(f"Title: {blog['title']}")
        print(f"URL: {blog['url']}")
        print(f"Body: {blog['body'][:100]}...")

if __name__ == "__main__":
    main()