#!/usr/bin/env python3
"""
Pure Python SQL parser to extract all blog data from blog.sql file.
Handles SQL escaping, NULL values, and multi-line content properly.
"""

import re
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SQLBlogParser:
    """Parses blog.sql file to extract all blog records programmatically."""
    
    def __init__(self):
        # Column names in INSERT order (29 total columns)
        self.columns = [
            'id', 'category', 'base_url', 'blog_title', 'blog_image', 'blog_app_image',
            'blog_short_description', 'blog_description', 'author', 'author_description',
            'author_image', 'author_linkedin', 'view_count', 'list_count', 'blog_date',
            'meta_title', 'meta_description', 'page_schema', 'country', 'state', 'city',
            'sitemap_status', 'blog_status', 'created_at', 'created_by', 'created_ipaddress',
            'updated_at', 'updated_by', 'updated_ipaddress'
        ]
        
    def parse_sql_file(self, sql_file_path: str) -> List[Dict[str, Any]]:
        """
        Parse the complete SQL file and extract all blog records.
        
        Args:
            sql_file_path: Path to blog.sql file
            
        Returns:
            List of blog records as dictionaries
        """
        logger.info(f"📁 Reading SQL file: {sql_file_path}")
        
        with open(sql_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        logger.info(f"📏 File size: {len(content):,} characters")
        
        # Extract all INSERT statements
        insert_statements = self._extract_insert_statements(content)
        logger.info(f"🔍 Found {len(insert_statements)} INSERT statements")
        
        # Parse each INSERT statement
        blog_records = []
        malformed_count = 0
        start_time = time.time()
        
        for i, statement in enumerate(insert_statements):
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                estimated_total = elapsed * len(insert_statements) / (i + 1)
                remaining = estimated_total - elapsed
                logger.info(f"⚡ Processed {i+1}/{len(insert_statements)} records. ETA: {remaining:.1f}s")
            
            try:
                record = self._parse_insert_statement(statement)
                if record:
                    blog_records.append(record)
            except Exception as e:
                malformed_count += 1
                logger.warning(f"⚠️  Malformed record {i+1}: {str(e)[:100]}...")
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Parsing complete! {len(blog_records)} valid records, {malformed_count} malformed in {elapsed_time:.1f}s")
        
        return blog_records
    
    def _extract_insert_statements(self, content: str) -> List[str]:
        """Extract individual value tuples from the massive INSERT statement."""
        
        # Find the VALUES section
        values_match = re.search(r'INSERT INTO `blog`.*?VALUES\s*(.*);', content, re.DOTALL)
        if not values_match:
            logger.error("No VALUES section found in SQL file")
            return []
        
        values_section = values_match.group(1).strip()
        
        # Parse individual value tuples from the VALUES section
        # Each tuple starts with ( and ends with ), followed by comma or end
        value_tuples = []
        
        # Split on ),( pattern but preserve the parentheses
        # Handle the complex case of nested quotes and escaped characters
        current_tuple = ""
        paren_depth = 0
        in_string = False
        quote_char = None
        i = 0
        
        while i < len(values_section):
            char = values_section[i]
            
            if not in_string:
                if char in ("'", '"'):
                    in_string = True
                    quote_char = char
                elif char == '(':
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1
                    if paren_depth == 0:
                        # End of current tuple
                        current_tuple += char
                        value_tuples.append(current_tuple.strip())
                        current_tuple = ""
                        # Skip comma and whitespace
                        while i + 1 < len(values_section) and values_section[i + 1] in ', \n\r\t':
                            i += 1
                        i += 1
                        continue
            else:
                # Inside quoted string
                if char == '\\' and i + 1 < len(values_section):
                    # Escaped character
                    current_tuple += char
                    i += 1
                    if i < len(values_section):
                        current_tuple += values_section[i]
                    i += 1
                    continue
                elif char == quote_char:
                    in_string = False
                    quote_char = None
            
            current_tuple += char
            i += 1
        
        # Handle last tuple if exists
        if current_tuple.strip():
            value_tuples.append(current_tuple.strip())
        
        logger.info(f"🔍 Extracted {len(value_tuples)} value tuples from SQL")
        return value_tuples
    
    def _parse_insert_statement(self, value_tuple: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single value tuple to extract field values.
        
        Args:
            value_tuple: Single tuple like "(1, 'title', 'url', ...)"
            
        Returns:
            Dictionary with parsed field values, or None if parsing fails
        """
        
        # Remove outer parentheses
        if not (value_tuple.startswith('(') and value_tuple.endswith(')')):
            raise ValueError("Value tuple should start with ( and end with )")
        
        values_str = value_tuple[1:-1].strip()
        
        # Parse comma-separated values respecting quotes
        values = self._parse_values_list(values_str)
        
        if len(values) != 29:
            raise ValueError(f"Expected 29 values, got {len(values)}")
        
        # Map values to column names
        record = {}
        for i, col_name in enumerate(self.columns):
            value = values[i]
            
            # Handle NULL values
            if value == 'NULL':
                record[col_name] = None
            else:
                # Remove quotes and unescape SQL strings
                if value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]  # Remove outer quotes
                    value = value.replace("\\'", "'")  # Unescape single quotes
                    value = value.replace('\\"', '"')  # Unescape double quotes
                    value = value.replace('\\\\', '\\')  # Unescape backslashes
                
                record[col_name] = value
        
        return record
    
    def _parse_values_list(self, values_str: str) -> List[str]:
        """
        Parse comma-separated values respecting SQL string quoting.
        
        This is the critical function that handles:
        - Escaped quotes within strings
        - Multi-line strings  
        - Nested quotes and special characters
        """
        
        values = []
        current_value = ""
        in_string = False
        quote_char = None
        i = 0
        
        while i < len(values_str):
            char = values_str[i]
            
            if not in_string:
                if char in ("'", '"'):
                    # Start of quoted string
                    in_string = True
                    quote_char = char
                    current_value += char
                elif char == ',' and not in_string:
                    # End of value
                    values.append(current_value.strip())
                    current_value = ""
                elif char == 'N' and values_str[i:i+4] == 'NULL':
                    # Handle NULL value
                    current_value += 'NULL'
                    i += 3  # Skip 'ULL'
                else:
                    current_value += char
            else:
                # Inside quoted string
                if char == '\\' and i + 1 < len(values_str):
                    # Escaped character - include both backslash and next char
                    current_value += char
                    i += 1
                    if i < len(values_str):
                        current_value += values_str[i]
                elif char == quote_char:
                    # End of quoted string
                    current_value += char
                    in_string = False
                    quote_char = None
                else:
                    current_value += char
            
            i += 1
        
        # Add final value
        if current_value.strip():
            values.append(current_value.strip())
        
        return values

def export_to_excel(blog_records: List[Dict[str, Any]], output_path: str):
    """Export blog records to Excel file."""
    
    # Select key columns for export
    export_data = []
    for record in blog_records:
        export_data.append({
            'id': record.get('id'),
            'title': record.get('blog_title'),
            'url': record.get('base_url'),
            'short_desc': record.get('blog_short_description'),
            'body': record.get('blog_description'),  # Keep as HTML
            'author': record.get('author'),
            'meta_title': record.get('meta_title'),
            'meta_description': record.get('meta_description'),
            'blog_date': record.get('blog_date'),
            'blog_status': record.get('blog_status')
        })
    
    df = pd.DataFrame(export_data)
    
    # Filter only active blogs (blog_status = 1)
    active_blogs = df[df['blog_status'] == '1']
    logger.info(f"📊 Filtered to {len(active_blogs)} active blogs (blog_status=1)")
    
    # Drop blog_status column from export (used only for filtering)
    active_blogs = active_blogs.drop('blog_status', axis=1)
    
    # Export to Excel
    active_blogs.to_excel(output_path, index=False)
    logger.info(f"💾 Exported to: {output_path}")
    
    return len(active_blogs)

def main():
    """Run complete SQL parsing and export."""
    
    # File paths
    sql_file = Path("database and model/blog.sql")
    output_file = Path("Documentation/all-blogs-extracted.xlsx")
    
    if not sql_file.exists():
        logger.error(f"❌ SQL file not found: {sql_file}")
        return 1
    
    # Ensure Documentation directory exists
    output_file.parent.mkdir(exist_ok=True)
    
    logger.info("🚀 Starting programmatic SQL parsing...")
    start_time = time.time()
    
    try:
        # Parse SQL file
        parser = SQLBlogParser()
        blog_records = parser.parse_sql_file(str(sql_file))
        
        if not blog_records:
            logger.error("❌ No blog records extracted!")
            return 1
        
        # Export to Excel
        active_count = export_to_excel(blog_records, str(output_file))
        
        total_time = time.time() - start_time
        logger.info(f"🎉 Extraction completed in {total_time:.1f}s!")
        logger.info(f"📊 Total parsed: {len(blog_records)} records")
        logger.info(f"📊 Active exported: {active_count} records")
        logger.info(f"📁 File saved: {output_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Parsing failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)