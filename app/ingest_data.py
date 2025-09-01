#!/usr/bin/env python3
"""
Data ingestion script for processing blog.sql and creating clean datasets.

Usage:
    python -m app.ingest_data database\ and\ model/blog.sql
"""

import sys
import logging
from pathlib import Path
from .data_processor import SQLProcessor, validate_blog_data, export_data
from .config import ensure_storage_dirs

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main ingestion script."""
    if len(sys.argv) != 2:
        print("Usage: python -m app.ingest_data <path_to_blog.sql>")
        sys.exit(1)
    
    sql_file_path = sys.argv[1]
    
    if not Path(sql_file_path).exists():
        print(f"Error: SQL file not found: {sql_file_path}")
        sys.exit(1)
    
    # Ensure storage directories exist
    ensure_storage_dirs()
    
    try:
        # Process SQL file with Claude
        processor = SQLProcessor()
        logger.info("Starting SQL processing with Claude API...")
        blogs = processor.process_sql_file(sql_file_path)
        
        # Validate data
        logger.info("Validating processed data...")
        validation_report = validate_blog_data(blogs)
        
        print(f"\n=== Data Processing Complete ===")
        print(f"Total blogs processed: {validation_report['total_blogs']}")
        print(f"Average body length: {validation_report['statistics']['avg_body_length']} characters")
        print(f"Blogs with dates: {validation_report['statistics']['blogs_with_dates']}")
        print(f"Blogs with authors: {validation_report['statistics']['blogs_with_authors']}")
        print(f"Total unique tags: {validation_report['statistics']['total_unique_tags']}")
        
        if validation_report['validation_issues']:
            print(f"\nValidation Issues:")
            for issue in validation_report['validation_issues']:
                print(f"  - {issue}")
        else:
            print("\nNo validation issues found!")
        
        # Export data
        logger.info("Exporting to parquet and xlsx...")
        parquet_path, xlsx_path = export_data(blogs)
        
        print(f"\n=== Export Complete ===")
        print(f"Parquet file: {parquet_path}")
        print(f"XLSX file: {xlsx_path}")
        print(f"\nReady for vector index building!")
        
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()