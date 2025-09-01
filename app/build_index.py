#!/usr/bin/env python3
"""
Vector index building script for blog Q&A system.

Usage:
    python -m app.build_index
"""

import logging
import pandas as pd
import hashlib
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer
from .config import get_config, get_storage_path, ensure_storage_dirs
from .utils_text import chunk_text_sentences

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VectorIndexBuilder:
    """Builds and manages the ChromaDB vector index."""
    
    def __init__(self):
        self.config = get_config()
        self.storage_path = get_storage_path()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        ensure_storage_dirs()
        
    def build_index(self) -> None:
        """Build complete vector index from blog data."""
        
        # Load blog data
        parquet_path = self.storage_path / "v2_blog_data.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Blog data not found at {parquet_path}. Run conversion script first."
            )
        
        logger.info(f"Loading blog data from {parquet_path}")
        df = pd.read_parquet(parquet_path)
        
        # Convert to BlogDoc objects
        blogs = self._dataframe_to_blogs(df)
        logger.info(f"Loaded {len(blogs)} blog documents")
        
        # Create chunks
        logger.info("Creating document chunks...")
        all_chunks = []
        for blog in blogs:
            chunks = self._create_chunks(blog)
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(blogs)} blogs")
        
        # Build vector index
        logger.info("Building vector index...")
        self._build_chromadb_index(all_chunks)
        
        logger.info("Vector index build complete!")
    
    def _dataframe_to_blogs(self, df: pd.DataFrame):
        """Convert DataFrame to BlogDoc objects."""
        blogs = []
        
        for _, row in df.iterrows():
            # Parse tags back from comma-separated string
            tags = []
            if pd.notna(row['tags']) and row['tags']:
                tags = [tag.strip() for tag in str(row['tags']).split(',') if tag.strip()]
            
            # Handle published_at
            published_at = None
            if pd.notna(row['published_at']):
                published_at = pd.to_datetime(row['published_at'], errors='coerce')
            
            blog = {
                'id': str(row['id']),
                'title': str(row['title']) if pd.notna(row['title']) else "",
                'url': str(row['url']) if pd.notna(row['url']) else "",
                'short_desc': str(row['short_desc']) if pd.notna(row['short_desc']) else "",
                'body': str(row['body']) if pd.notna(row['body']) else "",
                'author': str(row['author']) if pd.notna(row['author']) else "",
                'tags': tags,
                'published_at': published_at
            }
            blogs.append(blog)
        
        return blogs
    
    def _create_chunks(self, blog: Dict[str, Any]):
        """Create chunks from a blog document."""
        if not blog['body']:
            return []
        
        # Generate source ID
        source_id = hashlib.md5(f"{blog['id']}_{blog['url']}".encode()).hexdigest()
        
        # Create text chunks
        chunk_size = self.config['retrieval']['chunk_size']
        overlap = self.config['retrieval']['chunk_overlap']
        
        text_chunks = chunk_text_sentences(blog['body'], chunk_size, overlap)
        
        chunks = []
        for i, (chunk_text, start_char, end_char) in enumerate(text_chunks):
            chunk_id = f"{source_id}_{i}"
            
            chunk = {
                'chunk_id': chunk_id,
                'source_id': source_id,
                'title': blog['title'],
                'url': blog['url'],
                'content': chunk_text,
                'chunk_index': i,
                'char_start': start_char,
                'char_end': end_char
            }
            chunks.append(chunk)
        
        return chunks
    
    def _build_chromadb_index(self, chunks: List[Dict[str, Any]]) -> None:
        """Build ChromaDB vector index from chunks."""
        
        # Initialize ChromaDB
        chroma_path = self.storage_path / "chroma"
        client = chromadb.PersistentClient(path=str(chroma_path))
        
        # Create or get collection
        try:
            collection = client.get_collection("blog_chunks")
            # Clear existing data for fresh build
            client.delete_collection("blog_chunks")
        except:
            pass
        
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        
        collection = client.create_collection(
            name="blog_chunks",
            embedding_function=SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )
        
        # Prepare data for bulk insert
        chunk_ids = [chunk['chunk_id'] for chunk in chunks]
        documents = [chunk['content'] for chunk in chunks]
        metadatas = [
            {
                'source_id': chunk['source_id'],
                'title': chunk['title'],
                'url': chunk['url'],
                'chunk_index': chunk['chunk_index'],
                'char_start': chunk['char_start'],
                'char_end': chunk['char_end']
            }
            for chunk in chunks
        ]
        
        # Insert in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            
            logger.info(f"Inserting batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            
            collection.add(
                ids=chunk_ids[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
        
        logger.info(f"Successfully indexed {len(chunks)} chunks")

def main():
    """Main index building script."""
    try:
        builder = VectorIndexBuilder()
        builder.build_index()
        print("Vector index built successfully!")
        print("Run 'make run' to start the API server.")
        
    except Exception as e:
        logger.error(f"Index building failed: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())