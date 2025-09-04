"""
Local retrieval system for blog Q&A using manual vector index.
"""

import logging
from typing import List, Optional
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from .models import QuestionResponse, SourceInfo
from .config import get_config, get_storage_path
from .llm_providers import get_llm_provider

logger = logging.getLogger(__name__)

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class LocalRetriever:
    """Handles local blog search using manual vector index."""
    
    def __init__(self):
        self.config = get_config()
        self.storage_path = get_storage_path()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index_data = None
        self._load_index()
    
    def _load_index(self):
        """Load manual vector index from pickle file."""
        try:
            index_path = self.storage_path / "manual_vector_index.pkl"
            if index_path.exists():
                with open(index_path, 'rb') as f:
                    self.index_data = pickle.load(f)
                logger.info(f"Loaded manual index with {len(self.index_data['chunks'])} chunks")
            else:
                logger.warning(f"Manual index not found at {index_path}")
                self.index_data = None
        except Exception as e:
            logger.warning(f"Failed to load manual index: {e}")
            self.index_data = None
    
    def search(self, question: str, top_k: int = 5) -> List[dict]:
        """
        Search local blog content for relevant chunks.
        
        Args:
            question: User's question
            top_k: Number of results to return
            
        Returns:
            List of search hit dictionaries with relevance scores
        """
        if not self.index_data:
            logger.warning("Manual index not available, returning empty results")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([question])[0]
            
            # Calculate similarities with all chunks
            chunks = self.index_data['chunks']
            embeddings = self.index_data['embeddings']
            
            similarities = []
            for i, chunk_embedding in enumerate(embeddings):
                similarity = cosine_similarity(query_embedding, chunk_embedding)
                similarities.append((similarity, i))
            
            # Sort by similarity and take top_k
            similarities.sort(reverse=True)
            top_similarities = similarities[:top_k * 2]  # Get more for filtering
            
            hits = []
            min_relevance = self.config['retrieval']['min_local_relevance']
            
            for similarity, chunk_idx in top_similarities:
                if similarity >= min_relevance:
                    chunk = chunks[chunk_idx]
                    hits.append({
                        'content': chunk['content'],
                        'metadata': {
                            'source_id': chunk['source_id'],
                            'title': chunk['title'],
                            'url': chunk['url'],
                            'chunk_index': chunk['chunk_index']
                        },
                        'relevance': similarity
                    })
                
                if len(hits) >= top_k:
                    break
            
            logger.info(f"Found {len(hits)} relevant chunks for question")
            return hits
            
        except Exception as e:
            logger.error(f"Local search failed: {e}")
            return []
    
    async def compose_answer(self, hits: List[dict], question: str, provider: str = "gemini") -> Optional[dict]:
        """
        Compose answer from search hits using LLM generation.
        
        Args:
            hits: List of relevant search hits
            question: Original user question
            provider: LLM provider to use (gemini/openai)
            
        Returns:
            Formatted response dict or None if no good hits
        """
        if not hits:
            return None
        
        try:
            # Merge chunks from same blog post
            merged_hits = self._merge_same_source_hits(hits)
            
            if not merged_hits:
                return None
            
            # Get best hit for LLM context
            top_hit = merged_hits[0]
            best_content = top_hit['content']
            
            # Collect all blog URLs for citations
            blog_urls = []
            for hit in merged_hits:
                blog_info = {
                    'title': hit['metadata']['title'],
                    'url': hit['metadata']['url']
                }
                # Avoid duplicate URLs
                if not any(existing['url'] == blog_info['url'] for existing in blog_urls):
                    blog_urls.append(blog_info)
            
            # Generate answer using LLM
            llm_provider = get_llm_provider(provider)
            generated_answer = await llm_provider.generate_answer(question, best_content, blog_urls)
            
            # Format source info (primary source)
            source = SourceInfo(
                title=top_hit['metadata']['title'],
                url=top_hit['metadata']['url'],
                relevance=top_hit['relevance']
            )
            
            return {
                "answer": generated_answer,
                "excerpt": best_content[:200] + "..." if len(best_content) > 200 else best_content,
                "source": source,
                "fallback_used": False
            }
            
        except Exception as e:
            logger.error(f"Answer composition failed: {e}")
            return {
                "answer": f"Sorry, I encountered an error generating the response: {str(e)}",
                "excerpt": None,
                "source": None,
                "fallback_used": True
            }
    
    def _merge_same_source_hits(self, hits: List[dict]) -> List[dict]:
        """Merge chunks from the same blog post."""
        source_groups = {}
        
        for hit in hits:
            source_id = hit['metadata'].get('source_id')
            if source_id not in source_groups:
                source_groups[source_id] = []
            source_groups[source_id].append(hit)
        
        merged_hits = []
        for source_id, group in source_groups.items():
            if len(group) == 1:
                merged_hits.append(group[0])
            else:
                # Merge multiple chunks from same source
                merged_content = ' '.join([hit['content'] for hit in group])
                merged_hit = {
                    'content': merged_content,
                    'metadata': group[0]['metadata'],  # Use first chunk's metadata
                    'relevance': max(hit['relevance'] for hit in group)  # Best relevance
                }
                merged_hits.append(merged_hit)
        
        # Sort by relevance
        merged_hits.sort(key=lambda x: x['relevance'], reverse=True)
        return merged_hits
    

# Global retriever instance
_retriever = None

def get_retriever() -> LocalRetriever:
    """Get singleton retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = LocalRetriever()
    return _retriever

async def search_and_answer(question: str, provider: str = "gemini") -> QuestionResponse:
    """
    Main entry point for question answering.
    
    Args:
        question: User's question
        top_k: Number of results to search
        
    Returns:
        QuestionResponse with answer or fallback
    """
    retriever = get_retriever()
    
    # Search local content (fixed top_k = 3)
    hits = retriever.search(question, top_k=3)
    
    if hits:
        # Compose answer from local content using LLM
        local_response = await retriever.compose_answer(hits, question, provider)
        if local_response:
            return QuestionResponse(**local_response)
    
    # TODO: Implement web fallback in Phase 3
    # TODO: Implement guardrails in Phase 3
    
    # Check if index is still building
    retriever = get_retriever()
    if not retriever.index_data:
        return QuestionResponse(
            answer="The blog search index is still building. Please try again in a few minutes.",
            fallback_used=False,
            policy_reason="Index not ready"
        )
    
    # For now, return no results found
    return QuestionResponse(
        answer="I couldn't find relevant information in the blog content for your question.",
        fallback_used=False
    )