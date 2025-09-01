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
    
    def compose_answer(self, hits: List[dict]) -> Optional[dict]:
        """
        Compose answer from search hits using extractive summarization.
        
        Args:
            hits: List of relevant search hits
            
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
            
            # Get best hit for primary answer
            top_hit = merged_hits[0]
            
            # Extract best sentences for answer (extractive)
            answer_sentences = self._extract_best_sentences(
                [hit['content'] for hit in merged_hits[:3]]
            )
            answer = ' '.join(answer_sentences)
            
            # Get excerpt from top chunk
            excerpt = self._extract_excerpt(top_hit['content'])
            
            # Format source info
            source = SourceInfo(
                title=top_hit['metadata']['title'],
                url=top_hit['metadata']['url'],
                relevance=top_hit['relevance']
            )
            
            return {
                "answer": answer,
                "excerpt": excerpt,
                "source": source,
                "fallback_used": False
            }
            
        except Exception as e:
            logger.error(f"Answer composition failed: {e}")
            return None
    
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
    
    def _extract_best_sentences(self, contents: List[str]) -> List[str]:
        """Extract best sentences for answer composition."""
        max_sentences = self.config['retrieval']['max_snippet_sentences']
        
        # Simple extractive approach: take first sentence from each top chunk
        sentences = []
        for content in contents[:max_sentences]:
            # Get first sentence
            first_sentence = content.split('.')[0].strip()
            if first_sentence and len(first_sentence) > 10:
                sentences.append(first_sentence + '.')
        
        return sentences[:max_sentences]
    
    def _extract_excerpt(self, content: str) -> str:
        """Extract excerpt from content (2-3 sentences)."""
        sentences = content.split('.')
        
        # Take first 2-3 meaningful sentences
        excerpt_sentences = []
        for sentence in sentences[:3]:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:
                excerpt_sentences.append(sentence)
        
        return '. '.join(excerpt_sentences) + '.' if excerpt_sentences else content[:200] + '...'

# Global retriever instance
_retriever = None

def get_retriever() -> LocalRetriever:
    """Get singleton retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = LocalRetriever()
    return _retriever

async def search_and_answer(question: str, top_k: int = 5) -> QuestionResponse:
    """
    Main entry point for question answering.
    
    Args:
        question: User's question
        top_k: Number of results to search
        
    Returns:
        QuestionResponse with answer or fallback
    """
    retriever = get_retriever()
    
    # Search local content
    hits = retriever.search(question, top_k)
    
    if hits:
        # Compose answer from local content
        local_response = retriever.compose_answer(hits)
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