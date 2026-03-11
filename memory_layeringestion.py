"""
Ingestion & Vectorization Layer
Processes raw observations into vector embeddings with emotional context
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import hashlib

class MemoryIngestionLayer:
    """Transforms observations into vector embeddings with metadata"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the ingestion layer with embedding model.
        
        Args:
            model_name: Name of sentence-transformers model to use
        """
        self.logger = logging.getLogger(__name__)
        self.embedding_model = self._load_embedding_model(model_name)
        self.embedding_dimension = 384  # For all-MiniLM-L6-v2
        
    def _load_embedding_model(self, model_name: str):
        """Load the embedding model with error handling"""
        try:
            # Import check to ensure sentence-transformers is available
            from sentence_transformers import SentenceTransformer
            self.logger.info(f"📥 Loading embedding model: {model_name}")
            model = SentenceTransformer(model_name)
            self.logger.info("✅ Embedding model loaded successfully")
            return model
        except ImportError:
            self.logger.error("❌ sentence-transformers not installed. Install with: pip install sentence-transformers")
            raise
        except Exception as e:
            self.logger.error(f"❌ Failed to load embedding model: {e}")
            # Fallback to a simple TF-IDF if embedding model fails
            self.logger.warning("⚠️ Falling back to TF-IDF vectorizer")
            from sklearn.feature_extraction.text import TfidfVectorizer
            return TfidfVectorizer(max_features=384)
    
    def _extract_emotional_context(self, text: str) -> Tuple[str, str]:
        """
        Extract emotional context from text using simple keyword matching.
        Returns: (emotional_tag, cleaned_text)
        """
        emotional_keywords = {
            "[FRUSTRATED]": ["fail", "error", "can't", "won't", "broken", "stuck"],
            "[EXCITED]": ["success", "breakthrough", "discovered", "amazing", "perfect"],
            "[CONFIDENT]": ["solved", "optimized", "mastered", "excellent", "proven"],
            "[CURIOUS]": ["question", "why", "how", "explore", "investigate", "wonder"],
            "[STRATEGIC]": ["plan", "strategy", "approach", "method", "system"],
            "[WARNING]": ["danger", "risk", "caution", "alert", "critical"]
        }
        
        for emotion, keywords in emotional_keywords.items():
            if any(keyword in text.lower() for keyword in keywords):
                # Clean text by removing emotion markers if present
                cleaned = text.replace(emotion, "").strip()
                return emotion, cleaned
        
        return "[NEUTRAL]", text
    
    def vectorize(self, text: str, metadata: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Convert text observation into vector embedding with metadata.
        
        Args:
            text: Raw observation text
            metadata: Additional metadata about the observation
            
        Returns:
            Tuple of (vector_embedding, metadata_dict)
            
        Raises:
            ValueError: If text is empty or None
        """
        # Input validation
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        
        # Initialize metadata if not provided
        if metadata is None:
            metadata = {}
        
        try:
            # Extract emotional context
            emotion, cleaned_text = self._extract_emotional_context(text)
            
            # Prepend emotional context to text before embedding
            text_for_embedding = f"{emotion} {cleaned_text}"
            
            # Generate embedding
            if hasattr(self.embedding_model, 'encode'):
                # SentenceTransformer model
                vector = self.embedding_model.encode(text_for_embedding)
            else:
                # Fallback TF-IDF vectorizer
                vector = self.embedding_model.fit_transform([text_for_embedding]).toarray()[0]
                # Pad or truncate to 384 dimensions
                if len(vector) > self.embedding_dimension:
                    vector = vector[:self.embedding_dimension]
                else:
                    vector = np.pad(vector, (0, self.embedding_dimension - len(vector)))
            
            # Create comprehensive metadata
            memory_id = hashlib.md5(f"{text}{datetime.now().isoformat()}".encode()).hexdigest()[:16]
            
            enriched_metadata = {
                "memory_id": memory_id,
                "timestamp": datetime.now().isoformat(),
                "original_text": text,
                "cleaned_text": cleaned_text,
                "emotional_context": emotion,
                "embedding_dimension": len(vector),
                "confidence": 1.0,  # Initial confidence
                **metadata  # User-provided metadata overrides defaults if conflicts
            }
            
            # Ensure memory_type is set
            if "memory_type" not in enriched_metadata:
                enriched_metadata["memory_type"] = "episodic"  # Default
            
            self.logger.debug(f"✅ Vectorized observation with ID: {memory_id}")
            return np.array(vector, dtype=np.float32), enriched_metadata
            
        except Exception as e:
            self.logger.error(f"❌ Vectorization failed: {e}")
            # Return zero vector as fallback
            zero_vector = np.zeros(self.embedding_dimension, dtype=np.float32)
            fallback_metadata = {
                "memory_id": "error_fallback",
                "timestamp": datetime.now().isoformat(),
                "original_text": text[:100] if text else "empty",
                "error": str(e),
                "confidence": 0.1
            }
            return zero_vector, fallback_metadata
    
    def batch_vectorize(self, texts: List[str], metadata_list: Optional[List[Dict]] = None) -> List[Tuple[np.ndarray, Dict]]:
        """Vectorize multiple observations efficiently"""
        results = []
        metadata_list = metadata_list or [{}] * len(texts)
        
        for i, (text, metadata) in enumerate(zip(texts, metadata_list)):
            try:
                result = self.vectorize(text, metadata)
                results.append(result)
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to vectorize item {i}: {e}")
                # Add fallback
                zero_vector = np.zeros(self.embedding_dimension, dtype=np.float32)
                fallback_meta = {"error": str(e), "index": i}
                results.append((zero_vector, fallback_meta))
        
        return results