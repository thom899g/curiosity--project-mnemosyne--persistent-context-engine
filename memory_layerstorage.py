"""
Tiered Memory & Retrieval Layer
Implements L1, L2, L3 storage with hybrid search capabilities
"""
import numpy as np
from typing import Dict, List, Optional, Any
from collections import deque
import logging
from datetime import datetime, timedelta
import pickle
import os

# Conditional imports with fallbacks
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    from google.cloud.firestore_v1.base_query import FieldFilter
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    logging.warning("Firebase not available. L3 storage disabled.")

class TieredMemoryStorage:
    """Manages three-tier memory system with federated storage"""
    
    def __init__(self, firebase_credentials_path: Optional[str] = None):
        """
        Initialize tiered memory storage.
        
        Args:
            firebase_credentials_path: Path to Firebase service account JSON file
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize L1: Active Working Memory
        self.l1_buffer = deque(maxlen=20)  # Holds recent vectors + metadata
        self.l1_vectors = []  # Separate list for ANN indexing
        self.l1_metadata = []
        
        # Initialize L2: Recent Episodic Memory (7 missions)
        self.l2_cache = {}  # mission_id -> {vectors, metadata}
        self.l2_mission_history = deque(maxlen=7)
        
        # Initialize L3: Long-term Archival Memory (Firestore)
        self.l3_client = None
        self.l3_initialized = False
        
        if firebase_credentials_path and FIREBASE_AVAILABLE:
            self._initialize_firebase(firebase_credentials_path)
        
        # Initialize ANN indices (lazy loading)
        self.l1_ann_index = None
        self.l2_ann_index = None
        
        self.logger.info("✅ Tiered Memory Storage initialized")
    
    def _initialize_firebase(self, credentials_path: str):
        """Initialize Firebase connection for L3 storage"""
        try:
            if not os.path.exists(credentials_path):
                self.logger.error(f"❌ Firebase credentials not found: {credentials_path}")
                return
            
            cred = credentials.Certificate(credentials_path)
            firebase_admin.initialize_app(cred)
            self.l3_client = firestore.client()
            self.l3_initialized = True
            self.logger.info("✅ Firebase L3 storage initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Firebase initialization failed: {e}")
            self.l3_initialized = False
    
    def store_memory(self, vector: np.ndarray, metadata: Dict[str, Any]) -> str:
        """
        Store a memory across appropriate tiers.
        
        Args:
            vector: Vector embedding
            metadata: Memory metadata
            
        Returns:
            memory_id: Unique identifier for the stored memory
        """
        memory_id = metadata.get("memory_id", "unknown")
        memory_type = metadata.get("memory_type", "episodic")
        mission_id = metadata.get("mission_id", "current")
        
        # Store in L1 (always)
        self._store_l1(vector, metadata)
        
        # Store in L2 if episodic and mission is in recent 7
        if memory_type == "episodic":
            self._store_l2(vector, metadata, mission_id)
        
        # Store in L3 for all memories
        if self.l3_initialized:
            self._store_l3(vector, metadata)
        
        self.logger.debug(f"📝 Stored memory {memory_id} in tiers")
        return memory_id
    
    def _store_l1(self, vector: np.ndarray, metadata: Dict):
        """Store in L1 (Active Working Memory)"""
        self.l1_buffer.append((vector, metadata))
        self.l1_vectors.append(vector)
        self.l1_metadata.append(metadata)
        
        # Keep L1 ANN index fresh by marking for rebuild
        self.l1_ann_index = None
    
    def _store_l2(self, vector: np.ndarray, metadata: Dict, mission_id: str):
        """Store in L2 (Recent Episodic Memory)"""
        if mission_id