"""
Project Mnemosyne - Persistent Context Engine v2.0
Lightweight vector-based memory layer for the Evolution Ecosystem
"""
from .ingestion import MemoryIngestionLayer
from .storage import TieredMemoryStorage
from .daemon import GenerativeDaemon
from .integration import MemoryIntegrationLayer

__version__ = "2.0.0"
__all__ = [
    "MemoryIngestionLayer",
    "TieredMemoryStorage", 
    "GenerativeDaemon",
    "MemoryIntegrationLayer",
    "MemorySystem"
]

class MemorySystem:
    """Orchestrates all Mnemosyne components as a unified system"""
    
    def __init__(self, firebase_credentials_path: str = None):
        """
        Initialize the complete Mnemosyne memory system.
        
        Args:
            firebase_credentials_path: Path to Firebase service account JSON file
        """
        self.logger = self._setup_logger()
        self.logger.info("🚀 Initializing Project Mnemosyne v2.0")
        
        # Initialize layers in dependency order
        self.ingestion_layer = MemoryIngestionLayer()
        self.storage_layer = TieredMemoryStorage(firebase_credentials_path)
        self.daemon = GenerativeDaemon(self.storage_layer)
        self.integration_layer = MemoryIntegrationLayer(
            self.ingestion_layer,
            self.storage_layer,
            self.daemon
        )
        
        self.logger.info("✅ Project Mnemosyne initialized successfully")
    
    def _setup_logger(self):
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def process_observation(self, text: str, metadata: dict = None) -> str:
        """Main entry point for processing new observations"""
        return self.integration_layer.process_observation(text, metadata)
    
    def query_memories(self, query: str, memory_type: str = None, limit: int = 5) -> list:
        """Query memories with optional filtering"""
        return self.storage_layer.retrieve_memories(query, memory_type, limit)