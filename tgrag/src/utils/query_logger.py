import os
import json
from datetime import datetime
from .logging import get_logger

logger = get_logger(__name__)

class QueryLogger:
    """Logs TG-RAG queries and responses to a JSONL file."""
    
    def __init__(self, working_dir: str):
        self.working_dir = working_dir
        self.log_file = os.path.join(working_dir, "queries.jsonl")
        os.makedirs(working_dir, exist_ok=True)
        logger.info(f"Initialized QueryLogger at {self.log_file}")

    def log(self, query: str, response: str, retrieval_detail: dict = None, mode: str = "local"):
        """Log a single query interaction.
        
        Args:
            query: The user query string
            response: The LLM response string
            retrieval_detail: Optional dictionary containing retrieval metadata (PPR results, etc.)
            mode: Query mode used (local, global, naive)
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "query": query,
            "response": response,
            "retrieval_detail": retrieval_detail
        }
        
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to query log: {e}")
