import logging
import sys
from pathlib import Path

def setup_logging(log_dir: str = "./logs"):
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/app.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create logger
    logger = logging.getLogger("llama_index_api")
    return logger 