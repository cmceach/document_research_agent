import os
import logging
from typing import List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_required_env_vars(required_vars: List[str]) -> bool:
    """
    Check if all required environment variables are set.
    
    Args:
        required_vars: List of required environment variable names
        
    Returns:
        Boolean indicating whether all required environment variables are set
    """
    missing = [var for var in required_vars if not os.environ.get(var)]
    
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        logger.error("Please set these variables in your .env file or environment.")
        return False
    
    logger.debug("All required environment variables are set")
    return True 