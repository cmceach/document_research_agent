from typing import Any, Optional, Callable
from functools import wraps
import time
import logging
from collections import OrderedDict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LRUCache:
    """Least Recently Used (LRU) cache implementation."""
    
    def __init__(self, capacity: int = 1000, ttl: int = 3600):
        """
        Initialize LRU cache.
        
        Args:
            capacity: Maximum number of items to store
            ttl: Time to live in seconds for cached items
        """
        self.capacity = capacity
        self.ttl = ttl
        self.cache = OrderedDict()
        self.timestamps = {}
        self.custom_ttls = {}  # Store custom TTLs for specific keys
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache if it exists and hasn't expired."""
        if key not in self.cache:
            return None
            
        # Check if item has expired using its specific TTL
        ttl = self.custom_ttls.get(key, self.ttl)
        if time.time() - self.timestamps[key] > ttl:
            self.remove(key)
            return None
            
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Add item to cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional custom TTL for this specific item
        """
        if key in self.cache:
            # Move to end and update
            self.cache.move_to_end(key)
        else:
            # Check capacity
            if len(self.cache) >= self.capacity:
                # Remove least recently used item
                removed_key, _ = self.cache.popitem(last=False)
                if removed_key in self.custom_ttls:
                    del self.custom_ttls[removed_key]
                
        self.cache[key] = value
        self.timestamps[key] = time.time()
        if ttl is not None:
            self.custom_ttls[key] = ttl
    
    def remove(self, key: str):
        """Remove item from cache."""
        if key in self.cache:
            del self.cache[key]
            del self.timestamps[key]
            if key in self.custom_ttls:
                del self.custom_ttls[key]
    
    def clear(self):
        """Clear all items from cache."""
        self.cache.clear()
        self.timestamps.clear()
        self.custom_ttls.clear()

# Global cache instance
_global_cache = LRUCache()

def cached(ttl: Optional[int] = None):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Optional override for cache TTL
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Check cache
            result = _global_cache.get(key)
            if result is not None:
                logger.debug(f"Cache hit for {key}")
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            _global_cache.put(key, result, ttl)
            
            logger.debug(f"Cached result for {key}")
            return result
        return wrapper
    return decorator

def clear_cache():
    """Clear the global cache."""
    _global_cache.clear()
    logger.info("Global cache cleared") 