import time
from typing import Optional

class SimpleCache:
    """A tiny in-memory cache for demo purposes."""
    def __init__(self, ttl: int = 300):
        self.ttl = ttl
        self.store = {}  # key -> (value, expiry)

    def set(self, key: str, value: str):
        expiry = time.time() + self.ttl
        self.store[key] = (value, expiry)

    def get(self, key: str) -> Optional[str]:
        v = self.store.get(key)
        if not v:
            return None
        value, expiry = v
        if time.time() > expiry:
            del self.store[key]
            return None
        return value
