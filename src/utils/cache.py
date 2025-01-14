class Cache:
    _instance = None
    
    def __init__(self):
        self._cache = {}
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = Cache()
        return cls._instance
    
    def set(self, key, value):
        self._cache[key] = value
    
    def get(self, key):
        return self._cache.get(key)