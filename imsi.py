class Hashmap():
    def __init__(self):
        self.size = 15
        self.map = [None] * self.size
    def get_hash(self, key):
        hash_value = 0
        for char in str(key):
            hash_value += ord(char)
        return hash_value % self.size
    def add(self, key, value):
        hash_key = self.get_hash(key)
        hash_value = [key, value]

        if self.map[hash_key] is None:
            self.map[hash_key] = 
