class HashTable:
    def __init__(self, table_size):
        self.size = table_size
        self.hash_table = [0 for a in range(self.size)]
    def getKey(self, data):
        self.key = ord(data[0])
        print('key_tmp = ', ord(data[0]))
        return self.key
    def hashFuntion(self, key):
        return key % self.size
    def getAddress(self, key):
        myKey = self.getKey(key)
        hash_address = self.hashFuntion(myKey)
        return hash_address
    def save(self, key, value):
        hash_address = self.getAddress(key)
        self.hash_table[hash_address] = value
    def read(self, key):
        hash_address = self.getAddress(key)
        return self.hash_table[hash_address]

    def delete(self, key):
        hash_address = self.getAddress(key)
        if self.hash_table[hash_address] != 0:
            self.hash_table[hash_address] = 0
            return True
        else:
            return False

h_table = HashTable(8)
print(ord('a'))
h_table.save('', 1111)
h_table.save('b', '3333')
h_table.save('c', '5555')
h_table.save('d', '8888')
print(h_table.hash_table)
print(h_table.read('d'))

h_table.delete('d')

print(h_table.hash_table)