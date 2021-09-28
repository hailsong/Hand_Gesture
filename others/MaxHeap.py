class MaxHeap:
    def __init__(self):
        self.queue = []
    def insert(self, n):
        # 원소 임시 추가
        self.queue.append(n)
        last_index = len(self.queue) - 1
        while 0 <= last_index