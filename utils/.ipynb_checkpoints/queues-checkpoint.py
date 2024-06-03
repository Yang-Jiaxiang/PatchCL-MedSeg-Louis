import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

class Embedding_Queues():
    def __init__(self, num_classes, max_length=4096):
        self.num_classes = num_classes
        self.max_length = max_length
        self.queues = [deque(maxlen=max_length) for _ in range(self.num_classes)]

    def __getitem__(self, idx):
        return list(self.queues[idx])
    
    def __len__(self):
        return len(self.queues)
    
    def enqueue(self, new_embeddings):
        def extend_queue(queue, new_embedding):
            if new_embedding is not None:
                start_time = time.time()
                queue.extend(new_embedding)
                end_time = time.time()
                
        with ThreadPoolExecutor(max_workers=self.num_classes) as executor:
            executor.map(extend_queue, self.queues, new_embeddings)