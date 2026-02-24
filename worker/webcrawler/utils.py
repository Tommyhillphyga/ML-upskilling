import queue

class CheckableQueue(queue.Queue):
    def __contains__(self, item):
        with self.mutex:
            return item in self.queue
        
    def __len__ (self):
        return len(self.queue)



