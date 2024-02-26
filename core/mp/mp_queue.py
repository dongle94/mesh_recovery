import queue
from multiprocessing import Queue, Lock


class MessageQueue(object):
    def __init__(self, q_size=0):
        self.queue = Queue(maxsize=q_size)
        self.max_size = q_size
        self.lock = Lock()
        self.is_closed = False

    def get(self, block=True, timeout=0.01):
        """
        If block is false, it works same with get_nowait()
        -> if it can't bring data, occur queue.Empty Error
        If block is True and timeout is None, wait infinite time until get data.

        Args:
            block (bool): bool of lock
            timeout (float): waiting timeout

        Returns:
            data (any): queue return data
        """
        if self.is_closed is True:
            raise BrokenPipeError("Queue is Closed!")

        if self.queue.empty() is False:
            data = self.queue.get(block=block, timeout=timeout)
            return data
        else:
            raise queue.Empty

    def put(self, data, block=True, timeout=0.01):
        """
        Insert data into data Queue

        Args:
            data (any):
            block (bool):
            timeout (float):

        Returns:
            -
        """
        if self.is_closed is True:
            raise BrokenPipeError("Queue is Closed!")

        if self.queue.full() is False:
            self.queue.put(obj=data, block=block, timeout=timeout)
        else:
            raise queue.Full

    def is_empty(self):
        return self.queue.empty()

    @property
    def max_size(self):
        return self._max_size

    @max_size.setter
    def max_size(self, size):
        self._max_size = size

    def queue_size(self):
        return self.queue.qsize()

    def close(self):
        self.queue.close()
        self.is_closed = True

    @staticmethod
    def Full():
        return queue.Full
