import numpy as np


class DataProcessor:
    def __init__(self):
        self.data = np.array([])

    def append(self, item, times=1):
        for _ in range(times):
            self.data = np.append(self.data, item)
        return

    def avg(self, p=0):
        if self.is_empty():
            return -1
        return np.mean(self.data[p:])

    def std(self, p=0):
        return -1 if self.is_empty() else np.std(self.data[p:])

    def min(self, p=0):
        return np.min(self.data[p:])

    def max(self, p=0):
        return np.max(self.data[p:])

    def last(self):
        return -1 if self.is_empty() else self.data[-1]

    def clear(self):
        self.data = []

    def is_empty(self):
        return self.data.size == 0