from abc import ABC, abstractmethod


class Searcher(ABC):
    @abstractmethod
    def apply_batch(self, input, max_depth):
        pass
