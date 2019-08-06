import torch as th
from torch.nn.utils.rnn import pad_packed_sequence

from .base import Searcher


class TeacherSearcher(Searcher):
    def __init__(self, model):
        self.model = model

    def apply_batch(self, inputs, captions):
        model, hidden, word = self.model.partial(inputs)
        outputs = []
        captions, lengths = pad_packed_sequence(captions, batch_first=True)
        for i in range(max(lengths)):
            word = captions[:, i]
            (hidden, logprobs, *others) = model(hidden, word)
            outputs.append((word, logprobs, *others))
        return map(lambda x: th.stack(x, 1), zip(*outputs))
