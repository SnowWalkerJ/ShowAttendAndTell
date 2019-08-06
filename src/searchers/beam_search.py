from collections import namedtuple
from itertools import chain

from lazy_property import LazyProperty
import torch as th
from src.config import CONFIG
from .base import Searcher


class Node:
    def __init__(self, hidden, word, logprobs=None, logprob=0, parent=None, *others):
        self.hidden = hidden
        self.word = word
        self.parent = parent
        self.logprobs = logprobs
        self.logprob = logprob
        self.others = others

    def generate(self, model, width):
        (hidden, logprobs, *others) = model(self.hidden, self.word)
        top_logprob, idx = th.topk(logprobs, width, 1)
        for i in range(width):
            logprob = top_logprob[:, i]
            word = idx[:, i]
            yield Node(hidden, word, logprobs, logprob, self, *others)

    def iter(self):
        node = self
        while node.parent:
            yield node
            node = node.parent

    @LazyProperty
    def cumlogprob(self):
        return self.logprob + (self.parent.logprob if self.parent else 0)

    def aggregate(self):
        logprobs = th.stack(tuple(reversed([node.logprobs for node in self.iter()])), 1)
        outputs = th.stack(tuple(reversed([node.word for node in self.iter()])), 1)
        others = map(lambda x: th.stack(x, 1), zip(*reversed([node.others for node in self.iter()])))
        return (outputs, logprobs, *others)


class BeamSearcher(Searcher):
    def __init__(self, model, beam_width=3):
        self.beam_width = beam_width
        self.model = model

    def apply(self, model, init_hidden, init_word, max_depth):
        """
        init_hidden: B x C
        init_word: B x 1
        """
        nodes = [Node(init_hidden, init_word)]
        for depth in range(max_depth):
            nodes = sorted(chain(*(node.generate(model, self.beam_width) for node in nodes)), key=lambda x: x.cumlogprob)[:self.beam_width]
        best_node = max(nodes, key=lambda x: x.cumlogprob)
        return best_node.aggregate()

    def apply_batch(self, input, max_depth):
        batch_size = input.size(0)
        outputs, logprobs, others = [], [], []
        for i in range(batch_size):
            _input = input[i:i+1]
            model, hidden, word = self.model.partial(_input)
            (_outputs, _logprobs, *_others) = self.apply(model, hidden, word, max_depth)
            outputs.append(_outputs)
            logprobs.append(_logprobs)
            others.append(_others)
        outputs = th.cat(outputs, 0)
        logprobs = th.cat(logprobs, 0)
        others = map(lambda x: th.cat(x, 0), zip(*others))
        return (outputs, logprobs, *others)
