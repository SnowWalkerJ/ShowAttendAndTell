import torch as th
from torch.distributions import Categorical


class StochasticSearcher:
    def __init__(self, model):
        self.model = model

    def apply_batch(self, inputs, max_depth):
        model, hidden, word = self.model.partial(inputs)
        outputs = []
        for _ in range(max_depth):
            (hidden, logprobs, *others) = model(hidden, word)
            dist = Categorical(th.exp(logprobs))
            word = dist.sample()
            outputs.append((word, logprobs, *others))
        return map(lambda x: th.stack(x, 1), zip(*outputs))
