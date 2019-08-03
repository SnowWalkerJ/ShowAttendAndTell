from datetime import datetime

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss, 
from ignite.handlers import ModelCheckpoint
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from data import read_data
from searchers import BeamSearcher, StochasticSearcher


class Trainer:
    def __init__(self, model):
        self.model = model
        self.loss_fn = nn.NLLLoss()
        self.train_data = read_data("train")
        self.val_data = read_data("val")
        self.optimizer = optim.Adam(model.decoder.parameters(), lr=3e-4)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.995)
        self.device = th.device("cuda: 0")
        self.beam_searcher = BeamSearcher(self.model)
        self.stochastic_searcher = StochasticSearcher(self.model)
        self.trainer = create_supervised_trainer(
            self.model,
            self.optimizer,
            self.loss_fn,
            device=self.device,
            non_blocking=True,
        )
        self.evaluator = create_supervised_evaluator(self.model, metrics={
            "loss": Loss(self.loss_fn),
        }, device=device, non_blocking=True)
        time = datetime.now().strftime("%Y%m%d%H%M")
        self.writer = SummaryWriter(logdir=f"./logs/{time}")
        self.saver = ModelCheckpoint("./models", save_interval=5)
        self.register_events()

    def register_events(self):
        self.trainer.on(Events.EPOCH_STARTED)(self.set_searcher(self.stochastic_searcher))
        self.trainer.on(Events.EPOCH_COMPLETED)(self.set_searcher(self.beam_searcher))
        self.trainer.on(Events.EPOCH_COMPLETED)(self.evaluate(self.train_data, "train"))
        self.trainer.on(Events.EPOCH_COMPLETED)(self.evaluate(self.val_data, "val"))
        self.trainer.on(Events.EPOCH_COMPLETED)(self.example_string)
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.saver, {"model": self.model})

    def evaluate(self, dataset, name):
        loader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)

        def _evaluate(trainer):
            self.model.eval()
            epoch = trainer.state.epoch
            evaluator.run(loader)
            metrics = evaluator.state.metrics
            loss = metrics['loss']
            self.writer.add_scalar(f"{name}/loss", loss, epoch)

        return _evaluate

    def example_string(self, trainer):
        self.model.eval()
        epoch = trainer.state.epoch
        x, y = self.val_data[0]
        words, _ = self.model(x.unsqueeze(0))
        words = words.cpu().numpy()
        vocab = self.train_data.vocabulary
        sentence = " ".join(vocab[idx] for idx in words)
        self.writer.add_text("sample", sentence, epoch)

    def set_searcher(self, searcher):
        def fn():
            self.model.searcher = searcher
        return fn

    def run(self):
        self.trainer.run(DataLoader(self.train_data, batch_size=32, shuffle=True, pin_memory=True, num_workers=4))


if __name__ == "__main__":
    model = Model
