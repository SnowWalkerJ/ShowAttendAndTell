from collections import defaultdict
from datetime import datetime
from statistics import mean

import torch as th
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_packed_sequence
from torch.utils.tensorboard import SummaryWriter
from nltk.translate.bleu_score import corpus_bleu
from tqdm import trange

from src.data import collate_fn, read_data
from src.vocabulary import vocabulary
from src.model import Model, SoftAttention
from src.searchers import BeamSearcher, StochasticSearcher


class Trainer:
    def __init__(self, model):
        self.device = th.device("cuda: 1")
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.decoder.parameters(), lr=3e-4)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.995)
        self.name = datetime.now().strftime("%Y%m%d%H%M")
        self.writer = SummaryWriter(f"./logs/{self.name}")
        self.train_loader = DataLoader(read_data("train", "train"), batch_size=64, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
        self.beam_searcher = BeamSearcher(self.model)
        self.stochastic_searcher = StochasticSearcher(self.model)
        self.val_loaders = {
            "train": DataLoader(read_data("train", "val"), batch_size=90, num_workers=4, pin_memory=True, collate_fn=collate_fn),
            "val": DataLoader(read_data("val", "val"), batch_size=90, num_workers=4, pin_memory=True, collate_fn=collate_fn),
        }

    def train(self, num_epochs):
        for epoch in trange(num_epochs):
            self.train_once()
            for name, loader in self.val_loaders.items():
                self.evaluate(name, loader, epoch)
            self.save(epoch)

    def train_once(self):
        self.model.train()
        self.model.searcher = self.stochastic_searcher
        self.scheduler.step()
        for _, images, captions in self.train_loader:
            images = images.to(self.device)
            captions = captions.to(self.device)
            _, predicted, alpha = self.model(images, captions.batch_sizes.shape[0])
            loss = self.loss_fn(predicted, captions) # + (alpha.sum(1) - 1).pow(2).mean() * 0.01
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def save(self, epoch):
        filename = f"./models/{self.name}-{epoch:03d}.pth"
        th.save(self.model.state_dict(), filename)

    def evaluate(self, name, loader, epoch):
        self.model.eval()
        # self.model.searcher = self.beam_searcher
        self.model.searcher = self.stochastic_searcher
        PureLoss = []
        Regularization = []
        Caption = defaultdict(list)
        Output = {}
        for i, (ids, images, captions) in enumerate(loader):
            images = images.to(self.device)
            captions = captions.to(self.device)
            with th.no_grad():
                outputs, predicted, alpha = self.model(images, captions.batch_sizes.shape[0])
                pure_loss = self.loss_fn(predicted, captions).item()
                regularization = (alpha.sum(1) - 1).pow(2).mean().item()
                PureLoss.append(pure_loss)
                Regularization.append(regularization)
            captions, caption_lengths = pad_packed_sequence(captions)
            for image_id, output, caption, length in zip(ids, outputs, captions.t(), caption_lengths):
                Caption[image_id].append(caption[1:length-1].tolist())
                Output[image_id] = output[:length-2].tolist()
        self.writer.add_scalar(f"{name}/PureLoss", mean(PureLoss), epoch)
        self.writer.add_scalar(f"{name}/Regularization", mean(Regularization), epoch)
        references, hypotheses = zip(*((Caption[i], Output[i]) for i in Caption.keys()))
        bleu = corpus_bleu(list(references), list(hypotheses))
        self.writer.add_scalar(f"{name}/BLEU", bleu, epoch)
        example_text = " ".join(map(vocabulary.idx2word.__getitem__, hypotheses[0]))
        self.writer.add_text(f"{name}/example", example_text, epoch)
        example_text = " ".join(map(vocabulary.idx2word.__getitem__, references[0][0]))
        self.writer.add_text(f"{name}/reference", example_text, epoch)

    def loss_fn(self, predicted, target):
        target, lengths = pad_packed_sequence(target)
        target = target.t()
        loss = []
        for i, length in enumerate(lengths):
            loss.append(F.nll_loss(predicted[i, :length-1], target[i, 1:length]))
        return th.stack(loss).mean()


if __name__ == "__main__":
    model = Model(512, 512, len(vocabulary), 512, vocabulary.bgn, SoftAttention(), None)
    trainer = Trainer(model)
    trainer.train(120)
