from collections import defaultdict
from datetime import datetime
from statistics import mean
from random import random

import torch as th
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.rnn import pad_packed_sequence
from torch.utils.tensorboard import SummaryWriter
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

from src.data import collate_fn, read_data
from src.vocabulary import vocabulary
from src.model import Model, SoftAttention
from src.searchers import (
    BeamSearcher,
    StochasticSearcher,
    GreedySearcher,
    TeacherSearcher,
)


class Trainer:
    def __init__(self, model):
        self.device = th.device("cuda: 1")
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.decoder.parameters(), 3e-4)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.995)
        self.name = datetime.now().strftime("%Y%m%d%H%M")
        self.writer = SummaryWriter(f"./logs/{self.name}")
        self.train_loader = DataLoader(read_data("train", "train"), batch_size=32, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
        self.beam_searcher = BeamSearcher(self.model)
        self.greedy_searcher = GreedySearcher(self.model)
        self.stochastic_searcher = StochasticSearcher(self.model)
        self.teacher_searcher = TeacherSearcher(self.model)
        train_set = read_data("train", "val")
        val_set = read_data("val", "val")
        self.val_loaders = {
            "train": DataLoader(Subset(train_set, list(range(5000))), batch_size=32, num_workers=4, pin_memory=True, collate_fn=collate_fn),
            "val": DataLoader(Subset(val_set, list(range(5000))), batch_size=32, num_workers=4, pin_memory=True, collate_fn=collate_fn),
        }
        self.teacher_power = 1.0

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.train_once(epoch)
            for name, loader in self.val_loaders.items():
                self.evaluate(name, loader, epoch)
            self.save(epoch)
            self.teacher_power *= 0.97

    def train_once(self, epoch):
        self.model.train()
        for _, images, captions in tqdm(self.train_loader, desc=f"Epoch {epoch:02d}"):
            images = images.to(self.device)
            captions = captions.to(self.device)
            if random() < self.teacher_power:
                _, predicted, alpha = self.teacher_searcher.apply_batch(images, captions)
            else:
                _, predicted, alpha = self.stochastic_searcher.apply_batch(images, captions.batch_sizes.shape[0])
            loss = self.loss_fn(predicted, captions) + self.regulatization_doubly_stochastic(alpha, captions)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()

    def save(self, epoch):
        filename = f"./models/{self.name}-{epoch:03d}.pth"
        th.save(self.model.state_dict(), filename)

    @staticmethod
    def regulatization_doubly_stochastic(alpha, captions):
        _, lengths = pad_packed_sequence(captions, batch_first=True)
        loss = []
        for i, length in enumerate(lengths):
            loss.append((alpha[i, :length-1].sum(0) - ((length-1) / alpha.size(2))).pow(2))
        return th.stack(loss).mean()

    def evaluate(self, name, loader, epoch):
        self.model.eval()
        example_image = None
        PureLoss = []
        Regularization = []
        Caption = defaultdict(list)
        Output = {}
        for i, (ids, images, captions) in enumerate(tqdm(loader, desc=f"eval:{name}")):
            images = images.to(self.device)
            captions = captions.to(self.device)
            with th.no_grad():
                outputs, predicted, alpha = self.greedy_searcher.apply_batch(images, captions.batch_sizes.shape[0])
                pure_loss = self.loss_fn(predicted, captions).item()
                regularization = self.regulatization_doubly_stochastic(alpha, captions).item()
                PureLoss.append(pure_loss)
                Regularization.append(regularization)
            captions, caption_lengths = pad_packed_sequence(captions, batch_first=True)
            for image_id, output, caption, length in zip(ids, outputs, captions, caption_lengths):
                image_id = image_id.item()
                Caption[image_id].append(caption[1:length-1].tolist())
                Output[image_id] = output[:length-2].tolist()
            if epoch == 0 and i == 0:
                mu = th.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
                std = th.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
                example_image = images[1] * std + mu
        Caption, Output = self.resort(Caption, Output)
        self.writer.add_scalar(f"{name}/PureLoss", mean(PureLoss), epoch)
        self.writer.add_scalar(f"{name}/Regularization", mean(Regularization), epoch)
        bleu = self.bleu(Caption, Output)
        self.writer.add_scalar(f"{name}/BLEU", bleu, epoch)
        example_text = " ".join(map(vocabulary.idx2word.__getitem__, Output[1]))
        self.writer.add_text(f"{name}/example", example_text, epoch)
        if epoch == 0:
            example_text = " ".join(map(vocabulary.idx2word.__getitem__, Caption[1][0]))
            self.writer.add_text(f"{name}/reference", example_text, epoch)
            self.writer.add_image(f"{name}/image", example_image, epoch)

    @staticmethod
    def resort(Caption, Output):
        C, O = [], []
        for key in Caption.keys():
            C.append(Caption[key])
            O.append(Output[key])
        return C, O

    @staticmethod
    def bleu(Caption, Output):
        bleu = []
        for references, hypothesis in zip(Caption, Output):
            b = sentence_bleu(references, hypothesis)
            bleu.append(b)
        return mean(bleu)

    def loss_fn(self, predicted, target):
        target, lengths = pad_packed_sequence(target, batch_first=True)
        loss = []
        for i, length in enumerate(lengths):
            loss.append(F.nll_loss(predicted[i, :length-1], target[i, 1:length]))
        return th.stack(loss).mean()


if __name__ == "__main__":
    model = Model(512, 200, len(vocabulary), 1024, 1024, vocabulary.bgn, SoftAttention)
    model.decoder.load_pretrained_embedding("glove-twitter-200", vocabulary)
    trainer = Trainer(model)
    trainer.train(50)
