from collections import Counter
import os
import pickle

from pycocotools.coco import COCO
from nltk.tokenize import word_tokenize
from lazy_object_proxy import Proxy as lazy

from src.config import CONFIG


class Vocabulary:
    def __init__(self, dataset, minimum_occurance=3):
        self.idx2word = {}
        self.word2idx = {}
        self.i = 0
        self.pad = self.add_word("<PAD>")
        self.unk = self.add_word("<UNK>")
        self.bgn = self.add_word("<BGN>")
        self.end = self.add_word("<END>")
        ann_file = os.path.abspath(os.path.join("data", "annotations", CONFIG['coco'][dataset][0]))
        counter = self.__build_counter(COCO(ann_file))
        for word, count in counter.items():
            if count < minimum_occurance:
                continue
            self.add_word(word)

    def add_word(self, word):
        self.idx2word[self.i] = word
        self.word2idx[word] = self.i
        self.i += 1
        return self.i - 1

    @staticmethod
    def __build_counter(coco):
        counter = Counter()
        for img_id in coco.imgs.keys():
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            for ann in anns:
                caption = ann['caption']
                words = [x for x in word_tokenize(caption.lower()) if x.isalpha()]
                assert all(x.islower() for x in words)
                counter.update(words)
        return counter

    def wrap_sentence(self, sentence):
        idxs = [self.bgn]
        sentence = [x for x in word_tokenize(sentence.lower()) if x.isalpha()]
        for word in sentence:
            idxs.append(self(word))
        idxs.append(self.end)
        return idxs

    def __call__(self, word):
        return self.word2idx.get(word, self.unk)

    def __getitem__(self, idx):
        return self.idx2word.get(idx)

    def __len__(self):
        return len(self.idx2word)


def build_vocab(minimum_occurance=3):
    dataset = "train"
    if not os.path.exists(".cache"):
        os.mkdir(".cache")
    filename = f".cache/vocab_{dataset}_{minimum_occurance}.pkl"
    try:
        with open(filename, "rb") as f:
            vocab = pickle.load(f)
    except Exception:
        vocab = None
    if not vocab:
        print("building vocabulary...")
        vocab = Vocabulary(dataset, minimum_occurance)
        with open(filename, "wb") as f:
            pickle.dump(vocab, f)
        print("vocabulary built.")
    return vocab


vocabulary = lazy(build_vocab)
