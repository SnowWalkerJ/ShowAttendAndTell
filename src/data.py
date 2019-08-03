import os.path as path
import random

import torch as th
from torchvision.datasets import CocoCaptions
from torchvision import transforms as T

from src.config import CONFIG
from src.vocabulary import build_vocab


def read_data(dataset):
    if dataset == "train":
        transform = T.Compose([
            T.RandomClip(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = T.Compose([
            T.CenterClip(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    ann_file, data_dir = CONFIG['coco'][dataset]
    ann_file = path.abspath(path.join("data", "annotations", ann_file))
    data_dir = path.abspath(path.join("data", data_dir))
    vocabulary = build_vocab()
    target_transform = lambda x: th.LongTensor(vocabulary.wrap_sentence(random.choice(x)))
    data = CocoCaptions(data_dir, ann_file, transform=transform, target_transform=target_transform)
    data.vocabulary = vocabulary
    return data
