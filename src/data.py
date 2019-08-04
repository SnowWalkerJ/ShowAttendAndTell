import os
import os.path as path

import torch as th
from PIL import Image
from pycocotools.coco import COCO
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.nn.utils.rnn import pack_sequence

from src.config import CONFIG
from src.vocabulary import vocabulary


class CocoCaptions(VisionDataset):
    """`MS Coco Captions <http://mscoco.org/dataset/#captions-challenge2015>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    Example:

        .. code:: python

            import torchvision.datasets as dset
            import torchvision.transforms as transforms
            cap = dset.CocoCaptions(root = 'dir where images are',
                                    annFile = 'json annotation file',
                                    transform=transforms.ToTensor())

            print('Number of samples: ', len(cap))
            img, target = cap[3] # load 4th sample

            print("Image Size: ", img.size())
            print(target)

        Output: ::

            Number of samples: 82783
            Image Size: (3L, 427L, 640L)
            [u'A plane emitting smoke stream flying over a mountain.',
            u'A plane darts across a bright blue sky behind a mountain covered in snow',
            u'A plane leaves a contrail above the snowy mountain top.',
            u'A mountain that has a plane flying overheard in the distance.',
            u'A mountain view with a plume of smoke in the background']

    """

    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        super().__init__(root, transforms, transform, target_transform)
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.anns.keys()))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        record = self.coco.anns[self.ids[index]]
        target = record['caption']
        img_id = record['image_id']

        path = self.coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img_id, img, target

    def __len__(self):
        return len(self.ids)


def target_transform(x):
    return th.LongTensor(vocabulary.wrap_sentence(x))


class Padding:
    def __init__(self):
        self.fill = tuple(int(x * 256) for x in [0.485, 0.456, 0.406])

    def __call__(self, img):
        origin_size = img.width, img.height
        padding = tuple(max(225 - x, 0) // 2 for x in origin_size)
        if max(padding):
            return TF.pad(img, padding, self.fill)
        else:
            return img


def read_data(dataset, mode):
    if mode == "train":
        transform = T.Compose([
            Padding(),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = T.Compose([
            Padding(),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    ann_file, data_dir = CONFIG['coco'][dataset]
    ann_file = path.abspath(path.join("data", "annotations", ann_file))
    data_dir = path.abspath(path.join("data", data_dir))
    data = CocoCaptions(data_dir, ann_file, transform=transform, target_transform=target_transform)
    return data


def collate_fn(data):
    data.sort(key=lambda x: len(x[2]), reverse=True)
    ids, images, captions = zip(*data)

    ids = th.tensor(ids).long()
    images = th.stack(images, 0)

    targets = pack_sequence(captions)
    return ids, images, targets
