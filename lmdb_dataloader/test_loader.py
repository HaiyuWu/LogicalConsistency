import numpy as np
from os import path
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from glob import glob
import pandas as pd
from PIL import Image


class TestData(ImageFolder):
    def __init__(self,
                 image_path,
                 label_file,
                 attr_ids,
                 folder_paths_file,
                 im_paths_file):
        self.images = self._load_images(image_path, folder_paths_file, im_paths_file)
        self.attr_ids = attr_ids
        self.labels = None
        self.label_file = label_file
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(
                np.array([125.3, 123.0, 113.9]) / 255.0,
                np.array([63.0, 62.1, 66.7]) / 255.0),
        ])
        if self.label_file:
            self.labels = np.array(pd.read_csv(label_file))
        print(f"{len(self.images)} testing data")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        im_path = self.images[item]
        image = Image.open(im_path)
        label = False
        if self.label_file:
            label = self._find_label(self.images[item])
            if not isinstance(self.attr_ids, int):
                label = label[self.attr_ids]
        return self.transform(image), im_path, np.float32(label)

    def _find_label(self, im_path):
        im_id = path.split(im_path)[-1]
        pos = np.where(self.labels[:, 0] == im_id)[0][0]
        return self.labels[pos][1:]

    def _load_images(self, image_path, folder_paths_file, im_paths_file):
        if image_path:
            return glob(image_path + "/*")
        im_paths = []
        if folder_paths_file:
            with open(folder_paths_file, "r") as f:
                paths = f.readlines()
                for path in paths:
                    im_paths += glob(path.strip() + "/*")
        elif im_paths_file:
            with open(im_paths_file, "r") as f:
                for im_path in f.readlines():
                    im_paths.append(im_path.strip())
        else:
            raise AssertionError("Please give a folder paths file or image paths file.")
        return im_paths


class TestDataLoader(DataLoader):
    def __init__(self, config):
        self._dataset = TestData(config.test_im_path,
                                 config.test_label_file,
                                 config.attr_ids,
                                 config.folder_paths_file,
                                 config.im_paths_file)
        super(TestDataLoader, self).__init__(
            self._dataset,
            batch_size=config.batch_size,
            shuffle=False,
            pin_memory=config.pin_memory,
            num_workers=config.workers,
            drop_last=False,
        )
