import os
from os import path
import lmdb
import msgpack
from torch.utils.data import DataLoader
from glob import glob
from torchvision.datasets import ImageFolder
import numpy as np
import pandas as pd
from tqdm import tqdm


class AttributesDatasetBase(ImageFolder):
    def __init__(self, image_path, label_path):
        self.images = glob(image_path + "/*")
        self.labels = np.array(pd.read_csv(label_path))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        with open(self.images[item], "rb") as f:
            image = f.read()
        label = self._find_label(self.images[item])
        return image, label

    def _find_label(self, im_path):
        im_id = path.split(im_path)[-1]
        pos = np.where(self.labels[:, 0] == im_id)[0][0]
        return self.labels[pos][1:]


class CustomRawLoader(DataLoader):
    def __init__(self, image_path, label_path, workers, task):
        self._dataset = AttributesDatasetBase(image_path, label_path)
        shuffle = False
        if task == "train":
            print("Shuffle data...")
            shuffle = True
        super(CustomRawLoader, self).__init__(
            self._dataset, num_workers=workers, collate_fn=lambda x: x, shuffle=shuffle
        )


def list2lmdb(
    image_path,
    label_path,
    dest,
    task="test",
    num_workers=16,
    write_frequency=5000,
):
    print("Loading dataset from %s" % image_path)
    data_loader = CustomRawLoader(
        image_path, label_path, num_workers, task
    )
    if not path.exists(dest):
        os.makedirs(dest)
    name = f"{image_path.split('/')[2]}.lmdb"
    lmdb_path = path.join(dest, name)
    isdir = path.isdir(lmdb_path)
    print(f"Generate LMDB to {lmdb_path}")

    image_w = 178
    image_h = 218
    size = len(data_loader.dataset) * image_w * image_h * 3

    print(f"LMDB max size: {size}")

    db = lmdb.open(
        lmdb_path,
        subdir=isdir,
        map_size=size * 2,
        readonly=False,
        meminit=False,
        map_async=True,
    )

    print(len(data_loader.dataset))
    txn = db.begin(write=True)
    for idx, data in tqdm(enumerate(data_loader)):
        image, label = data[0]
        # txn.put(
        #     "{}".format(idx).encode("ascii"), msgpack.packb((image, label), default=m.encode)
        # )
        txn.put(
            "{}".format(idx).encode("ascii"), msgpack.dumps((image, list(label)))
        )
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)
    idx += 1

    # finish iterating through dataset
    txn.commit()
    keys = ["{}".format(k).encode("ascii") for k in range(idx)]
    with db.begin(write=True) as txn:
        txn.put(b"__keys__", msgpack.dumps(keys))
        txn.put(b"__len__", msgpack.dumps(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", "-im", help="List of images.")
    parser.add_argument("--label_path", "-l", help="label file.")
    parser.add_argument("--task", "-t", help="train/val/test.")
    parser.add_argument("--workers", "-w", help="Workers number.", default=16, type=int)
    parser.add_argument("--dest", "-d", help="Path to save the lmdb file.")

    args = parser.parse_args()

    list2lmdb(
        args.image_path,
        args.label_path,
        args.dest,
        args.task,
        args.workers,
        write_frequency=5000
    )
