from os import path
import lmdb
import msgpack
import numpy as np
import six
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class LMDB(Dataset):
    def __init__(self, db_path, attr_ids):
        self.db_path = db_path
        self.attr_ids = attr_ids
        self.env = lmdb.open(
            db_path,
            subdir=path.isdir(db_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        with self.env.begin(write=False) as txn:
            self.length = msgpack.loads(txn.get(b"__len__"))
            self.keys = msgpack.loads(txn.get(b"__keys__"))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                np.array([125.3, 123.0, 113.9]) / 255.0,
                np.array([63.0, 62.1, 66.7]) / 255.0),
        ])
        print(f"{self.length} data")

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = msgpack.loads(byteflow)
        # load image
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

        # load label
        target = np.float32(unpacked[1])
        if not isinstance(self.attr_ids, int):
            target = target[self.attr_ids]
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return self.length


class LMDBDataLoader(DataLoader):
    def __init__(self, config, lmdb_path, train=True):

        self._dataset = LMDB(lmdb_path, config.attr_ids)

        batch_size = config.batch_size

        super(LMDBDataLoader, self).__init__(
            self._dataset,
            batch_size=batch_size,
            shuffle=train,
            pin_memory=config.pin_memory,
            num_workers=config.workers,
            drop_last=train,
        )
