from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from os import path


def main(label_file,
         image_file):
    # read labels
    labels = np.array(pd.read_csv(label_file))
    duplicate_dict = {}
    for im_names in tqdm(labels[:, 0]):
        if "_" in im_names:
            duplicate_dict[im_names] = 0
    # read binary predictions
    im_paths = np.asarray(pd.read_csv(image_file, header=None)).squeeze()
    im_dict = {}
    for im_path in tqdm(im_paths):
        im_id = path.split(im_path)[1]
        im_dict[im_id] = im_path
    update = []
    for im_id, predict in tqdm(im_dict.items()):
        try:
            a = duplicate_dict[im_id]
        except KeyError:
            update.append(predict)
    np.savetxt(image_file, update, fmt="%s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Filter out the satisfied images"
    )
    parser.add_argument(
        "--image_file", "-i",
        help="The .txt file that stores the image paths of the first 30,000 identities in WF260M.", type=str
    )
    parser.add_argument(
        "--label_file", "-l",
        help="The ground truth label file.", type=str, default="./FH37K/facial_hair_annotations.csv"
    )
    args = parser.parse_args()
    main(args.image_file,
         args.image_file)
