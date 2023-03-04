import shutil
import pandas as pd
import numpy as np
from os import path, makedirs
from glob import glob
import argparse


def im_name_collection(csv_label_file):
    return np.array(pd.read_csv(csv_label_file))[:, 0]


def webface_0_0_im_paths(folder_0_0_path):
    im_paths = {}
    folder_paths = glob(folder_0_0_path + "/*")
    for folder in folder_paths:
        for image in glob(folder + "/*"):
            im_name = path.split(image)[-1]
            im_paths[im_name] = image
    return im_paths


def dataset_partition(csv_partition_file):
    im_partition_dict = {}
    df = np.array(pd.read_csv(csv_partition_file))
    for data in df:
        im_partition_dict[data[0]] = data[1]
    return im_partition_dict


def dataset_collection(csv_label_file,
                       csv_partition_file,
                       folder_0_0_path,
                       celeba_dataset_folder,
                       dataset_folder):
    image_names = im_name_collection(csv_label_file)
    webface_im_paths = webface_0_0_im_paths(folder_0_0_path)
    image_partition = dataset_partition(csv_partition_file)
    sub_folders = ["train", "val", "test"]

    for image_name in image_names:
        if "_" in image_name:
            image_path = webface_im_paths[image_name]
        else:
            image_path = celeba_dataset_folder + f"{image_name}.jpg"
        partition_idx = image_partition[image_name]
        aim_folder = f"{dataset_folder}/{sub_folders[partition_idx]}"
        if not path.exists(aim_folder):
            makedirs(aim_folder)
        shutil.copy(image_path, aim_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="FH37K dataset collection."
    )
    parser.add_argument(
        "--webface_0_0_folder", "-web00", help="folder path of webface 0_0.", type=str
    )
    parser.add_argument(
        "--celeba_dataset_folder", "-celeba", help="folder path of celeba dataset folder.", type=str
    )
    parser.add_argument(
        "--partition_file", "-pf", help="dataset partition .cvs file.", type=str
    )
    parser.add_argument(
        "--dataset_folder", "-d", help="FH37K dataset saving folder.", type=str, default="./FH37K"
    )
    parser.add_argument(
        "--label_file", "-lf", help="FH37K dataset label file.", type=str
    )
    args = parser.parse_args()

    dataset_collection(args.label_file,
                       args.partition_file,
                       args.webface_0_0_folder,
                       args.celeba_dataset_folder,
                       args.dataset_folder)

