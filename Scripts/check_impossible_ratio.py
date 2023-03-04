import argparse
import numpy as np
import os.path as path
import os
from tqdm import tqdm
import torch


def main(file_path,
         ):
    print(f"File path: {file_path}\n")
    saved_im_paths = {}
    print("Start confidence collection...")
    total_number = 0
    failed = 0
    incomplete = 0
    impossible = 0
    with open(file_path, "r") as f:
        for line in tqdm(f.readlines()):
            temp_confidences = []
            im_path, confidences = line.strip().split("\t")
            try:
                for confidence in confidences.split(","):
                    temp_confidences.append(int(confidence))
            except Exception:
                for confidence in confidences[1:-1].split(","):
                    temp_confidences.append(int(confidence))
            total_number += 1
            failed_number, incomplete_number, impossible_number = condition_checking(temp_confidences)
            failed += failed_number
            incomplete += incomplete_number
            impossible += impossible_number
    print(f"Total number: {total_number}\nFailed number: {failed}\nFailed ratio: {round(failed / total_number, 4) * 100}\nIncomplete number: {incomplete}\nImpossible number: {impossible}")


def condition_checking(confidences):
    length = [0, 4, 9, 13, 17, 22]
    fail = 0
    incomplete = 0
    impossible = 0

    # incomplete cases
    # 1. no results in Beard area, Mustache, Sideburns, or Bald
    # 2. not clean shaven but no results in Beard length
    if sum(confidences[length[0]:length[1]]) == 0 or sum(confidences[length[2]:length[3]]) == 0 \
            or sum(confidences[length[3]:length[4]]) == 0 or sum(confidences[length[4]:length[5]]) == 0 \
            or (sum(confidences[length[1]:length[2]]) == 0 and confidences[0] == 0):
        fail = 1
        incomplete = 1

    # impossible label combinations
    # 1. Clean Shaven + Beard length (5 o clock shadow, short, median, long)
    # 2. Clean Shaven + Mustache is connected to beard
    # 3. Clean Shaven + Sideburns is connected to beard
    # 4. China area + Sideburns is connected to beard
    # 5. Bald (top and sides or sides only) + having sideburns (Sideburns present, Sideburns is connected to beard)
    # 6. More than two choices on Mustache, Sideburns, and Bald
    # 7. More than two choices on Beard area and Beard length except Info not Vis
    # 8. Mustache is connected to beard + no beard (Clean Shaven, Info not Vis)
    # 9. Sideburns is connected to beard + not side to side
    elif (confidences[0] == 1 and sum(confidences[length[1]:length[2]]) != 0) or \
            (confidences[0] == 1 and confidences[11] == 1) or \
            (confidences[0] == 1 and confidences[15] == 1) or \
            (confidences[1] == 1 and confidences[15] == 1) or \
            (sum(confidences[19:21]) == 1 and sum(confidences[14:16]) != 0) or \
            (sum(confidences[9:13]) > 1) or \
            (sum(confidences[13:17]) > 1) or \
            (sum(confidences[17:22]) > 1) or \
            (sum(confidences[0:3]) > 1) or \
            (confidences[0] == 0 and sum(confidences[4:8]) > 1) or \
            (confidences[11] == 1 and sum(confidences[1:3]) == 0) or \
            (confidences[15] == 1 and confidences[2] == 0):
        fail = 1
        impossible = 1
    return fail, incomplete, impossible


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Filter out the satisfied images"
    )
    parser.add_argument(
        "--binary_file", "-bt", help="The .txt file that stores the image paths and the binary version results.",
        type=str
    )
    args = parser.parse_args()
    main(args.binary_file
         )
