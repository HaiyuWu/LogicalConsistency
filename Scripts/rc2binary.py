import argparse
import numpy as np
import os.path as path
import os
from tqdm import tqdm


def main(file_path,
         thr,
         output,
         save_folder,
         compensate
         ):
    print(f"File path: {file_path}\n"
          f"Threshold: {thr}\n")
    saved_im_paths = {}
    print("Start confidence collection...")
    loss_function = "BCE"
    # loss_function = file_path.split("/")[-1].split("_")[0]
    file_name = file_path.split("/")[-1]
    with open(file_path, "r") as f:
        for line in tqdm(f.readlines()):
            temp_confidences = []
            im_path, confidences = line.strip().split("\t")
            for confidence in confidences[1:-1].split(","):
                temp_confidences.append(float(confidence))
            labels = binary_convertor(temp_confidences, thr, loss_function, compensate)
            saved_im_paths[im_path] = labels.astype(np.int8)
    print("Start writing...")
    _create_dir(save_folder)
    if not output:
        output = f"binary_{file_name}"
    result_file_path = path.join(save_folder, output)
    if os.path.exists(result_file_path):
        os.remove(result_file_path)
    with open(result_file_path, "a+") as f:
        for im_path, label in tqdm(saved_im_paths.items()):
            f.write(f"{im_path}\t{','.join(map(str, label))}\n")
    print(f"Results have been written to {result_file_path}")


def binary_convertor(confidences, thr, loss_function, compensate):
    out_index = [[0, 1, 2],
                 [4, 5, 6, 7],
                 [9, 10, 11, 12],
                 [13, 14, 15, 16],
                 [17, 18, 19, 20, 21],
                 [3, 8]]
    confidences = np.array(confidences)
    if compensate:
        return _check_incomplete(confidences, thr)
    return confidences > thr
    # elif loss_function == "MIE":
    #     predicted = np.zeros(confidences.shape)
    #     for indexes in out_index[:-1]:
    #         temp_array = predicted[indexes]
    #         temp_array[np.argmax(confidences[indexes])] += 1
    #         predicted[indexes] = temp_array
    #     predicted[out_index[-1]] = (confidences[out_index[-1]] > 0.5)
    #     return predicted


def _check_incomplete(confidences, thr):
    group_head_tail_indexes = [0, 4, 9, 13, 17, 22]
    binary_results = (confidences > thr).astype(np.int)
    # incomplete cases. for beard length, only if beard area is not clean shaven.
    for i in range(len(group_head_tail_indexes) - 1):
        sub_confidence = confidences[group_head_tail_indexes[i]:group_head_tail_indexes[i + 1]]
        sub_results = binary_results[group_head_tail_indexes[i]:group_head_tail_indexes[i + 1]]
        if np.sum(sub_results) == 0:
            if i == 1 and binary_results[0] == 1:
                continue
            sub_results[np.argmax(sub_confidence)] += 1
        binary_results[group_head_tail_indexes[i]:group_head_tail_indexes[i + 1]] = sub_results
    return binary_results


def _create_dir(dir_path):
    if not path.exists(dir_path):
        os.makedirs(dir_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Filter out the satisfied images"
    )

    parser.add_argument(
        "--threshold", "-t", help="the threshold value that is used to convert the labels.",
        type=float, default=0.5
    )
    parser.add_argument(
        "--result_file", "-rf", help="The .txt file that stores the image paths and the confidence values.",
        type=str
    )
    parser.add_argument(
        "--save_folder", "-sf", help="output file folder.", type=str, default="./binary_results",
    )
    parser.add_argument(
        "--output", "-o", help="output file name.", type=str, default=None,
    )
    parser.add_argument(
        "--compensate", "-c", help="doing label compensation.", action="store_true"
    )
    args = parser.parse_args()
    main(args.result_file,
         args.threshold,
         args.output,
         args.save_folder,
         args.compensate
         )
