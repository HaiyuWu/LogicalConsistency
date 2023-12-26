import os
from models.model_wrapper import ModelWrapper
import torch
import pandas as pd
import argparse
import torch.nn as nn
from tqdm import tqdm
from os import path
from lmdb_dataloader.test_loader import TestDataLoader
from easydict import EasyDict
from torch.nn import DataParallel


class Config(EasyDict):
    def __init__(self, args):
        self.test_im_path = args.test_im_path
        self.test_label_file = args.test_label_file
        self.attr_ids = args.attr_ids

        self.model = args.model_name
        self.static = args.static
        self.test_model = args.model
        self.whole_model = args.whole_model

        self.output = args.output
        self.save_p_n_acc = args.save_p_n_acc
        self.save_p_n_image = args.save_p_n_image
        self.val_result = args.val_result
        self.raw_confidence = args.raw_confidence

        self.batch_size = args.batch_size
        self.label_compensation = args.label_compensation
        self.im_paths_file = args.im_paths_file
        self.pre_trained = False
        self.workers = 16
        self.dataset = args.dataset
        if self.dataset == "celeba":
            self.in_channel = 2048
            self.num_out = 40
        else:
            self.in_channel = 2048
            self.num_out = 22
        self.pin_memory = True
        self.impossible_detection = args.impossible_detection
        self.folder_paths_file = args.folder_paths_file
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.output or self.save_p_n_acc:
            self._create_dir(self.val_result)
        if not isinstance(self.attr_ids, int):
            self.num_out = len(self.attr_ids)
        self.out_index = [[0, 1, 2],
                          [4, 5, 6, 7],
                          [9, 10, 11, 12],
                          [13, 14, 15, 16],
                          [17, 18, 19, 20, 21],
                          [3, 8]]
        self.group_head_tail_indexes = [0, 4, 9, 13, 17, 22]

    def _create_dir(self, dir_path):
        if not path.exists(dir_path):
            os.makedirs(dir_path)


class Test:
    def __init__(self, config):
        # loading dataset
        print("Loading datasets...")
        self.config = config
        self.sigmoid = nn.Sigmoid()
        print(config)

        self.o_file = "output"
        self.test_loader = TestDataLoader(self.config)

        if self.config.whole_model:
            self.cls_model = torch.load(self.config.test_model, map_location="cuda:0")
        elif self.config.static:
            print(f"Model will use {torch.cuda.device_count()} GPUs!")
            if not self.config.model:
                raise AssertionError("Please enter the name of the aim model (MOON, AFFACT, resnet50, densenet121)")
            self.cls_model = ModelWrapper(self.config)

            try:
                self.cls_model.load_state_dict(torch.load(self.config.test_model))
            except Exception:
                self.cls_model = DataParallel(self.cls_model)
                self.cls_model.load_state_dict(torch.load(self.config.test_model, map_location=torch.device('cpu')))
        else:
            raise AssertionError("Please enter the type of model weights: --whole_model or --static")

        self.cls_model.to(self.config.device)
        self.cls_model.eval()

        if self.config.output:
            self.o_file = self.config.output
        self.failed = 0
        self.total = 0
        self.p_total, self.n_total = torch.tensor([0] * self.config.num_out).to(self.config.device), \
                                     torch.tensor([0] * self.config.num_out).to(self.config.device)
        self.correct, self.correct_p, self.correct_n = torch.tensor([0] * self.config.num_out).to(self.config.device), \
                                                       torch.tensor([0] * self.config.num_out).to(self.config.device), \
                                                       torch.tensor([0] * self.config.num_out).to(self.config.device)

    def run(self):
        logger = {}
        print("Start evaluation...")
        with torch.no_grad():
            for j, (images, im_paths, labels) in enumerate(tqdm(self.test_loader), 0):
                images = images.to(self.config.device)
                prediction = self.cls_model(images)
                prediction = torch.sigmoid(prediction)
                if self.config.label_compensation:
                    predicted = self._check_incomplete(prediction)
                else:
                    predicted = (prediction > 0.5)

                for raw_data, label, im_path in zip(prediction, predicted, im_paths):
                    logger[im_path] = [raw_data, label.int()]
                if self.config.test_label_file:
                    labels = labels.to(self.config.device)

                    self.total += labels.size(0)
                    for i in range(len(labels[0])):
                        self.n_total[i] += len(torch.where(labels[:, i] == 0)[0])
                        self.p_total[i] += len(torch.where(labels[:, i] == 1)[0])
                    for i, person in enumerate(predicted == labels):
                        self.correct_counter(i, person, predicted, labels, self.config.impossible_detection)
        if self.config.test_label_file:
            if self.config.num_out == 22:
                self._result_printer(self.correct,
                                     self.total,
                                     self.correct_n,
                                     self.n_total,
                                     self.correct_p,
                                     self.p_total)
            else:
                p_acc = self.correct_p / self.p_total
                n_acc = self.correct_n / self.n_total
                avg_acc = (p_acc + n_acc)/2
                print(f"Accuracy -- {torch.mean(avg_acc)} -- {self.total} samples")
                print(f"Positive acc -- {torch.mean(p_acc)}       Negative acc -- {torch.mean(n_acc)}")
                print(f"Acc on positive samples: {p_acc} -- {self.p_total} samples")
                print(f"Acc on negative samples: {n_acc} -- {self.n_total} samples")
                print(avg_acc[([0, 4, 5, 16, 20, 22, 24, 28, 35])])
                print(torch.mean(avg_acc[([0, 4, 5, 16, 20, 22, 24, 28, 35])]))
                print(self.failed / self.total)

        if not self.config.test_label_file or self.config.output:
            result_file = f"{os.path.join(self.config.val_result, self.o_file)}.txt"
            if path.exists(result_file):
                os.remove(result_file)
            print(f"Saving results to {result_file}")
            with open(result_file, "a+") as f:
                for im_path, label in logger.items():
                    out_data = 0 if self.config.raw_confidence else 1
                    f.write(f"{im_path}\t{label[out_data].cpu().tolist()}\n")

    def correct_counter(self, index, person, predicted_label, ground_truth, impossible_detection):
        flag = False
        if impossible_detection:
            flag = self.condition_checking(predicted_label[index], self.config.dataset)

        self.failed += flag

        p_pos = torch.where(person == 1)[0]
        if not flag:
            for pos in p_pos:
                if int(ground_truth[index][pos]) == 1:
                    self.correct_p[pos] += 1
                elif int(ground_truth[index][pos]) == 0:
                    self.correct_n[pos] += 1
            self.correct += person

    def condition_checking(self, confidences, dataset):
        if dataset != "celeba":
            length = self.config.group_head_tail_indexes

            # incomplete cases
            # 1. no results in Beard area, Mustache, Sideburns, or Bald
            # 2. not clean shaven but no results in Beard length
            if sum(confidences[length[0]:length[1]]) == 0 or sum(confidences[length[2]:length[3]]) == 0 \
                    or sum(confidences[length[3]:length[4]]) == 0 or sum(confidences[length[4]:length[5]]) == 0 \
                    or (sum(confidences[length[1]:length[2]]) == 0 and confidences[0] == 0):
                return True

            # impossible label combinations
            # 1. Clean Shaven + Beard length
            # 2. Clean Shaven + Mustache is connected to beard
            # 3. Clean Shaven + Sideburns is connected to beard
            # 4. Chin area + Sideburns is connected to beard
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
                return True
            return False
        else:
            # 1. No beard + (5 o'clock shadow, goatee, mustache)
            if confidences[24] == 1 and sum(confidences[([0, 16, 22])]) > 0:
                return True
            # 2. Bangs + Receding hairline
            if confidences[5] == 1 and confidences[28] == 1:
                return True
            # 3. Bald + (Bangs, Receding hairline, wearing hat)
            if confidences[4] == 1 and sum(confidences[([5, 28, 35])]) > 0:
                return True
            # 4. female + (5 o'clock shadow, goatee, mustache, having beard)
            if confidences[20] == 0 and (sum(confidences[([0, 16, 22])]) > 0 or confidences[24] == 0):
                return True
            return False

    def _result_printer(self,
                        num_correct,
                        num_total,
                        num_correct_n,
                        num_n_total,
                        num_correct_p,
                        num_p_total):

        CLASS_LIST = ["Clean_shaven", "Chin_area", "Side_to_side", "Beard_area-Info_not_vis",
                      "5_o_clock_shadow", "Short", "Medium", "Long", "Bread_length-Info_not_vis",
                      "Mustache-None", "Isolated", "Mustache-Connected_to_beard", "Mustache-Info_not_vis",
                      "Sideburns-None", "Sideburns_present", "Sideburns-Connected_to_beard", "Sideburns-Info_not_vis",
                      "Bald-False", "Top_only", "Top_and_sides", "Sides_only", "Bald-Info_not_vis"]

        accs = 100 * num_correct / num_total
        n_accs = 100 * num_correct_n / num_n_total
        p_accs = 100 * num_correct_p / num_p_total

        print(f"Overall acc: {round(float(torch.mean(accs)), 2)}\t\t"
              f"Negative acc: {round(float(torch.mean(n_accs)), 2)}\t\t"
              f"Positive acc: {round(float(torch.mean(p_accs)), 2)}")

        dict = {}

        for i in range(len(CLASS_LIST)):
            print(f"{CLASS_LIST[i]}: {round(float((n_accs[i] + p_accs[i])/2), 2)}\t\t"
                  f"Negative acc: {round(float(n_accs[i]), 2)}/{num_n_total[i]}\t\t"
                  f"Positive acc: {round(float(p_accs[i]), 2)}/{num_p_total[i]}")
            if self.config.save_p_n_acc:
                dict[CLASS_LIST[i]] = [round(float((n_accs[i] + p_accs[i])/2), 2),
                                       round(float(n_accs[i]), 2),
                                       int(num_n_total[i]),
                                       round(float(p_accs[i]), 2),
                                       int(num_p_total[i])]
        if dict:
            if self.config.output:
                prefix = self.config.output
            else:
                prefix = self.config.test_model.split("/")[-1][:-4]
            print(f"positive negative accuracy saved to {os.path.join(self.config.val_result)}")
            pd.DataFrame(dict, index=["Overall_acc",
                                      "Negative_acc",
                                      "# of Negative",
                                      "Positive_acc",
                                      "# of Positive"]).to_csv(os.path.join(self.config.val_result,
                                                                            f"{prefix}-positive_negative_acc.csv"))

    def _check_incomplete(self, confidences):
        binary_results = (confidences > 0.5).to(torch.int16)
        # incomplete cases. for beard length, only if beard area is not clean shaven.
        for i in range(len(self.config.group_head_tail_indexes) - 1):
            sub_confidence = confidences[:,
                             self.config.group_head_tail_indexes[i]:self.config.group_head_tail_indexes[i + 1]]
            sub_results = binary_results[:,
                          self.config.group_head_tail_indexes[i]:self.config.group_head_tail_indexes[i + 1]]
            incomplete_positions = torch.where(torch.sum(sub_results, 1) == 0)[0]
            if i == 1:
                incomplete_positions = incomplete_positions[
                    torch.where(binary_results[incomplete_positions, 0] == 0)[0]]
            binary_results[:, self.config.group_head_tail_indexes[i]:self.config.group_head_tail_indexes[i + 1]] = \
                self._result_compensation(sub_confidence, sub_results, incomplete_positions)
        return binary_results

    def _result_compensation(self, confidences, sub_binary_results, incomplete_pos):
        if len(incomplete_pos) != 0:
            sub_binary_results[(incomplete_pos,
                                torch.max(confidences[incomplete_pos, :], 1)[1])] += 1
        return sub_binary_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Test the classification model"
    )

    # self.config.model and training parameters
    parser.add_argument(
        "--dataset", "-dataset", help="which dataset.", type=str, default="fh37k"
    )
    parser.add_argument(
        "--test_im_path", "-i", help="path of test folder.", type=str, default=None
    )
    parser.add_argument(
        "--test_label_file", "-l", help="path of ground truth label file.", type=str, default=None
    )
    parser.add_argument(
        "--batch_size", "-bs", help="batch size.", type=int, default=16
    )
    parser.add_argument(
        "--model", "-tm", help="model that needs to be tested.", type=str
    )
    parser.add_argument(
        "--attr_ids", "-ai", help="attributes ids that are used to train.", type=int, nargs='+', default=-1,
    )
    parser.add_argument(
        "--output", "-o", help="output file name.", type=str, default=None,
    )
    parser.add_argument(
        "--static", "-s", help="the model was saved with state_dict.", action="store_true",
    )
    parser.add_argument(
        "--model_name", "-mn", help="model name of the static weights.", type=str, default=None
    )
    parser.add_argument(
        "--folder_paths_file", "-fpf", help="path of a .txt file that stores the image folders", type=str, default=None
    )
    parser.add_argument(
        "--im_paths_file", "-ipf", help="path of a .txt file that stores the image paths", type=str, default=None
    )
    parser.add_argument(
        "--whole_model", "-wm", help="The whole model was saved.", action="store_true",
    )
    parser.add_argument(
        "--save_p_n_acc", "-pna", help="write the positive and negative acc out.", action="store_true",
    )
    parser.add_argument(
        "--raw_confidence", "-rc", help="Save the raw prediction results.", action="store_true",
    )
    parser.add_argument(
        "--save_p_n_image", "-pni", help="Save the TP, FP, TN, TF images.", action="store_true",
    )
    parser.add_argument(
        "--label_compensation", "-lc",
        help="Choose the attribute with the maximum confidence in the incomplete groups.", action="store_true",
    )
    parser.add_argument(
        "--impossible_detection", "-id",
        help="Doing impossible detection for the prediction.", action="store_true",
    )
    parser.add_argument(
        "--val_result", "-vr", help="folder to save the evaluation results.", type=str, default="./val_result"
    )
    args = parser.parse_args()

    config = Config(args)
    test = Test(config)
    test.run()
