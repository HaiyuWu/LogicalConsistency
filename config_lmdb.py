from easydict import EasyDict
import torch
from os import path, makedirs


class Config(EasyDict):
    def __init__(self, args):
        self.train_data = args.train_data
        self.val_data = args.val_data
        self.save_root = args.save_root
        self.model = args.model
        self.acc_file = args.acc_file
        self.loss_file = args.loss_file
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.epoch = args.epoch
        self.rounds = args.rounds
        self.workers = 16
        self.pin_memory = True
        self.label_compensation_val = args.label_compensation_val
        self.attr_ids = args.attr_ids
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_out = 22
        self.pre_trained = args.pre_trained
        self.lmbda = args.lmbda
        self.alpha = args.alpha

        self.best_acc = path.join(self.save_root,
                                  f"best_acc_labmda:{self.lmbda}_alpha:{self.alpha}_bs:{self.batch_size}_lr:{self.learning_rate}")
        self.best_acc_and_p = path.join(self.save_root,
                                        f"best_acc_and_p_labmda:{self.lmbda}_alpha:{self.alpha}_bs:{self.batch_size}_lr:{self.learning_rate}")

        if not path.exists(self.best_acc):
            makedirs(self.best_acc)
        if not path.exists(self.best_acc_and_p):
            makedirs(self.best_acc_and_p)

        # predict the results separately
        self.out_index = [[0, 1, 2],
                          [4, 5, 6, 7],
                          [9, 10, 11, 12],
                          [13, 14, 15, 16],
                          [17, 18, 19, 20, 21],
                          [3, 8]]
        self.out_list = [3, 4, 3, 3, 4, 5]
        # exclusive groups
        self.ex_given_attrs = [0, 1, 4, 5, 6, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20]
        self.ex_groups = [[1, 2, 4, 5, 6, 7, 8, 11, 15],
                          [2, 15],
                          [5, 6, 7],
                          [6, 7],
                          [7],
                          [10, 11, 12],
                          [11, 12],
                          [12],
                          [14, 15, 16],
                          [15, 16, 19, 20],
                          [16, 19, 20],
                          [18, 19, 20, 21],
                          [19, 20, 21],
                          [20, 21],
                          [21]]
        # inclusive groups
        self.dep_given_attrs = [1, 2, 11, 15]
        self.dep_groups = [[4, 5, 6, 7, 8],
                           [4, 5, 6, 7, 8],
                           [1, 2],
                           [2]]
        # group indexes
        self.group_head_tail_indexes = [0, 4, 9, 13, 17, 22]
