import argparse
from os import path
from tqdm import tqdm
import numpy as np
import os
from glob import glob
from models.model_wrapper import ModelWrapper
from models.lcploss import LCPLoss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import DataParallel
import time
from config_lmdb import Config
from lmdb_dataloader.lmdb_train_loader import LMDBDataLoader


class Train:
    def __init__(self, config):
        self.config = config
        if not isinstance(self.config.attr_ids, int):
            self.config.num_out = len(self.config.attr_ids)
        self.model = ModelWrapper(self.config)

        if self.config.pre_trained:
            print(f"Train with the pretrained model {self.config.model}...")
        else:
            print(f"Train {self.config.model} from scratch...")

        self.model = self.model.to(self.config.device)
        print(self.model)
        if torch.cuda.device_count() > 1:
            print(f"Model will use {torch.cuda.device_count()} GPUs!")
            self.model = DataParallel(self.model)

        print("Load training set")
        self.train_loader = LMDBDataLoader(
            config=self.config,
            lmdb_path=self.config.train_data,
            train=True,
        )
        self.test_loader = LMDBDataLoader(
            config=self.config,
            lmdb_path=self.config.val_data,
            train=False,
        )

        self.opt = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.opt)
        self.criterion = nn.BCEWithLogitsLoss()
        self.lcploss = LCPLoss(self.config)
        self.max_acc = 0.0
        self.acc_round_log = []
        self.max_acc_overall = 0.0
        self.epoch = self.config.epoch
        self.min_p = 10
        self.loss_log = []

    def run(self):
        for round in range(self.config.rounds):
            print("Start training...")
            max_acc = 0.0
            for iteration in range(self.epoch):
                self.model.train()
                it = tqdm(self.train_loader)
                losses = []
                # for data in train_Loader:
                for i, data in enumerate(it):
                    start_time = time.time()
                    inputs, labels = data
                    inputs, labels = inputs.to(self.config.device), labels.to(self.config.device)
                    self.opt.zero_grad()
                    output = self.model(inputs)
                    binary_output = self._check_incomplete(output)
                    lcploss, pdep, pex = self.lcploss(binary_output, True)
                    loss = (1 - self.config.lmbda) * self.criterion(output, labels) + self.config.lmbda * lcploss
                    losses.append(loss.item())
                    self.loss_log.append(loss.item())

                    loss.backward()
                    self.opt.step()
                    time_used = time.time() - start_time

                    it.set_postfix_str("Round: %d  loss: %.3f time: %.4f pdep: %.4f pex: %.4f" % (iteration + 1,
                                                                                                  loss.item(),
                                                                                                  time_used * 5,
                                                                                                  pdep,
                                                                                                  pex))
                self.evaluate(max_acc, iteration)
                self.scheduler.step(np.mean(losses))

        print(self.max_acc_overall)
        np.save(self.config.loss_file, np.array(self.loss_log))

    def evaluate(self, max_acc, iteration):
        self.model.eval()
        total = 0
        correct = torch.tensor([0] * 22).cuda()

        predicted_all = []
        with torch.no_grad():
            for j, (images, targets) in enumerate(self.test_loader):
                images, targets = images.to(self.config.device), targets.to(self.config.device)
                prediction = self.model(images)
                if self.config.label_compensation_val:
                    predicted = self._check_incomplete(prediction)
                else:
                    predicted = prediction.data > 0.5

                predicted_all.append(predicted)

                total += targets.size(0)
                for person in (predicted == targets):
                    correct += person
        accs = (100 * correct / total).cpu()
        total_acc = torch.mean(accs)

        predicted_all = torch.cat(predicted_all, dim=0)

        try:
            self.acc_round_log[iteration].append(accs)
        except Exception:
            self.acc_round_log.append([accs, ])
        p, pdep, pex = self.lcploss(predicted_all, True)
        print(total_acc, p, pdep, pex)
        if p < self.min_p:
            self.min_p = p
            if len(glob(self.config.best_acc_and_p + "/*.pth")) > 0:
                os.remove(glob(self.config.best_acc_and_p + "/*.pth")[0])
            self.save_static(self.config.best_acc_and_p, iteration, p, total_acc)
        elif p == self.min_p:
            if total_acc > self.max_acc_overall:
                if len(glob(self.config.best_acc_and_p + "/*.pth")) > 0:
                    os.remove(glob(self.config.best_acc_and_p + "/*.pth")[0])
                self.save_static(self.config.best_acc_and_p, iteration, p, total_acc)
        if total_acc > max_acc:
            max_acc = total_acc
            if max_acc >= self.max_acc_overall:
                self.max_acc_overall = max_acc
                if len(glob(self.config.best_acc + "/*.pth")) > 0:
                    os.remove(glob(self.config.best_acc + "/*.pth")[0])
                self.save_static(self.config.best_acc, iteration, p, total_acc)

    def save_static(self, save_path, iteration, p, acc):
        torch.save(
            self.model.state_dict(),
            path.join(
                save_path,
                "model_{}_accuracy:{:.4f}_p:{:.4f}_step:{}.pth".format(self.config.model,
                                                                       acc,
                                                                       p,
                                                                       iteration),
            ),
        )

    def _result_compensation(self, confidences, sub_binary_results, incomplete_pos):
        if len(incomplete_pos) != 0:
            sub_binary_results[(incomplete_pos,
                                torch.max(confidences[incomplete_pos, :], 1)[1])] += 1
        return sub_binary_results

    def _check_incomplete(self, confidences):
        binary_results = (confidences > 0.5).to(torch.int16)
        # incomplete cases
        # check Beard area, Beard length, Mustache, Sideburns, Bald
        for i in range(len(self.config.group_head_tail_indexes) - 1):
            sub_confidence = confidences[:,
                             self.config.group_head_tail_indexes[i]:self.config.group_head_tail_indexes[i + 1]]
            sub_results = binary_results[:,
                          self.config.group_head_tail_indexes[i]:self.config.group_head_tail_indexes[i + 1]]
            # get incomplete position of a batch prediction
            incomplete_positions = torch.where(torch.sum(sub_results, 1) == 0)[0]
            # for beard length, only if beard area is not clean shaven.
            if i == 1:
                incomplete_positions = incomplete_positions[
                    torch.where(binary_results[incomplete_positions, 0] == 0)[0]]
            # update compensated binary prediction
            binary_results[:, self.config.group_head_tail_indexes[i]:self.config.group_head_tail_indexes[i + 1]] = \
                self._result_compensation(sub_confidence, sub_results, incomplete_positions)
        return binary_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an attribute classification model"
    )

    # self.config.model and training parameters
    parser.add_argument(
        "--train_data", "-td", help="path of lmdb train file.", type=str
    )
    parser.add_argument(
        "--val_data", "-vd", help="path of validation images.", type=str
    )
    parser.add_argument(
        "--val_labels", "-vl", help="path of validation labels.", type=str
    )
    parser.add_argument(
        "--save_root", "-sr", help="root of results.", type=str
    )
    parser.add_argument(
        "--model", "-m", help="model name.", type=str
    )
    parser.add_argument(
        "--acc_file", "-af", help="saving the accuracy during the training in this file.", type=str
    )
    parser.add_argument(
        "--loss_file", "-lf", help="saving the loss during the training in this file.", type=str
    )
    parser.add_argument(
        "--batch_size", "-bs", help="bach size.", type=int, default=128
    )
    parser.add_argument(
        "--learning_rate", "-lr", help="learning rate.", type=float, default=1e-3
    )
    parser.add_argument(
        "--alpha", "-a", help="mie coefficient alpha.", type=float, default=1
    )
    parser.add_argument(
        "--lmbda", "-l", help="loss coefficient lambda.", type=float, default=0.1
    )
    parser.add_argument(
        "--epoch", "-e", help="# of epochs.", type=int, default=50
    )
    parser.add_argument(
        "--rounds", "-r", help="# of training rounds.", type=int, default=1
    )
    parser.add_argument(
        "--attr_ids", "-ai", help="attributes ids that are used to train.", type=int, nargs='+', default=-1,
    )
    parser.add_argument(
        "--pre_trained", "-pt", help="using pre-trained model from model-zoo.", action="store_true"
    )
    parser.add_argument(
        "--label_compensation_val", "-lcv", help="using label compensation when picking model.", action="store_false"
    )
    args = parser.parse_args()

    config = Config(args)

    torch.manual_seed(0)
    np.random.seed(0)

    train = Train(config)
    train.run()
