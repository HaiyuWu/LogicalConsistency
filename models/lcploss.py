import torch
import torch.nn as nn
import numpy as np


def probability(results, length, attrs, groups):
    mi = []
    p = []
    flag = False
    for i in range(length):
        try:
            if flag:
                continue
            group_results = torch.sum(results[:, groups[i]], 1)
            # P_in = P(group>1|cur==1)
            p.append(len(torch.where(group_results[torch.where(results[:, attrs[i]] == 1)[0]] > 0)[0]) /
                     len(torch.where(results[:, attrs[i]] == 1)[0]))
            flag = True
        except ZeroDivisionError:
            p.append(0.0)
    return torch.tensor(np.mean(mi)), torch.tensor(np.mean(p))


class LCPLoss(nn.Module):
    def __init__(self, config):
        super(LCPLoss, self).__init__()
        self.ex_given_attrs = config.ex_given_attrs
        self.ex_groups = config.ex_groups
        self.dep_given_attrs = config.dep_given_attrs
        self.dep_groups = config.dep_groups
        self.ex_len = len(self.ex_given_attrs)
        self.dep_len = len(self.dep_given_attrs)
        self.batch_size = config.batch_size
        self.out_index = config.out_index
        self.alpha = config.alpha
        self.head_tail_indexes = config.group_head_tail_indexes

    def forward(self, predictions, ask_in_ex=False):
        miin, pdep = probability(predictions, self.dep_len, self.dep_given_attrs, self.dep_groups)
        miex, pex = probability(predictions, self.ex_len, self.ex_given_attrs, self.ex_groups)
        loss = (1 - pdep + pex * self.alpha) ** 2
        if ask_in_ex:
            return loss, pdep, pex
        else:
            return loss
