import numpy as np
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def sample_accuracy(probs, gt):
    max_index = torch.max(probs, dim=1)[1].squeeze()
    true_or_false = torch.where(max_index.eq(torch.max(gt, dim=1)[1].squeeze())==True, 1, 0)
    true_or_false_sum = torch.sum(true_or_false, dim=1)
    corrects = torch.where(true_or_false_sum==gt.size(3), 1, 0)
    accuracy = torch.mean(corrects.float())
    return accuracy

def attributes_accuracy(probs, gt):
    max_index = torch.max(probs, dim=1)[1].squeeze()
    true_or_false = torch.where(max_index.eq(torch.max(gt, dim=1)[1].squeeze())==True, 1, 0)
    accuracy = torch.mean(true_or_false.float(), dim=0)
    return accuracy

def sample_accuracy(probs, gt):
    