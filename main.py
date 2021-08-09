import os
import yaml
import shutil
import argparse
import numpy as np
import multiprocessing
from PIL import Image

import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from model import ResNet, Bottleneck
from dataloader import CelebA_loader
from utils import AverageMeter, sample_accuracy, attributes_accuracy

p = argparse.ArgumentParser()
p.add_argument('--cfile', '-c', type=str, default='config')
p.add_argument('--checkpoints', type=str, default='checkpoints')
p.add_argument('--gpu', '-g', type=str, default='0',
               help='# of GPU. (1 GPU: single GPU)')
p.add_argument('--seed_pytorch', type=int, default=np.random.randint(4294967295))
p.add_argument('--seed_numpy', type=int, default=np.random.randint(4294967295))
args = p.parse_args()
np.random.seed(args.seed_numpy)
torch.manual_seed(args.seed_pytorch)

##################################
# Loading training configure
##################################
with open(args.cfile) as yml_file:
    config = yaml.safe_load(yml_file.read())['training']

batch_size = config['batch_size']
start_epoch = config['start_epoch']
end_epoch = config['end_epoch']
lr = config['lr']
momentum = config['momentum']
weight_decay = config['weight_decay']
tb = config['tb']
img_size = config['img_size']
num_classes = config['num_classes']

os.makedirs(args.checkpoints, exist_ok=True)
tb_path = os.path.join(args.checkpoints, tb)
if os.path.exists(tb_path):    shutil.rmtree(tb_path)
tb = SummaryWriter(log_dir=tb_path)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda:0')
best_acc = 0

def xent4multi(logits, gt, batch=None):
    log_softmax = F.log_softmax(logits, dim=1)
    xent_loss = -torch.sum(gt*log_softmax) / batch
    return xent_loss

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    print('Model save..')
    torch.save(state, filepath)
    if is_best:
        print('==> Updating the best model..')
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def main():
    global best_acc
    iters = 0
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    train_data = CelebA_loader(attribute_index=num_classes, size=img_size, train=True, transform=train_transform)
    train_sets = DataLoader(dataset=train_data,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=multiprocessing.cpu_count())
    
    test_data = CelebA_loader(attribute_index=num_classes, size=img_size, train=False, transform=transforms.ToTensor())
    test_sets = DataLoader(dataset=test_data,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=multiprocessing.cpu_count())
    
    scheduler = [int(end_epoch*0.5), int(end_epoch*0.75)]
    net = nn.DataParallel(ResNet(block=Bottleneck, layers=[3, 4, 23, 3], num_classes=num_classes)).to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    adjust_learning_rate = lr_scheduler.MultiStepLR(optimizer, scheduler, gamma=0.1)
    
    for epoch in range(start_epoch, end_epoch):
        iters = train(epoch, train_sets, net, optimizer, iters)
        adjust_learning_rate.step()
        val_loss, val_acc = validation(epoch, net, test_sets)
        
        is_best = val_acc > best_acc
        if is_best:
            best_acc = max(val_acc, best_acc)
        save_checkpoint({
            'epoch': epoch+1,
            'seed_numpy': args.seed_numpy,
            'seed_pytorch': args.seed_pytorch,
            'state_dict': net.state_dict(),
            'acc_clean': val_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict()}, is_best, checkpoint=args.checkpoints)
        
def train(epoch, train_iter, net, optimizer, iters):
    net.train()
    acc = AverageMeter()
    losses = AverageMeter()
    for idx, (inputs, targets) in enumerate(train_iter):
        inputs, targets = inputs.to(device), targets.to(device)
        net.zero_grad()
        logits = net(inputs)
        loss = xent4multi(logits, targets, batch=inputs.size(0))
        loss.backward()
        optimizer.step()
        
        iters += 1
        if idx % 100 == 0:
            accuracy = sample_accuracy(torch.softmax(logits, dim=1), targets)
            acc.update(accuracy, inputs.size(0))
            losses.update(loss.item(), inputs.size(0))
            print('| %d epoch (%d/%d) | loss: %.4f (avg: %.4f) | accuracy: %.4f (avg: %.4f) |' % (epoch, idx, len(train_iter), losses.val, losses.avg, acc.val, acc.avg))
            tb.add_scalar('Accuracy_train', acc.avg, iters)
            tb.add_scalar('Loss_train', losses.avg, iters)
            
    return iters
        
def validation(epoch, net, test_iter):
    net.eval()
    acc = AverageMeter()
    losses = AverageMeter()
    for idx, (inputs, targets) in enumerate(test_iter):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            logits = net(inputs)
        loss = xent4multi(logits, targets, batch=inputs.size(0))
        
        accuracy = sample_accuracy(torch.softmax(logits, dim=1), targets)
        acc.update(accuracy, inputs.size(0))
        losses.update(loss.item(), inputs.size(0))
    
    print('  Evaluation result @ %d epoch: loss %.4f, accuracy %.4f' % (epoch, losses.avg, acc.avg))
    tb.add_scalar('Accuracy_test', acc.avg, epoch)
    tb.add_scalar('Loss_test', losses.avg, epoch)
    return losses.avg, acc.avg
    
if __name__ == '__main__':
    main()
        