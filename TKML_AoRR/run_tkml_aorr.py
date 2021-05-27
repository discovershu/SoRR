from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from TKML_AoRR.generate_noise_data.yeast_convert_to_data_loader import yeast_dataloader_generation

import os
from itertools import count
import time
import random
import numpy as np

from TKML_AoRR.models.models import *
from TKML_AoRR.models.preact_resnet import *

from torchvision.utils import save_image

if not torch.cuda.is_available():
    print('cuda is required but cuda is not available')
    exit()

# == parser start
parser = argparse.ArgumentParser(description='PyTorch')
# base setting 1: fixed
parser.add_argument('--job-id', type=int, default=1)
parser.add_argument('--seed', type=int, default=1000)
# base setting 2: fixed
parser.add_argument('--test-batch-size', type=int, default=100)
parser.add_argument('--momentum', type=float, default=0)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--data-path', type=str, default='./dataset/')
# experiment setting
parser.add_argument('--dataset', type=str, default='yeast')  # yeast, emotions, scene
parser.add_argument('--noise_random_seed', type=int, default=1)
parser.add_argument('--train_size', type=float, default=0.8)
parser.add_argument('--normal_type', type=int, default=0)  # 0:[-1,1], 1:[0,1]
parser.add_argument('--noise_ratio', type=int, default=10)
parser.add_argument('--data-aug', type=int, default=0)
parser.add_argument('--model', type=str, default='Linear')
# method setting
parser.add_argument('--lr', type=float, default=0.3)
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--k_prime', type=int, default=1)
parser.add_argument('--m', type=int, default=5)
parser.add_argument('--batch-size', type=int, default=1933)  # yeast 1933, emotions 474, scene 1925
parser.add_argument('--ssize', type=int, default=1700)# k
parser.add_argument('--method', type=int, default=1)
# --method=0: standard
# --method=1: q-SGD
args = parser.parse_args()
torch.cuda.set_device(args.gpu)
# == parser end
data_path = args.data_path + args.dataset
# if not os.path.isdir(data_path):
#     os.makedirs(data_path)

result_path = './results/'
if not os.path.isdir(result_path):
    os.makedirs(result_path)
result_path += args.dataset + '_' + str(args.data_aug) + '_' + args.model
result_path += '_' + str(args.method) + '_' + str(args.batch_size)
if args.method != 0:
    result_path += '_' + str(args.ssize)
result_path += '_' + str(args.job_id)
filep = open(result_path + '.txt', 'w')
with open(__file__) as f:
    filep.write('\n'.join(f.read().split('\n')[1:]))
filep.write('\n\n')

out_str = str(args)
print(out_str)
filep.write(out_str + '\n')

if args.seed is None:
    args.seed = random.randint(1, 10000)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.enabled = True

out_str = 'initial seed = ' + str(args.seed)
print(out_str)
filep.write(out_str + '\n\n')

# ===============================================================
# === dataset setting
# ===============================================================
kwargs = {}
train_transform = transforms.Compose([transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])
train_Sampler = None
test_Sampler = None
Shuffle = True
if args.dataset in ['yeast', 'emotions', 'scene']:
    if args.dataset == 'yeast':
        nh = 103
        num_class = 14
    elif args.dataset == 'emotions':
        nh = 72
        num_class = 6
    else:
        nh = 294
        num_class = 6
    nw = 1
    nc = 1
    end_epoch = 1000
    data_path = '{}{}/ori/{}.dat'.format(args.data_path, args.dataset, args.dataset)
    train_data, test_data = \
        yeast_dataloader_generation(args.dataset, data_path, \
                                    args.noise_random_seed, args.train_size, \
                                    args.normal_type, args.noise_ratio)
    Shuffle = False
else:
    print('specify dataset')
    exit()
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, sampler=train_Sampler,
                                           shuffle=Shuffle, **kwargs)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, sampler=test_Sampler,
                                          shuffle=False, **kwargs)

# ===============================================================
# === model setting
# ===============================================================
if args.model == 'LeNet':
    model = LeNet(nc, nh, nw, num_class).cuda()
elif args.model == 'PreActResNet18':
    model = PreActResNet18(nc, num_class).cuda()
elif args.model == 'Linear' or args.model == 'SVM':
    dx = nh * nw * nc
    model = Linear(dx, num_class).cuda()
else:
    print('specify model')
    exit()


# ===============================================================
# === utils def
# ===============================================================
def lr_decay_func(optimizer, lr_decay=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1
    return optimizer


def lr_scheduler(optimizer, epoch, lr_decay=0.1, interval=10):
    if args.data_aug == 0:
        if epoch == 10 or epoch == 50:
            optimizer = lr_decay_func(optimizer, lr_decay=lr_decay)
    if args.data_aug == 1:
        if epoch == 10 or epoch == 100:
            optimizer = lr_decay_func(optimizer, lr_decay=lr_decay)
    return optimizer


class multiClassHingeLoss(nn.Module):
    def __init__(self):
        super(multiClassHingeLoss, self).__init__()

    def forward(self, output, y):
        index = torch.arange(0, y.size()[0]).long().cuda()
        output_y = output[index, y.data.cuda()].view(-1, 1)
        loss = output - output_y + 1.0
        loss[index, y.data.cuda()] = 0
        loss[loss < 0] = 0
        loss = torch.sum(loss, dim=1) / output.size()[1]
        return loss


hinge_loss = multiClassHingeLoss()


def tkml_loss(output, y, k_prime):
    GT_min = torch.min(y * output + (1 - y) * 10000, 1)[0].view(-1, 1)
    loss = output - GT_min + 1.0
    loss[loss < 0] = 0
    loss = torch.sort(loss, dim=-1, descending=True)[0][:, k_prime]
    return loss


def tkml_acc_metric(output, y, k_prime):
    output = output.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    count = 0
    for i in range(y.shape[0]):
        GT_label_set = set(np.transpose(np.argwhere(y[i] == 1))[0])
        predict_index_set = set(output[i].argsort()[-k_prime:][::-1])
        if k_prime > len(GT_label_set):
            count = count + int(predict_index_set.issuperset(GT_label_set))
        else:
            count = count + int(GT_label_set.issuperset(predict_index_set))
    return count


# ===============================================================
# === train optimization def
# ===============================================================
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
ssize = args.ssize
lambda_k = Variable(torch.Tensor([0]), requires_grad=True).cuda()
lambda_hat = Variable(torch.Tensor([0]), requires_grad=True).cuda()


def train(epoch, m_value):
    global optimizer, ssize
    model.train()

    for batch_idx, (x, y) in enumerate(train_loader):
        bs = y.size(0)
        x = Variable(x.cuda())
        y = Variable(y.cuda())
        h1 = model(x)
        cr_loss = tkml_loss(h1, y, args.k_prime)
        if args.method == 0:
            loss = torch.mean(cr_loss)
        elif args.method == 1:
            if batch_idx == 0:
                lambda_k.data = torch.topk(cr_loss, min(ssize, bs), sorted=True, dim=0)[0][-1].data.flatten()
                lambda_hat.data = torch.topk(cr_loss, min(m_value, bs), sorted=True, dim=0)[0][-1].data.flatten()
            loss_term_1 = (min(ssize, bs) - min(m_value, bs)) * lambda_k
            loss_term_2 = (y.size()[0] - min(m_value, bs)) * lambda_hat
            loss_term_3 = cr_loss - lambda_k
            loss_term_3[loss_term_3 < 0] = 0
            loss_term_3 = lambda_hat - loss_term_3
            loss_term_3[loss_term_3 < 0] = 0
            loss_term_3 = torch.sum(loss_term_3)
            loss = loss_term_1 + loss_term_2 - loss_term_3
            loss = loss / (min(ssize, bs) - min(m_value, bs))
        else:
            print('specify method')
            exit()
        optimizer.zero_grad()
        lambda_k.retain_grad()
        lambda_hat.retain_grad()
        loss.backward()
        optimizer.step()
        lambda_k.data = lambda_k.data - args.lr * lambda_k.grad.data
        lambda_hat.data = lambda_hat.data + args.lr * lambda_hat.grad.data
        lambda_k.grad.data.zero_()
        lambda_hat.grad.data.zero_()

    optimizer.zero_grad()


# ===============================================================
# === train/test output def
# ===============================================================
def output(data_loader, test_best_acc):
    if data_loader == train_loader:
        model.train()
    elif data_loader == test_loader:
        model.eval()
    total_loss = 0
    total_correct = 0
    total_size = 0
    for batch_idx, (x, y) in enumerate(data_loader):
        x, y = Variable(x.cuda()), Variable(y.cuda())
        h1 = model(x)
        cr_loss = tkml_loss(h1, y, args.k_prime)
        loss = torch.mean(cr_loss)
        tkml_acc_count = tkml_acc_metric(h1, y, args.k_prime)
        total_loss += loss * y.size(0)
        total_correct += tkml_acc_count
        total_size += y.size(0)
        # print
    total_loss /= total_size
    total_acc = 100. * float(total_correct) / float(total_size)
    if data_loader == train_loader:
        out_str = 'tr_l={:.3f} tr_a={:.2f}:'.format(total_loss, total_acc)
    elif data_loader == test_loader:
        if total_acc > test_best_acc:
            test_best_acc = total_acc
        out_str = 'te_l={:.3f} te_a={:.2f} best_a={:.2f}:'.format(total_loss, total_acc, test_best_acc)
    print(out_str, end=' ')
    filep.write(out_str + ' ')
    return total_loss, total_acc, test_best_acc


# ===============================================================
# === start computation
# ===============================================================
# == for plot
# pl_result = np.zeros((end_epoch+1, 3, 2))  # epoch * (train, test, time) * (loss , acc)
# == main loop start
acc_list = []
best_acc_list = []
for i in range(args.ssize):
    if (i == 1) or ((i % 25 == 0) and (i != 0) and (i < args.batch_size - 1)) or (i == (args.batch_size - 1)):
        model.fc.reset_parameters()
        time_start = time.time()
        total_acc = 0
        best_acc = 0
        test_best_acc = 0
        print('i:', i)
        for epoch in range(end_epoch + 1):
            out_str = str(epoch)
            print(out_str, end=' ')
            filep.write(out_str + ' ')
            if epoch >= 1:
                train(epoch, i)
            output(train_loader, test_best_acc)
            _, total_acc, test_best_acc = output(test_loader, test_best_acc)
            # pl_result[epoch, 0, :] = output(train_loader)
            # pl_result[epoch, 1, :] = output(test_loader)
            time_current = time.time() - time_start
            # pl_result[epoch, 2, 0] = time_current
            # np.save(result_path + '_' + 'pl', pl_result)
            out_str = 'time={:.1f}:'.format(time_current)
            print(out_str)
            filep.write(out_str + '\n')
            epoch = epoch + 1
        acc_list.append(round(total_acc, 2))
        best_acc_list.append(round(test_best_acc, 2))
# np.save('./results/AoRR/data_{}_aorr_noise_{}_k_{}_seed_{}_acc_list.npy'.format( \
#     args.dataset, args.noise_ratio, args.k_prime, args.noise_random_seed), acc_list)
# np.save('./results/AoRR/data_{}_aorr_noise_{}_k_{}_seed_{}_best_acc_list.npy'.format( \
#     args.dataset, args.noise_ratio, args.k_prime, args.noise_random_seed), best_acc_list)
print(acc_list)
print(best_acc_list)