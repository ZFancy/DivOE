# -*- coding: utf-8 -*-
from torch.autograd import Variable
import numpy as np
import os
import pickle
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from models.resnet import resnet50

if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.validation_dataset import validation_split
    from utils.display_results import show_performance, get_measures, print_measures, print_measures_with_std
    import utils.score_calculation as lib

parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with DivOE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--calibration', '-c', action='store_true',
                    help='Train a model to be used for calibration. This holds out some data for validation.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=4, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.0001, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size.')
parser.add_argument('--oe_batch_size', type=int, default=64, help='Batch size.')
parser.add_argument('--num_to_avg', type=int, default=2, help='Average measures across num_to_avg runs.')
parser.add_argument('--use_xent', '-x', action='store_true', help='Use cross entropy scoring instead of the MSP.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots/', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='./snapshots/pretrained',
                    help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')
parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin|gradnorm')
# EG specific
parser.add_argument('--m_in', type=float, default=-25.,
                    help='margin for in-distribution; above this value will be penalized')
parser.add_argument('--m_out', type=float, default=-7.,
                    help='margin for out-distribution; below this value will be penalized')
parser.add_argument('--score', type=str, default='MSP', help='MSP|energy')
parser.add_argument('--seed', type=int, default=1, help='seed for np(tinyimages80M sampling); 1|2|8|100|107')
parser.add_argument('--extrapolation_ratio',type=float,default=0.5,help='the ratio of extrapolated outliers in the whole batch')
parser.add_argument('--epsilon',type=float,default=0.01,help='extrapolation epsilon')
parser.add_argument('--rel_step_size',type=float,default=1/4,help='extrapolation relative step size')
parser.add_argument('--num_steps',type=int,default=5,help='optimization step number')
parser.add_argument('--extrapolation_score', type=str, default='MSP', help='MSP|energy for extrapolation optimization')

args = parser.parse_args()

print(f"This experiment starts from {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")

state = {k: v for k, v in args._get_kwargs()}
print(state)

torch.manual_seed(1)
np.random.seed(args.seed)

# mean and standard deviation of channels of CIFAR-10 images
mean= torch.Tensor([0.5, 0.5, 0.5]).view(3,1,1).tolist()
std = torch.Tensor([0.5, 0.5, 0.5]).view(3,1,1).tolist()

import imagenet21_aug
train_transform_id = trn.Compose([trn.Resize((224,224)), trn.RandomCrop(224, padding = 4), trn.RandomHorizontalFlip(), trn.ToTensor() , trn.Normalize(mean, std)])
train_transform_ood = trn.Compose([trn.Resize((224,224)), trn.RandomCrop(224, padding = 4), trn.RandomHorizontalFlip(), trn.ToTensor() , trn.Normalize(mean, std)])
train_transform_ood.transforms.insert(0, imagenet21_aug.RandAugment(1, 9, args=args))
test_transform = trn.Compose([trn.Resize((224,224)), trn.ToTensor(), trn.Normalize(mean, std)])

train_data_in  = dset.ImageFolder(root="../data/ImageNet/train", transform = train_transform_id)
test_data      = dset.ImageFolder(root="../data/ImageNet/val",   transform = test_transform)
train_data_ood = dset.ImageFolder(root="../data/imagenet21k_resized/train", transform = train_transform_ood)
num_classes = 1000

calib_indicator = ''
if args.calibration:
    train_data_in, val_data = validation_split(train_data_in, val_share=0.1)
    calib_indicator = '_calib'

train_loader_in  = torch.utils.data.DataLoader(train_data_in,  batch_size=args.batch_size,    shuffle=True,  num_workers=args.prefetch, pin_memory=False)
train_loader_out = torch.utils.data.DataLoader(train_data_ood, batch_size=args.oe_batch_size, shuffle=True,  num_workers=args.prefetch, pin_memory=True)
test_loader      = torch.utils.data.DataLoader(test_data,      batch_size=args.test_bs,    shuffle=False, num_workers=args.prefetch, pin_memory=False)

# Create model
net = resnet50(pretrained = True).cuda()

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

optimizer = torch.optim.SGD(
    net.parameters(), state['learning_rate'], momentum=state['momentum'],
    weight_decay=state['decay'], nesterov=True)


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader_in),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.learning_rate))

def extrapolate(model, data, epsilon, rel_step_size=1/4, num_steps=5,rand_init=True):
    #because the normalization std equals 0.5
    epsilon=2*epsilon
    model.eval()
    data=data.cuda()
    x_adv = data.detach() + torch.from_numpy(
        np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
    x_adv = torch.clamp(x_adv, -1.0, 1.0)
    for k in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)
        model.zero_grad()
        with torch.enable_grad():
            if args.extrapolation_score == 'MSP':
                loss_adv = -(output.mean(1) - torch.logsumexp(output, dim=1)).mean()
            elif args.extrapolation_score == 'energy':
                Ec_out = -torch.logsumexp(output, dim=1)
                loss_adv = torch.pow(F.relu(args.m_out - Ec_out), 2).mean()
        loss_adv.backward()
        step_size=epsilon*rel_step_size
        eta = step_size * x_adv.grad.sign()
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, -1.0, 1.0)
    x_adv = Variable(x_adv, requires_grad=False)
    model.train()
    return x_adv

# /////////////// Training ///////////////

def train():
    net.train()  # enter train mode
    loss_avg = 0.0

    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))
    for in_set, out_set in zip(train_loader_in, train_loader_out):
        aug_length=int(len(out_set[0])*args.extrapolation_ratio)
        adv_outlier=extrapolate(net,out_set[0][:aug_length],args.epsilon,args.rel_step_size,
                               args.num_steps)
        data = torch.cat((in_set[0], adv_outlier.cpu(), out_set[0][aug_length:]), 0)
        target = in_set[1]

        data, target = data.cuda(), target.cuda()

        # forward
        x = net(data)

        # backward
        scheduler.step()
        optimizer.zero_grad()

        loss = F.cross_entropy(x[:len(in_set[0])], target)
        # cross-entropy from softmax distribution to uniform distribution
        if args.score == 'energy':
            Ec_out = -torch.logsumexp(x[len(in_set[0]):], dim=1)
            Ec_in = -torch.logsumexp(x[:len(in_set[0])], dim=1)
            loss += 0.1 * (torch.pow(F.relu(Ec_in - args.m_in), 2).mean() + torch.pow(F.relu(args.m_out - Ec_out),
                                                                                      2).mean())
        elif args.score == 'MSP':
            loss += 0.5 * -(x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()

        loss.backward()
        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2
    state['train_loss'] = loss_avg


# test function
def test():
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            # forward
            output = net(data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)



if args.test:
    test()
    print(state)
    exit()

if args.extrapolation_score == 'MSP':
    save_info = f'MSP_DivOE'
elif args.extrapolation_score == 'energy':
    save_info = f'energy_DivOE'

if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

with open(os.path.join(args.save, save_info+'_s' + str(args.seed) + '_training_results.csv'), 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

print('Beginning Training\n')

# Main loop
for epoch in range(0, args.epochs):
    state['epoch'] = epoch

    begin_epoch = time.time()

    train()
    test()

    # Save model
    torch.save(net.state_dict(),
               os.path.join(args.save, save_info+'_s' + str(args.seed) + '_epoch_' + str(epoch) + '.pt'))

    # Let us not waste space and delete the previous model
    prev_path = os.path.join(args.save, save_info+'_s' + str(args.seed) + '_epoch_'+ str(epoch - 1) + '.pt')
    if os.path.exists(prev_path) and epoch % 10 != 0: os.remove(prev_path)

    # Show results
    with open(os.path.join(args.save, save_info+'_s' + str(args.seed) + '_training_results.csv'), 'a') as f:
        f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
            (epoch + 1),
            time.time() - begin_epoch,
            state['train_loss'],
            state['test_loss'],
            100 - 100. * state['test_accuracy'],
        ))

    # # print state with rounded decimals
    # print({k: round(v, 4) if isinstance(v, float) else v for k, v in state.items()})

    print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f}'.format(
        (epoch + 1),
        int(time.time() - begin_epoch),
        state['train_loss'],
        state['test_loss'],
        100 - 100. * state['test_accuracy'])
    )

net.eval()

ood_num_examples = len(test_data) // 5
expected_ap = ood_num_examples / (ood_num_examples + len(test_data))

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()

def get_ood_scores(loader, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                break

            data = data.cuda()

            output = net(data)
            smax = to_np(F.softmax(output, dim=1))

            if args.use_xent:
                _score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1))))
            else:
                if args.score == 'energy':
                    _score.append(-to_np((args.T * torch.logsumexp(output / args.T, dim=1))))
                elif args.score == 'MSP':  # original MSP and Mahalanobis (but Mahalanobis won't need this returned)
                    _score.append(-np.max(smax, axis=1))

            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                if args.use_xent:
                    _right_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[right_indices])
                    _wrong_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[wrong_indices])
                else:
                    _right_score.append(-np.max(smax[right_indices], axis=1))
                    _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()

in_score, right_score, wrong_score = get_ood_scores(test_loader, in_dist=True)

num_right = len(right_score)
num_wrong = len(wrong_score)
print('Accuracy {:.2f}'.format(100 * num_right / (num_wrong + num_right)))

# /////////////// Error Detection ///////////////

print('\n\nError Detection')
show_performance(wrong_score, right_score, method_name=save_info)

# /////////////// OOD Detection ///////////////
auroc_list, aupr_list, fpr_list = [], [], []

def get_and_print_results(ood_loader, num_to_avg=args.num_to_avg):
    aurocs, auprs, fprs = [], [], []

    for _ in range(num_to_avg):
        out_score = get_ood_scores(ood_loader)
        if args.out_as_pos:  # OE's defines out samples as positive
            measures = get_measures(out_score, in_score)
        else:
            measures = get_measures(-in_score, -out_score)
        aurocs.append(measures[0]);
        auprs.append(measures[1]);
        fprs.append(measures[2])
    print(in_score[:3], out_score[:3])
    auroc = np.mean(aurocs);
    aupr = np.mean(auprs);
    fpr = np.mean(fprs)
    auroc_list.append(auroc);
    aupr_list.append(aupr);
    fpr_list.append(fpr)

    if num_to_avg >= 2:
        print_measures_with_std(aurocs, auprs, fprs, save_info)
    else:
        print_measures(auroc, aupr, fpr, save_info)


# /////////////// Textures ///////////////
ood_data = dset.ImageFolder(root="../data/dtd/images", transform = test_transform)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=4, pin_memory=True)
print('\n\nTexture Detection')
get_and_print_results(ood_loader)

# /////////////// Places365 ///////////////
ood_data = dset.ImageFolder(root="../data/Places", transform = test_transform)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=4, pin_memory=True)
print('\n\nPlaces365 Detection')
get_and_print_results(ood_loader)

# /////////////// iNaturalist ///////////////
ood_data = dset.ImageFolder(root="../data/iNaturalist", transform = test_transform)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=4, pin_memory=True)
print('\n\niNaturalist Detection')
get_and_print_results(ood_loader)

# /////////////// SUN ///////////////
ood_data = dset.ImageFolder(root="../data/SUN", transform = test_transform)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=4, pin_memory=True)
print('\n\nSUN Detection')
get_and_print_results(ood_loader)

# /////////////// Mean Results ///////////////

print('\n\nMean Test Results!!!!!')
print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=save_info)