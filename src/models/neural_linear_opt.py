import os
import sys
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.autograd import Variable

from scipy.stats import invgamma

import warnings

warnings.filterwarnings("ignore")
import logging

class NeuralLinear(object):
    '''
    the neural linear model
    '''

    def __init__(self, args, model, repr_dim, output_dim):

        self.args = args
        self.model = model
        self.output_dim = output_dim  # output dim of ood selection branch
        self.repr_dim = repr_dim
        self.a = torch.tensor([args.a0 for _ in range(self.output_dim)]).cuda()
        self.b = torch.tensor([args.b0 for _ in range(self.output_dim)]).cuda()
        # Update formula in BDQN
        self.sigma = args.sigma  # W prior variance
        self.sigma_n = args.sigma_n  # noise variacne
        self.eye = torch.eye(self.repr_dim).cuda()
        self.mu_w = torch.normal(0, 0.01, size=(self.output_dim, self.repr_dim)).cuda()
        cov_w = np.array([self.sigma * np.eye(self.repr_dim) for _ in range(self.output_dim)])
        self.cov_w = torch.from_numpy(cov_w).cuda()

        self.beta_s = None
        self.train_x_ood = torch.empty(0, 3, 32, 32)
        self.train_x_id_base = torch.empty(0, 3, 32, 32)

    def update_representation(self):
        latent_z = torch.empty(0, self.model.nChannels, device="cuda")
        print('begin updating representation')
        data_loader = torch.utils.data.DataLoader(
            SimpleDataset(self.train_x, self.train_y),
            batch_size=256, shuffle=False, num_workers=2)
        self.model.eval()
        with torch.no_grad():
            for images, _ in data_loader:
                partial_latent_z = self.model.get_representation(images.cuda())
                latent_z = torch.cat((latent_z, partial_latent_z), dim=0)
            self.latent_z = latent_z
            assert len(self.latent_z) == len(self.train_x)

    def train_blr(self, train_loader_in, train_loader_out, criterion, optimizer,scheduler, epoch, save_dir, log,
                  energy_model=False):
        batch_time = AverageMeter()
        out_confs = AverageMeter()
        in_confs = AverageMeter()
        in_losses = AverageMeter()
        out_losses = AverageMeter()
        out_energy_losses = AverageMeter()
        in_energy_losses = AverageMeter()
        nat_top1 = AverageMeter()
        log.debug("######## Start training NN at epoch {} ########".format(epoch))
        end = time.time()
        out_len = len(train_loader_out.dataset)
        in_len = len(train_loader_in.dataset)
        if epoch == 0:
            noise = torch.normal(0.0, self.args.sigma_n ** 0.5, size=(1,)).item()
            self.train_y_id_base = torch.full((in_len, 1), -1 * self.args.conf + noise, dtype=torch.float)
            #self.train_y_ood_base = torch.full((out_len, 1), 1 * self.args.conf + noise, dtype=torch.float)
        elif epoch >= self.args.BUF_SIZE:
            cyclic_id = epoch % self.args.BUF_SIZE
            start_idx = cyclic_id * out_len
        train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))
        for i, (in_set, out_set) in enumerate(zip(train_loader_in, train_loader_out)):
            in_len = len(in_set[0])
            out_len = len(out_set[0])
            if epoch == 0:
                self.train_x_id_base = torch.cat((self.train_x_id_base, in_set[0]), dim=0)
            if epoch < self.args.BUF_SIZE:
                self.train_x_ood = torch.cat((self.train_x_ood, out_set[0]), dim=0)
            else:
                replace_idx = np.arange(start_idx, start_idx + out_len)
                self.train_x_ood[replace_idx] = out_set[0]
            in_input = in_set[0].cuda()
            in_target = in_set[1].cuda()
            out_input = out_set[0].cuda()
            out_target = out_set[1].cuda()

            self.model.train()

            cat_input = torch.cat((in_input, out_input), 0)
            cat_output = self.model(cat_input)
            in_output = cat_output[:in_len]
            out_output = cat_output[in_len:]

            in_loss = criterion(in_output, in_target)
            in_losses.update(in_loss.data, in_len)

            E = -torch.logsumexp(cat_output, dim=1)
            Ec_in = E[:in_len]
            Ec_out = E[in_len:]
            in_energy_loss = torch.pow(F.relu(Ec_in - self.args.m_in), 2).mean()
            out_energy_loss = torch.pow(F.relu(self.args.m_out - Ec_out), 2).mean()
            in_energy_losses.update(in_energy_loss.data, in_len)
            out_energy_losses.update(out_energy_loss.data, out_len)
            loss = in_loss + self.args.energy_beta * (out_energy_loss + in_energy_loss)
            nat_prec1 = accuracy(in_output.data, in_target, topk=(1,))[0]

            nat_top1.update(nat_prec1, in_len)

            scheduler.step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.args.print_freq == 0:
                log.debug('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'In Loss {in_loss.val:.4f} ({in_loss.avg:.4f})\t'
                          'Prec@1 {nat_top1.val:.3f} ({nat_top1.avg:.3f})\t'
                          'InE Loss {in_e_loss.val:.4f} ({in_e_loss.avg:.4f})\t'
                          'OutE Loss {out_e_loss.val:.4f} ({out_e_loss.avg:.4f})\t'.format(
                    epoch, i, len(train_loader_in), batch_time=batch_time,
                    in_loss=in_losses,
                    in_e_loss=in_energy_losses, nat_top1=nat_top1,
                    out_e_loss=out_energy_losses))

        #this is added to make the length of train_y_ood equal to that of train_x_ood when ood loader is not scanned through
        if epoch==0:
            self.train_y_ood_base = torch.full((len(self.train_x_ood), 1), 1 * self.args.conf + noise, dtype=torch.float)

        if epoch < self.args.BUF_SIZE:
            self.train_x_id = self.train_x_id_base.repeat(epoch + 1, 1, 1, 1)
            self.train_y_id = self.train_y_id_base.repeat(epoch + 1, 1)
            self.train_y_ood = self.train_y_ood_base.repeat(epoch + 1, 1)
        self.train_x = torch.cat((self.train_x_id, self.train_x_ood), dim=0)
        self.train_y = torch.cat((self.train_y_id, self.train_y_ood), dim=0)

    def sample_BDQN(self):
        # Sample sigma^2, and beta conditional on sigma^2
        with torch.no_grad():
            d = self.mu_w[0].shape[0]
            try:
                for i in range(self.output_dim):
                    mus = self.mu_w[i].double()
                    covs = self.cov_w[i][np.newaxis, :, :].double()
                    multivariates = MultivariateNormal(mus, covs[0]).sample().reshape(1, -1)
                    if i == 0:
                        beta_s = multivariates
                    else:
                        beta_s = torch.cat((beta_s, multivariates), dim=0)
            except Exception as e:
                print('Err in Sampling BDQN Details:', e)
                multivariates = MultivariateNormal(torch.zeros(d), torch.eye(d)).sample().reshape(1, -1)
                if i == 0:
                    beta_s = multivariates
                else:
                    beta_s = torch.cat((beta_s, multivariates), dim=0)
            self.beta_s = beta_s.float()

    def predict(self, x):
        latent_z = self.model.get_representation(x)
        return torch.matmul(self.beta_s, latent_z.T).T

    def update_bays_reg_BDQN(self, log):
        with torch.no_grad():
            log.debug("######## Start updating bayesian linear layer ########")
            # Update action posterior with formulas: \beta | z,y ~ N(mu_q, cov_q)
            z = self.latent_z.double()
            y = self.train_y.squeeze().cuda()
            s = torch.matmul(z.T, z)
            A = s / self.sigma_n + 1 / self.sigma * self.eye
            B = torch.matmul(z.T, y.double()) / self.sigma_n
            # A_eig_val, A_eig_vec = torch.symeig(A, eigenvectors=True)
            inv = torch.inverse(A.double())
            self.mu_w[0] = torch.matmul(inv, B).squeeze()
            temp_cov = self.sigma * inv
            eig_val, eig_vec = torch.symeig(temp_cov, eigenvectors=True)
            # log.debug("After inverse. eigenvalue (pd): {}".format(eig_val[:20]) )
            # log.debug("After inverse. eigenvalue (pd): {}".format(eig_val[-20:]) )
            if torch.any(eig_val < 0):
                self.cov_w[0] = torch.matmul(torch.matmul(eig_vec, torch.diag(torch.abs(eig_val))), torch.t(eig_vec))
            else:
                self.cov_w[0] = temp_cov

    def validate(self, val_loader, model, criterion, epoch, log, energy_model=False):
        log.debug("######## Start validating at epoch {} ########".format(epoch))
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        in_energy = AverageMeter()
        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            loss = criterion(output, target.to(output.device))
            Ec_in = -torch.logsumexp(output, dim=1).mean()
            in_energy.update(Ec_in.data, input.size(0))
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target, topk=(1,))[0]
            losses.update(loss.data, input.size(0))
            top1.update(prec1, input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.args.print_freq == 0:
                log.debug('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'In E {e.val:.4f} ({e.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    e=in_energy, top1=top1))

        log.debug(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
        return top1.avg

    def train_blr_adv(self, train_loader_in, train_loader_out, criterion, optimizer,scheduler, epoch,log,args):
        batch_time = AverageMeter()
        out_confs = AverageMeter()
        in_confs = AverageMeter()
        in_losses = AverageMeter()
        out_losses = AverageMeter()
        out_energy_losses = AverageMeter()
        in_energy_losses = AverageMeter()
        nat_top1 = AverageMeter()
        log.debug("######## Start training NN at epoch {} ########".format(epoch))
        end = time.time()
        out_len = len(train_loader_out.dataset)
        in_len = len(train_loader_in.dataset)
        if epoch == 0:
            noise = torch.normal(0.0, self.args.sigma_n ** 0.5, size=(1,)).item()
            self.train_y_id_base = torch.full((in_len, 1), -1 * self.args.conf + noise, dtype=torch.float)
            self.train_y_ood_base = torch.full((out_len, 1), 1 * self.args.conf + noise, dtype=torch.float)
        elif epoch >= self.args.BUF_SIZE:
            cyclic_id = epoch % self.args.BUF_SIZE
            start_idx = cyclic_id * out_len
        train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))
        for i, (in_set, out_set) in enumerate(zip(train_loader_in, train_loader_out)):
            in_len = len(in_set[0])
            out_len = len(out_set[0])
            if epoch == 0:
                self.train_x_id_base = torch.cat((self.train_x_id_base, in_set[0]), dim=0)
            if epoch < self.args.BUF_SIZE:
                self.train_x_ood = torch.cat((self.train_x_ood, out_set[0]), dim=0)
            else:
                replace_idx = np.arange(start_idx, start_idx + out_len)
                self.train_x_ood[replace_idx] = out_set[0]
            in_input = in_set[0].cuda()
            in_target = in_set[1].cuda()
            #out_input = out_set[0].cuda()

            aug_length = int(len(out_set[0]) * args.augment_ratio)
            adv_outlier = PGD(self.model, out_set[0][:aug_length], args)

            out_target = out_set[1].cuda()

            self.model.train()

            cat_input = torch.cat((in_input,adv_outlier,out_set[0][aug_length:].cuda()), 0)
            cat_output = self.model(cat_input)
            in_output = cat_output[:in_len]
            out_output = cat_output[in_len:]

            in_loss = criterion(in_output, in_target)
            in_losses.update(in_loss.data, in_len)

            E = -torch.logsumexp(cat_output, dim=1)
            Ec_in = E[:in_len]
            Ec_out = E[in_len:]
            in_energy_loss = torch.pow(F.relu(Ec_in - self.args.m_in), 2).mean()
            out_energy_loss = torch.pow(F.relu(self.args.m_out - Ec_out), 2).mean()
            in_energy_losses.update(in_energy_loss.data, in_len)
            out_energy_losses.update(out_energy_loss.data, out_len)
            loss = in_loss + self.args.energy_beta * (out_energy_loss + in_energy_loss)
            nat_prec1 = accuracy(in_output.data, in_target, topk=(1,))[0]

            nat_top1.update(nat_prec1, in_len)

            scheduler.step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.args.print_freq == 0:
                log.debug('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'In Loss {in_loss.val:.4f} ({in_loss.avg:.4f})\t'
                          'Prec@1 {nat_top1.val:.3f} ({nat_top1.avg:.3f})\t'
                          'InE Loss {in_e_loss.val:.4f} ({in_e_loss.avg:.4f})\t'
                          'OutE Loss {out_e_loss.val:.4f} ({out_e_loss.avg:.4f})\t'.format(
                    epoch, i, len(train_loader_in), batch_time=batch_time,
                    in_loss=in_losses,
                    in_e_loss=in_energy_losses, nat_top1=nat_top1,
                    out_e_loss=out_energy_losses))

        if epoch < self.args.BUF_SIZE:
            self.train_x_id = self.train_x_id_base.repeat(epoch + 1, 1, 1, 1)
            self.train_y_id = self.train_y_id_base.repeat(epoch + 1, 1)
            self.train_y_ood = self.train_y_ood_base.repeat(epoch + 1, 1)
        self.train_x = torch.cat((self.train_x_id, self.train_x_ood), dim=0)
        self.train_y = torch.cat((self.train_y_id, self.train_y_ood), dim=0)


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
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t().to(target.device)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.labels = labels
        self.images = images
        self.offset = 0
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        index = (index + self.offset) % len(self.images)
        # Load data and get label
        X = self.images[index]
        if self.transform:
            X = self.transform(X)
        y = self.labels[index]

        return X, y


class customTinyImageNet(torch.utils.data.Dataset):
    def __init__(self, tiny_imagenet):
        self.tiny_imagenet = tiny_imagenet
        self.offset = 0  # offset index

    def __len__(self):
        return len(self.tiny_imagenet)

    def __getitem__(self, index):
        index = (index + self.offset) % len(self.tiny_imagenet)

        img, _ = self.tiny_imagenet[index]

        return img, 0  # 0 is the class

def mixup_outlier_for_one_batch(x, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x

def cutmix_outlier_for_one_batch(x,alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).cuda()
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    # actual_lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
    return x

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def PGD(model, data, args,rand_init=True):
    epsilon=args.epsilon
    rel_step_size=args.rel_step_size
    num_steps=args.num_steps

    model.eval()
    data=data.cuda()
    #nat_output = model(data)
    #with torch.no_grad():
    #    output=model(data)
    #    print(f'before adv attack,MSP score is {torch.max(torch.softmax(output,dim=1),dim=1)[0][0]}')
    x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for k in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)
        model.zero_grad()
        with torch.enable_grad():
            if args.attack_score=='MSP':
                loss_adv = -(output.mean(1) - torch.logsumexp(output, dim=1)).mean()
                #criterion_kl = nn.KLDivLoss(size_average=False).cuda()
                #loss_adv += args.kl_loss_weight * criterion_kl(F.log_softmax(output, dim=1), F.softmax(nat_output, dim=1))
            elif args.attack_score=='energy':
                Ec_out = -torch.logsumexp(output, dim=1)
                loss_adv = torch.pow(F.relu(args.m_out-Ec_out), 2).mean()
        loss_adv.backward()
        step_size=epsilon*rel_step_size
        eta = step_size * x_adv.grad.sign()
        # Update adversarial data
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    x_adv = Variable(x_adv, requires_grad=False)
    #with torch.no_grad():
    #    output=model(x_adv)
    #    print(f'after adv attack,MSP score is {torch.max(torch.softmax(output,dim=1),dim=1)[0][0]}')
    model.train()
    return x_adv