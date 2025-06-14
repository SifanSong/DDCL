#!/usr/bin/env python
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import simsiam.loader

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
#parser.add_argument('data', metavar='DIR',
#                    help='path to dataset')
parser.add_argument('--data', type=str, help='path to dataset directory')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# simsiam specific configs:
parser.add_argument('--dim', default=2048, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--pred-dim', default=512, type=int,
                    help='hidden dimension of the predictor (default: 512)')
parser.add_argument('--fix-pred-lr', action='store_true',
                    help='Fix learning rate for the predictor')
parser.add_argument('--last_dim', default=-1, type=int,
                    help='feature dimension (default: -1)')

## ava_gpu
parser.add_argument('--ava-gpu', default=None, type=str, help='ava GPU id to use. 1 or 0,1,2')
parser.add_argument('--exp_dir', type=str, help='path to experiment directory')
parser.add_argument('--trial', type=str, default='1', help='trial id')
parser.add_argument('--save_freq', default=100, type=int, help='save model frequency')
parser.add_argument('--aug', default='aug1', type=str, choices=['aug1', 'aug1_2', 'aug1_4_2'])
parser.add_argument('--net', default='simsiam', type=str)
parser.add_argument('--hp1', default=1.0, type=float, help='hp1')
parser.add_argument('--lambd', default=0.005, type=float, help='lambd')
## SepCL_v1
parser.add_argument('--sep_lambd', default="1.0", type=str, help='sep_lambd, 1.0, or 1.0-300-300')
parser.add_argument('--avg_size', default=1, type=int, help='1, 3, 4')
parser.add_argument('--projector_img', default='2048-2048-2048', type=str, help='projector_img')
parser.add_argument('--projector_aug', default='2048-2048-2048', type=str, help='projector_aug')
parser.add_argument('--reg_lambd', default="_reg0.001", type=str, help='_reg0.001 default for L1/L2')
## warmup
parser.add_argument('--warmup_epochs', type=int, default=0, help='number of warmup_epochs')
## dataset_type
# parser.add_argument('--dataset_type', default='Imagenet', type=str, choices=['Imagenet', 'IN100'])

def main():
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.ava_gpu)

    args.trial_dir = os.path.join(args.exp_dir, args.trial)
    if not os.path.exists(args.trial_dir):
        os.makedirs(args.trial_dir)

    print("args.sep_lambd", args.sep_lambd)
    if "-" in args.sep_lambd:
        args.sep_lambd = list(map(float, args.sep_lambd.split('-')))
    else:
        args.sep_lambd = float(args.sep_lambd)

    args.reg_lambd = float(args.reg_lambd.split("reg")[1])
    print("args.reg_lambd", args.reg_lambd)

    args.if_pretrain = True
    print(vars(args))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
    # create model
    print("=> creating model '{}'".format(args.arch))

    if args.net == "simsiam":
        import simsiam.builder
        model = simsiam.builder.SimSiam(
            models.__dict__[args.arch],
            args.dim, args.pred_dim, args=args, version = "torchvision")
    elif args.net == "simsiam_last1638" or args.net == "simsiam_last2048":
        import simsiam.builder
        model = simsiam.builder.SimSiam(None, args.dim, args.pred_dim, args)
    elif args.net == "DDCL" or args.net == "CLeVER_DDCL":
        import simsiam.DRL_builder
        model = simsiam.DRL_builder.DRL_Simsiam(args.dim, args.pred_dim, args)

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    if args.distributed:
        # Apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    print(model) # print model after SyncBatchNorm

    # define loss function (criterion) and optimizer
    criterion = nn.CosineSimilarity(dim=1).cuda(args.gpu)
    print("Loss: CosineSimilarity loaded")
    L1 = nn.L1Loss()

    if args.fix_pred_lr:
        print("fixed lr")
        if args.net == "simsiam" or args.net == "simsiam_last1638" or args.net == "simsiam_last2048":
            optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                            {'params': model.module.predictor.parameters(), 'fix_lr': True}]
        elif args.net == "DDCL" or args.net == "CLeVER_DDCL":
            optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                            {'params': model.module.projector_img.parameters(), 'fix_lr': False},
                            {'params': model.module.projector_aug.parameters(), 'fix_lr': False},
                            {'params': model.module.predictor_img.parameters(), 'fix_lr': True},
                            {'params': model.module.predictor_aug.parameters(), 'fix_lr': True}]
    else:
        print("all lr are not fixed")
        optim_params = model.parameters()

    optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        args.warmup_epochs = 0
        print("args.warmup_epochs has been changed to", args.warmup_epochs)
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    if args.aug == "aug1":
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    elif args.aug == "aug1_2":
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(-90, 90)),
            transforms.ToTensor(),
            normalize
        ]

    elif args.aug == "aug1_4_1":
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(-90, 90)),
            transforms.RandomApply([
                transforms.ElasticTransform(alpha=50.0)
            ], p=0.5),
            transforms.ToTensor(),
            normalize
        ]

    elif args.aug == "aug1_4_2":
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(-90, 90)),
            transforms.RandomApply([
                transforms.ElasticTransform(alpha=100.0)
            ], p=0.5),
            transforms.ToTensor(),
            normalize
        ]

    elif args.aug == "aug1_4_3":
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(-90, 90)),
            transforms.RandomApply([
                transforms.ElasticTransform(alpha=150.0)
            ], p=0.5),
            transforms.ToTensor(),
            normalize
        ]
        
    print("args.aug", args.aug)

    train_dataset = datasets.ImageFolder(
        traindir,
        simsiam.loader.TwoCropsTransform(transforms.Compose(augmentation)))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)


    if args.warmup_epochs > 0:
        optimizer2 = torch.optim.SGD(model.parameters(),
                      lr=1e-9,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay)

        for wp_epoch in range(0, args.warmup_epochs):
            adjust_learning_rate_wp(optimizer2, init_lr, wp_epoch, args)
            print("Warmming up...")
            train(train_loader, model, criterion, L1, optimizer2, wp_epoch, args)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, L1, optimizer, epoch, args)

        #if not args.multiprocessing_distributed or (args.multiprocessing_distributed
        #        and args.rank % ngpus_per_node == 0):
        if (epoch+1) % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename=os.path.join(args.trial_dir, 'ckpt_epoch_{}_{}.pth'.format(epoch, args.trial)))

    save_checkpoint({
        'epoch': epoch + 1,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
    }, is_best=False, filename=os.path.join(args.trial_dir, '{}_last.pth'.format(args.trial)))

def train(train_loader, model, criterion, L1, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')

    if args.net == "simsiam" or args.net == "simsiam_last1638" or args.net == "simsiam_last2048":
        losses1 = AverageMeter('Loss1', ':.4e')
        losses2 = AverageMeter('Loss2', ':.4e')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, losses1, losses2],
            prefix="Epoch: [{}]".format(epoch))
    elif args.net == "DDCL":
        losses_img = AverageMeter('Loss_img', ':.4e')
        losses_aug = AverageMeter('Loss_aug', ':.4e')
        losses1_img = AverageMeter('Loss1_img', ':.4e')
        losses2_img = AverageMeter('Loss2_img', ':.4e')
        losses1_aug = AverageMeter('Loss1_aug', ':.4e')
        losses2_aug = AverageMeter('Loss2_aug', ':.4e')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, losses_img, losses_aug, losses1_img, losses2_img, losses1_aug, losses2_aug],
            prefix="Epoch: [{}]".format(epoch))
    elif args.net == "CLeVER_DDCL":
        losses_img = AverageMeter('Loss_img', ':.4e')
        losses_aug = AverageMeter('Loss_aug', ':.4e')
        losses_reg_proj = AverageMeter('Loss_reg_proj', ':.4e')
        losses_reg_pred = AverageMeter('Loss_reg_pred', ':.4e')
        losses1_img = AverageMeter('Loss1_img', ':.4e')
        losses2_img = AverageMeter('Loss2_img', ':.4e')
        losses1_aug = AverageMeter('Loss1_aug', ':.4e')
        losses2_aug = AverageMeter('Loss2_aug', ':.4e')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, losses_img, losses_aug, losses_reg_proj, losses_reg_pred, losses1_img, losses2_img, losses1_aug, losses2_aug],
            prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        if args.net == "simsiam" or args.net == "simsiam_last1638" or args.net == "simsiam_last2048":
            # compute output and loss
            p1, p2, z1, z2 = model(x1=images[0], x2=images[1])
            #loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
            loss1 = - criterion(p1, z2).mean()
            loss2 = - criterion(p2, z1).mean()
            loss = 0.5 * loss1 + 0.5 * loss2
            loss_list = [loss, 0.5 * loss1, 0.5 * loss2]

            losses.update(loss_list[0].item(), images[0].size(0))
            losses1.update(loss_list[1].item(), images[0].size(0))
            losses2.update(loss_list[2].item(), images[0].size(0))

        elif args.net == "DDCL" or args.net == "CLeVER_DDCL":
            z1_img, z2_img, z1_aug, z2_aug, p1_img, p2_img, p1_aug, p2_aug, _,_,_,_ = model(im_aug1=images[0], im_aug2=images[1])

            loss1_img = - criterion(p1_img, z2_img).mean()
            loss2_img = - criterion(p2_img, z1_img).mean()
            loss1_aug = criterion(p1_aug, z2_aug).mean()
            loss2_aug = criterion(p2_aug, z1_aug).mean()
            
            loss_img = 0.5 * loss1_img + 0.5 * loss2_img
            loss_aug = 0.5 * loss1_aug + 0.5 * loss2_aug
            loss_aug = L1(loss_aug, 0*loss_aug)

            if args.net == "CLeVER_DDCL":
                reg_projector_img = sum(p.pow(2.0).sum() for p in model.module.projector_img.parameters())
                reg_projector_aug = sum(p.pow(2.0).sum() for p in model.module.projector_aug.parameters())
                reg_predictor_img = sum(p.pow(2.0).sum() for p in model.module.predictor_img.parameters())
                reg_predictor_aug = sum(p.pow(2.0).sum() for p in model.module.predictor_aug.parameters())

                loss_reg_projector = (reg_projector_img - reg_projector_aug)
                loss_reg_predictor = (reg_predictor_img - reg_predictor_aug)
                loss_reg_projector = L1(loss_reg_projector, 0*loss_reg_projector)
                loss_reg_predictor = L1(loss_reg_predictor, 0*loss_reg_predictor)
                loss = loss_img + args.sep_lambd*loss_aug + args.reg_lambd*(loss_reg_projector + loss_reg_predictor)/2
                loss_list = [loss, loss_img, loss1_img/2, loss2_img/2, loss_aug, loss1_aug/2, loss2_aug/2, loss_reg_projector, loss_reg_predictor]

                losses.update(loss_list[0].item(), images[0].size(0))
                losses_img.update(loss_list[1].item(), images[0].size(0))
                losses_aug.update(loss_list[4].item(), images[0].size(0))
                losses_reg_proj.update(loss_list[7].item(), images[0].size(0))
                losses_reg_pred.update(loss_list[8].item(), images[0].size(0))
                losses1_img.update(loss_list[2].item(), images[0].size(0))
                losses2_img.update(loss_list[3].item(), images[0].size(0))
                losses1_aug.update(loss_list[5].item(), images[0].size(0))
                losses2_aug.update(loss_list[6].item(), images[0].size(0))
            else:
                loss = loss_img + args.sep_lambd * loss_aug ## default 1.0
                loss_list = [loss, loss_img, 0.5 * loss1_img, 0.5 * loss2_img, loss_aug, 0.5 * loss1_aug, 0.5 * loss2_aug]

                losses.update(loss_list[0].item(), images[0].size(0))
                losses_img.update(loss_list[1].item(), images[0].size(0))
                losses_aug.update(loss_list[4].item(), images[0].size(0))
                losses1_img.update(loss_list[2].item(), images[0].size(0))
                losses2_img.update(loss_list[3].item(), images[0].size(0))
                losses1_aug.update(loss_list[5].item(), images[0].size(0))
                losses2_aug.update(loss_list[6].item(), images[0].size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_list[0].backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr

def adjust_learning_rate_wp(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = 1e-9
    # cosine lr schedule
    lr += init_lr * (epoch/args.warmup_epochs) #0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    #print(epoch, lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def normalize_channels( _x, inplace=False):
    #tmp = torch.flatten(_x, start_dim=2)
    _min = _x.min(dim=-1, keepdim=True)[0]
    _max = _x.max(dim=-1, keepdim=True)[0]
    #print("_min, _max", _min, _max)
    #del tmp
    if inplace:
        return _x.add_(_min, alpha=-1).mul_(1 / _max.add(_min, alpha=-1))
    else:
        return _x.add(_min, alpha=-1).mul(1 / _max.add(_min, alpha=-1))


def normalize_channels_mul( _x, inplace=False):
    #tmp = torch.flatten(_x, start_dim=2)
    _min = _x.min(dim=-1, keepdim=True)[0]
    _max = _x.max(dim=-1, keepdim=True)[0]
    #print("_min, _max", _min, _max)
    #del tmp
    if inplace:
        return _x.mul_(1 / _max.abs().add(1e-38))
    else:
        return _x.mul(1 / _max.abs().add(1e-38))


if __name__ == '__main__':
    main()
