#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import urllib.request

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

import timm
from timm.models.layers import trunc_normal_

from simsiam.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

from PIL import Image

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
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4096, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
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

# additional configs:
parser.add_argument('--pretrained', default='', type=str,
                    help='path to simsiam pretrained checkpoint')
parser.add_argument('--lars', action='store_true',
                    help='Use LARS')
parser.add_argument('--last_dim', default=-1, type=int,
                    help='feature dimension (default: -1)')

## ava_gpu
parser.add_argument('--ava-gpu', default=None, type=str, help='ava GPU id to use. 1 or 1,2')
parser.add_argument('--exp_dir', type=str, help='path to experiment directory')
parser.add_argument('--filename1', default='checkpoint.pth.tar', type=str, help='filename1')
parser.add_argument('--filename2', default='model_best.pth.tar', type=str, help='filename2')
parser.add_argument('--hp1', default=1.0, type=float, help='hp1')
parser.add_argument('--else_part', default='main_part', type=str, choices=['main_part', 'else_part'])
parser.add_argument('--net', default='simsiam', type=str)
parser.add_argument('--avg_size', default=1, type=int, help='1, 3, 4')
parser.add_argument('--li_aug', default='_li_aug1', type=str, choices=['_li_aug1', '_li_aug1_2', '_li_aug1_4_2'])
## dataset_type
parser.add_argument('--dataset_type', default='Imagenet', type=str, choices=['Imagenet', 'IN100', "Flowers102", "Food101", "OxfordIIITPet", "CUB200"])
parser.add_argument('--ds_mode', default='linear', type=str) #, choices=['linear', 'semi_1', 'semi_10', 'ds_ft'])
parser.add_argument('--backbone_lr', default=0.1, type=float, help='backbone lr for semi/finetune')


def get_backbone(backbone_name, num_cls=10, hp1=1.0, else_part = 'main_part', args = None):
    models = {'resnet18': ResNet18(low_dim=num_cls, hp1=hp1, else_part=else_part, args=args),
              'resnet34': ResNet34(low_dim=num_cls, hp1=hp1, else_part=else_part, args=args),
              'resnet50': ResNet50(low_dim=num_cls, hp1=hp1, else_part=else_part, args=args),
              'resnet101': ResNet101(low_dim=num_cls, hp1=hp1, else_part=else_part, args=args),
              'resnet152': ResNet152(low_dim=num_cls, hp1=hp1, else_part=else_part, args=args)}

    return models[backbone_name]

best_acc1 = 0

def main():
    args = parser.parse_args()

    args.if_pretrain = False
    print(vars(args))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.ava_gpu)

    if args.ds_mode.startswith("semi"):
        args.semi_para = int(args.ds_mode.split("_")[1])#/100
        print("args.ds_mode, args.semi_para", args.ds_mode, args.semi_para)
    else:
         print("args.ds_mode", args.ds_mode)

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
    global best_acc1
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
    if args.dataset_type == "Imagenet":
        args.num_cls = 1000
    elif args.dataset_type == "IN100":
        args.num_cls = 100
    elif args.dataset_type == "Flowers102":
        args.num_cls = 102
    elif args.dataset_type == "Food101":
        args.num_cls = 101
    elif args.dataset_type == "OxfordIIITPet":
        args.num_cls = 37
    elif args.dataset_type == "CUB200":
        args.num_cls = 200
    print("args.num_cls", args.num_cls)

    if args.net == "simsiam":
        model = models.__dict__[args.arch](num_classes=args.num_cls) 
    elif args.net == "simsiam_last1638" or args.net == "simsiam_last2048":
        model = get_backbone(args.arch, args.num_cls, hp1 = args.hp1, else_part=args.else_part, args = args)
    else:
        model = get_backbone(args.arch, args.num_cls, hp1 = args.hp1, else_part=args.else_part, args = args)

    # freeze all layers but the last fc
    if args.ds_mode == "linear":
        for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
        print("backbone frozen")
    else:
        print("all not be frozen, now semi")
    # print("tmp print model", model)

    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    # load from pre-trained, before DistributedDataParallel constructor
    print("assert args.pretrained", args.pretrained)
    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder up to before the embedding layer
                if k.startswith('module.encoder') and not k.startswith('module.encoder.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
    
            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"} and msg.unexpected_keys == []

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256
    if args.ds_mode.startswith("semi") or args.ds_mode == "ds_ft":
        init_backbone_lr = args.backbone_lr * args.batch_size / 256
        print("init_backbone_lr, init_lr", init_backbone_lr, init_lr)

    if args.distributed:
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
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    print(model) # print model after SyncBatchNorm
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.ds_mode.startswith("semi") or args.ds_mode == "ds_ft":
        classifier_parameters, model_parameters = [], []
        for name, param in model.named_parameters():
            if name in {'fc.weight', 'fc.bias'}:
                classifier_parameters.append(param)
            else:
                model_parameters.append(param)
        param_groups = [dict(params=classifier_parameters, lr=init_lr)]
        param_groups.append(dict(params=model_parameters, lr=init_backbone_lr))
        optimizer = torch.optim.SGD(param_groups, 0, momentum=0.9, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        print("len(parameters)", len(parameters))

    elif args.ds_mode == "linear":
        # optimize only the linear classifier
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        print("len(parameters)", len(parameters))
        assert len(parameters) == 2  # fc.weight, fc.bias

        optimizer = torch.optim.SGD(parameters, init_lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    if args.lars:
        print("=> use LARS optimizer.")
        from apex.parallel.LARC import LARC
        optimizer = LARC(optimizer=optimizer, trust_coefficient=.001, clip=False)
    else:
        print("=> use SGD optimizer.")

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.dataset_type == "Imagenet" or args.dataset_type == "IN100" or args.dataset_type == "CUB200": ## 其中 semi 目前只适用于 Imagenet 1k，其他的都只能用linear
        if args.li_aug == "_li_aug1":
            train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
            print("_li_aug1 activated")
        elif args.li_aug == "_li_aug1_2":
            train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(degrees=(-90, 90)),
                    transforms.ToTensor(),
                    normalize,
                ]))
            print("_li_aug1_2 activated")
        elif args.li_aug == "_li_aug1_4_2":
            train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(degrees=(-90, 90)),
                    transforms.RandomApply([
                        transforms.ElasticTransform(alpha=100.0)
                    ], p=0.5),
                    transforms.ToTensor(),
                    normalize,
                ]))
            print("_li_aug1_4_2 activated")
    
        if args.ds_mode.startswith("semi") and (args.semi_para in {1, 10}): ## semi 目前只适用于 Imagenet 1k，其他的都只能用linear
            if args.dataset_type == "Imagenet":
                # args.train_files = urllib.request.urlopen(f'https://raw.githubusercontent.com/google-research/simclr/master/imagenet_subsets/{args.semi_para}percent.txt').readlines()
                args.train_files = open(f'/home/sifan2/ImageNet/semi_files/Imagenet_{args.semi_para}percent.txt').readlines()
            elif args.dataset_type == "IN100":
                args.train_files = open('/home/sifan2/ImageNet/semi_files/IN100_'+str(args.semi_para)+'percent.txt').readlines()
    
            train_dataset.samples = []
            for fname in args.train_files:
                # fname = fname.decode().strip() ## urllib.request.urlopen use this line
                fname = fname.strip()
                cls = fname.split('_')[0]
                train_dataset.samples.append(
                    (traindir +"/"+ cls +"/"+ fname, train_dataset.class_to_idx[cls]))
            print("percent ImageNet has been loaded for semi:", args.dataset_type)
        elif args.ds_mode == "ds_ft":
            print("ds_ft training for", args.dataset_type)
        else:
            print("linear training for", args.dataset_type)
    
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None
    
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=256, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    else: ## 这里仅适用 linear 的 downstream 其他数据集

        if args.li_aug == "_li_aug1":
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            print("_li_aug1 activated")
        elif args.li_aug == "_li_aug1_2":
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=(-90, 90)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            print("_li_aug1_2 activated")
        elif args.li_aug == "_li_aug1_4_2":
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=(-90, 90)),
                transforms.RandomApply([
                    transforms.ElasticTransform(alpha=100.0)
                ], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            print("_li_aug1_4_2 activated")

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        if args.dataset_type == "Flowers102":
            train_dataset = datasets.Flowers102(root="/home/sifan2/Datasets/", split='train', download=True, transform=transform_train) # finished
            val_dataset = datasets.Flowers102(root="/home/sifan2/Datasets/", split='test', download=True, transform=transform_test) # finished
            print("Flowers102 has been loaded")
        elif args.dataset_type == "Food101":
            train_dataset = datasets.Food101(root="/home/sifan2/Datasets/", split='train', download=True, transform=transform_train) # finished
            val_dataset = datasets.Food101(root="/home/sifan2/Datasets/", split='test', download=True, transform=transform_test) # finished
            print("Food101 has been loaded")
        elif args.dataset_type == "OxfordIIITPet": ## 37 category
            train_dataset = datasets.OxfordIIITPet(root="/home/sifan2/Datasets/", split='trainval', download=True, transform=transform_train) # finished
            val_dataset = datasets.OxfordIIITPet(root="/home/sifan2/Datasets/", split='test', download=True, transform=transform_test) # finished
            print("OxfordIIITPet has been loaded")

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=256, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if args.ds_mode == "linear": ## "linear" here, "semi" using scheduler
            adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        if args.ds_mode.startswith("semi") or args.ds_mode == "ds_ft":
            scheduler.step()
            pg = optimizer.param_groups
            lr_classifier = pg[0]['lr']
            lr_backbone = pg[1]['lr'] if len(pg) == 2 else 0
            print("in semi: lr_classifier, lr_backbone", lr_classifier, lr_backbone)
        else:
            for param_group in optimizer.param_groups:
                lr_classifier = param_group['lr']
            print("in linear: lr_classifier", lr_classifier)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args)
            if epoch == args.start_epoch:
                if args.ds_mode == "linear":
                    sanity_check(model.state_dict(), args.pretrained)
                else:
                    print("semi, no sanity_check")

    print('Best acc:', best_acc1)

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    if args.ds_mode.startswith("semi") or args.ds_mode == "ds_ft":
        model.train()
    elif args.ds_mode == "linear":
        model.eval()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    filename = args.filename1
    torch.save(state, args.exp_dir+"/"+filename)
    if is_best:
        shutil.copyfile(args.exp_dir+"/"+filename, args.exp_dir+"/"+args.filename2)


def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'module.encoder.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.encoder.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


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
        param_group['lr'] = cur_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
