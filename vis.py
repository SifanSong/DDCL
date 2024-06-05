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

from simsiam.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from PIL import Image

from collections import OrderedDict
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.nn.functional import sigmoid
import imageio
import cv2
import numpy as np

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
parser.add_argument('--net', default='simsiam', type=str)
#parser.add_argument('--lars', action='store_true',
#                    help='Use LARS')
parser.add_argument('--last_dim', default=-1, type=int,
                    help='feature dimension (default: -1)')

## ava_gpu
#parser.add_argument('--ava-gpu', default=None, type=str, help='ava GPU id to use. 1 or 1,2')
parser.add_argument('--exp_dir', type=str, help='path to experiment directory')
parser.add_argument('--filename1', default='checkpoint.pth.tar', type=str, help='filename1')
parser.add_argument('--filename2', default='model_best.pth.tar', type=str, help='filename2')
parser.add_argument('--hp1', default=1.0, type=float, help='hp1')
parser.add_argument('--else_part', default='main_part', type=str, choices=['main_part', 'else_part'])
parser.add_argument('--test_aug_type', default='basic', type=str, help='test_aug_type')
parser.add_argument('--vis_output_dir', default=None, type=str, help='vis_output_dir')
parser.add_argument('--avg_size', default=1, type=int, help='1, 3, 4')
## dataset_type
parser.add_argument('--dataset_type', default='Imagenet', type=str, choices=['Imagenet', 'IN100'])

parser.add_argument('--only_aug_vis', default="only_aug_vis", type=str, choices=['only_aug_vis', 'cam'])

def get_backbone(backbone_name, num_cls=10, hp1=1.0, else_part = 'main_part', args = None):
    models = {'resnet18': ResNet18(low_dim=num_cls, hp1=hp1, else_part=else_part, args=args),
              'resnet34': ResNet34(low_dim=num_cls, hp1=hp1, else_part=else_part, args=args),
              'resnet50': ResNet50(low_dim=num_cls, hp1=hp1, else_part=else_part, args=args),
              'resnet101': ResNet101(low_dim=num_cls, hp1=hp1, else_part=else_part, args=args),
              'resnet152': ResNet152(low_dim=num_cls, hp1=hp1, else_part=else_part, args=args)}

    return models[backbone_name]

def main():
    args = parser.parse_args()
    args.if_pretrain = False
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
    #global best_acc1
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
    print("args.num_cls", args.num_cls)

    if args.net == "simsiam":
        model = models.__dict__[args.arch](num_classes=args.num_cls)
    else:
        model = get_backbone(args.arch, args.num_cls, hp1 = args.hp1, else_part=args.else_part, args = args)

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


    if args.test_aug_type == "basic":
        test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif args.test_aug_type == "basic_aug1_4_2":
        test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomApply([
                transforms.ElasticTransform(alpha=100.0)
            ], p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif args.test_aug_type == "aug1":
        test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            #transforms.RandomResizedCrop(args.img_dim, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(p=1.0),
            #transforms.RandomApply([
            #    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            #], p=0.8),
            #transforms.RandomGrayscale(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif args.test_aug_type == "aug1_2":
        test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            #transforms.RandomResizedCrop(args.img_dim, scale=(0.2, 1.)),
            #transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomRotation(degrees=(45, 45)),
            #transforms.RandomApply([
            #    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            #], p=0.8),
            #transforms.RandomGrayscale(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif args.test_aug_type == "aug1_3":
        test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            #transforms.RandomResizedCrop(args.img_dim, scale=(0.2, 1.)),
            #transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomRotation(degrees=(90, 90)),
            #transforms.RandomApply([
            #    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            #], p=0.8),
            #transforms.RandomGrayscale(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif args.test_aug_type == "aug1_3_aug1_4_2":
        test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomRotation(degrees=(90, 90)),
            transforms.RandomApply([
                transforms.ElasticTransform(alpha=100.0)
            ], p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif args.test_aug_type == "aug1_4":
        test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            #transforms.RandomResizedCrop(args.img_dim, scale=(0.2, 1.)),
            #transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomRotation(degrees=(135, 135)),
            #transforms.RandomApply([
            #    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            #], p=0.8),
            #transforms.RandomGrayscale(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif args.test_aug_type == "aug1_5":
        test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            #transforms.RandomResizedCrop(args.img_dim, scale=(0.2, 1.)),
            #transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomRotation(degrees=(180, 180)),
            #transforms.RandomApply([
            #    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            #], p=0.8),
            #transforms.RandomGrayscale(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    print("args.test_aug_type", args.test_aug_type)

    valdir = os.path.join(args.data, 'val')
    if args.dataset_type == "Imagenet":
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, test_transforms),
            batch_size=64, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    elif args.dataset_type == "IN100":
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, test_transforms),
            batch_size=16, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    #print(torch.load(args.pretrained)['state_dict'].keys())

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder up to before the embedding layer
                if k.startswith('module.'): #and not k.startswith('module.encoder.fc'):
                    # remove prefix
                    state_dict[k[len("module."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            model.load_state_dict(state_dict)
            print("model has been loaded:", args.pretrained)

    model.eval()

    for i, (images, target) in enumerate(val_loader):
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        if args.only_aug_vis == "only_aug_vis":
            pass
        else:
            #print(images.shape)
            target_layers = [model.layer4[-1]]
            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
            grayscale_cam = cam(input_tensor=images, targets=None)
            grayscale_cam = grayscale_cam[0]

        rgb_img = images[0]
        rgb_img = torch.permute(rgb_img, (1, 2, 0))
        rgb_img_np = rgb_img.detach().cpu().numpy() 
        rgb_img_np = rgb_img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])

        if args.only_aug_vis == "only_aug_vis":
            if args.vis_output_dir != None:
                save_image_name = str(i)+".jpg"
                imageio.imwrite(args.vis_output_dir + '/' + save_image_name, rgb_img_np)
        else:
            try:
                visualization0 = show_cam_on_image(rgb_img_np, grayscale_cam, colormap = cv2.COLORMAP_JET, use_rgb=True) ## cv2.COLORMAP_JET
                visualization0 = (visualization0).astype(np.uint8)
                if args.vis_output_dir != None:
                    save_image_name = str(i)+".jpg"
                    imageio.imwrite(args.vis_output_dir + '/' + save_image_name, visualization0)
            except:
                pass
    

        i+=1
        if i>800:
            break

if __name__ == '__main__':
    main()


































