
#----------------------------------------------------
#  Pytorch
#----------------------------------------------------
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.utils.data.distributed
import torch.distributed as dist

#----------------------------------------------------
#  Load CNN-architecture
#----------------------------------------------------
from models.network import get_network

#--------------
#  Datalodader
#--------------
from loader import custom_dataloader

#----------------------------------------------------
#  Etc
#----------------------------------------------------
import os, logging
import argparse
import numpy as np

#--------------
# Util
#--------------
from utils.dir_maker import DirectroyMaker
from utils.AverageMeter import AverageMeter
from utils.metric import metric_ece_aurc_eaurc
from utils.color import Colorer
from utils.etc import progress_bar, is_main_process, save_on_master, paser_config_save, set_logging_defaults
from utils.metric import SCELoss, ECELoss


def parse_args():
    parser = argparse.ArgumentParser(description='Progressive Self-Knowledge Distillation : PS-KD')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_rate', default=0.1, type=float, help='learning rate decay rate')
    parser.add_argument('--lr_decay_schedule', default=[150, 225], nargs='*', type=int, help='when to drop lr')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight_decay')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number')
    parser.add_argument('--end_epoch', default=300, type=int, help='number of training epoch to run')
    parser.add_argument('--PSKD', action='store_true', help='PSKD')
    parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size (default: 128), this is the total'
                                                                    'batch size of all GPUs on the current node when '
                                                                    'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--experiments_dir', type=str, default='models',help='Directory name to save the model, log, config')
    parser.add_argument('--classifier_type', type=str, default='ResNet18', help='Select classifier')
    parser.add_argument('--data_path', type=str, default=None, help='download dataset path')
    parser.add_argument('--data_type', type=str, default=None, help='type of dataset')
    parser.add_argument('--alpha_T',default=0.8 ,type=float, help='alpha_T')
    parser.add_argument('--saveckp_freq', default=299, type=int, help='Save checkpoint every x epochs. Last model saving set to 299')
    parser.add_argument('--rank', default=-1, type=int,help='node rank for distributed training')
    parser.add_argument('--world_size', default=1, type=int,help='number of distributed processes')
    parser.add_argument('--dist_backend', default='nccl', type=str,help='distributed backend')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:8080', type=str,help='url used to set up distributed training')
    parser.add_argument('--workers', default=40, type=int, help='number of workers for dataloader')
    parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
    parser.add_argument('--resume', type=str, default=None, help='load model path')
    parser.add_argument('--model_path', help='path to PyTorch model file')
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
    parser.add_argument('--epoch', type=int, default=160, help='for which epoch this script is run')

    args = parser.parse_args()
    return args

# Define a function to load the model
def load_model(model_path):
    model = torch.load(model_path)
    return model

#----------------------------------------------------
#  Top-1 / Top -5 accuracy
#----------------------------------------------------
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# Define a function to calculate the SCE & ECE loss for a given input
def calculate_sce_ece_loss(model, data_loader, args):
    # epoch: Just a dummy variable 
    val_top1 = AverageMeter()
    val_top5 = AverageMeter()
    val_losses = AverageMeter()


    criterion_CE = nn.CrossEntropyLoss().cuda(args.gpu)
    targets_list = []
    confidences = []

    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(data_loader):
            if args.gpu is not None:
                inputs = inputs.cuda(args.gpu, non_blocking=True)
                targets = targets.cuda(args.gpu, non_blocking=True)
                
            #for ECE, AURC, EAURC
            targets_numpy = targets.cpu().numpy()
            targets_list.extend(targets_numpy.tolist())
                
            # model output
            outputs = model(inputs)
            
            # for ECE, AURC, EAURC
            softmax_predictions = F.softmax(outputs, dim=1)
            softmax_predictions = softmax_predictions.cpu().numpy()
            for values_ in softmax_predictions:
                confidences.append(values_.tolist())
                
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            loss = criterion_CE(outputs, targets)
            val_losses.update(loss.item(), inputs.size(0))
            
            #Top1, Top5 Err
            err1, err5 = accuracy(outputs.data, targets, topk=(1, 5))
            val_top1.update(err1.item(), inputs.size(0))
            val_top5.update(err5.item(), inputs.size(0))

            progress_bar(args.epoch, batch_idx, len(data_loader), args,'val_loss: {:.3f} | val_top1_acc: {:.3f} | val_top5_acc: {:.3f} | correct/total({}/{})'.format(
                        val_losses.avg,
                        val_top1.avg,
                        val_top5.avg,
                        correct,
                        total))
    
    # Calculate authentic ECE
    ece_auth = metric_ece_aurc_eaurc(confidences, targets_list, bin_size=0.1)[0]

    # Calculate the ECE and SCE loss
    confidences = np.asarray(confidences)
    targets_list = np.asarray(targets_list)
    SCE = SCELoss().loss(confidences, targets_list, n_bins=15, logits=False)
    ECE = ECELoss().loss(confidences, targets_list, n_bins=15, logits=False)

    return SCE, ECE, ece_auth

# Define the main function that takes a model path as an argument and calculates the SCE loss
def main(args):
    # Define a model in Pytorch
    model = get_network(args)

    # Load the model
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(args.model_path)['net'])
    model.to(device)   
    model.eval()

    #---------------------------------------------------
    #  Load Dataset
    #---------------------------------------------------
    train_loader, valid_loader, test_loader, train_sampler = custom_dataloader.dataloader(args)
    
    # Calculate the SCE and ECE Loss
    sce_loss, ece_loss, ece_auth = calculate_sce_ece_loss(model, test_loader, args)

    print('SCE loss:', sce_loss)
    print('ECE loss:', ece_loss)
    print('ECE Auth loss:', ece_auth)

if __name__ == '__main__':
    import argparse
    args = parse_args()

    main(args)

