#!/usr/bin/env python3
"""
PDFNet Training Script

Main training script for PDFNet model.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from .dataloaders import Mydataset as Mydataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime
from torch.autograd import Variable
from . import utiles
import os
from timm.scheduler import create_scheduler
import gc
from torch.utils.tensorboard import SummaryWriter
from .models.PDFNet import build_model
import random
from torch.cuda.amp import autocast, GradScaler
import shutil
import sys
from .common_utils import get_files


def copy_allfiles(src, dest, not_case=["valid_sample", "runs"]):
    """Copy all files from src to dest, excluding specified directories."""
    for root, dirs, files in os.walk(src):
        # Calculate corresponding path in target folder
        relative_path = os.path.relpath(root, src)
        flag = 0
        for not_case_ in not_case:
            if not_case_ in relative_path:
                flag = 1
                break
        if flag == 1:
            continue
        target_subfolder = os.path.join(dest, relative_path)
        # Create corresponding subfolders
        os.makedirs(target_subfolder, exist_ok=True)
        # Copy files
        for file_name in files:
            source_file = os.path.join(root, file_name)
            target_file = os.path.join(target_subfolder, file_name)
            shutil.copy(source_file, target_file)


def setup_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


def get_args_parser():
    """Get argument parser for training."""
    parser = argparse.ArgumentParser('PDFNet_swinB training script', add_help=False)
    parser.add_argument('--COPY', default=True, type=bool)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='Number of steps to accumulate gradients when updating parameters, set to 1 to disable this feature')
    parser.add_argument('--update_half', default=False, type=bool,
                        help='update_half')
    parser.add_argument('--model', default='PDFNet_swinB', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=1024, type=int, help='images input size')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--model_ema', action='store_true')
    parser.add_argument('--no_model_ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model_ema_decay', type=float, default=0.99996, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--lr_noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr_noise_pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr_noise_std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay_epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown_epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience_epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay_rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated_aug', action='store_true')
    parser.add_argument('--no_repeated_aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--finetune_epoch', default=0, type=int, help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data_path', default='/DATA/DIS-DATA', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=1, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--data_set', default='DIS', choices=['DIS', 'HRSOD', 'UHRSD'], type=str, help='Image Net dataset path')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval_metric', default='F1', type=str, help='eval metric')
    parser.add_argument('--dist_eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--checkpoints_save_path', default='checkpoints', type=str, help='checkpoints_save_path')
    parser.add_argument('--DEBUG', default=False, type=bool, help='DEBUG')

    return parser


def train_main(args):
    """Main training function."""
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    seed = args.seed
    setup_seed(seed)

    dataset_train = Mydataset.build_dataset(is_train=True, args=args)
    dataset_val = Mydataset.build_dataset(is_train=False, args=args)

    data_loader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, 
                                   num_workers=args.num_workers, persistent_workers=True)
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=args.batch_size, shuffle=True, 
                                 num_workers=args.num_workers, persistent_workers=True)

    print(f"Creating model: {args.model}")
    model, model_name = build_model(args)

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        if 'model' in checkpoint:
            checkpoint_model = checkpoint['model']
        else:
            checkpoint_model = checkpoint
        model_dict = model.state_dict()
        checkpoint_model = {k: v for k, v in checkpoint_model.items() 
                           if k in model_dict and v.shape == model_dict[k].shape}
        model.load_state_dict(checkpoint_model, strict=False)
    
    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', round(n_parameters/1024/1024), "M")
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % (args.batch_size * args.update_freq))
    print("Update frequent = %d" % args.update_freq)

    optimizer = utiles.build_optimizer(args, model)
    lr_scheduler, EPOCH = create_scheduler(args, optimizer)

    train_time = str(datetime.datetime.today()).replace(' ', '_').replace(':', '_')[:-7]
    if not args.DEBUG:
        writer = SummaryWriter(log_dir='runs/'+args.model+train_time)

    this_checkpoints_dir = None
    train_time = str(datetime.datetime.today()).replace(' ', '_').replace(':', '_')[:-7]
    if this_checkpoints_dir is None:
        this_checkpoints_dir = f'{args.checkpoints_save_path}/{model_name}{train_time}'
    if not args.DEBUG:
        os.makedirs(this_checkpoints_dir, exist_ok=True)

    best_valid_f1 = -1
    best_valid_mae = -1
    iter_pbar = tqdm(total=len(data_loader_train))
    mean_epoch_time = 0
    rest_time = 0

    tmp_f1, tmp_mae = 0, 0

    scaler = GradScaler()
    if not args.DEBUG:
        os.makedirs(f'valid_sample/{args.model}{train_time}', exist_ok=True)
        if args.COPY:
            os.makedirs(f'valid_sample/{args.model}{train_time}/project_copy', exist_ok=True)
            copy_allfiles(os.getcwd(), f'valid_sample/{args.model}{train_time}/project_copy')
    
    large_loss_name_list = []
    
    for epoch in range(EPOCH):
        if epoch < args.finetune_epoch and args.finetune_epoch > 0:
            lr_scheduler.step(epoch)
            continue

        loss_list = []
        model.train()
        epoch_loss, epoch_R_loss = 0, 0
        epoch_target_loss = 0
        epoch_starttime = datetime.datetime.now()
        iters = 0
        
        for i, data in enumerate(data_loader_train):
            if epoch % 2 == 1 and args.update_half:
                if data['image_name'] not in large_loss_name_list:
                    iter_pbar.update()
                    continue
                else:
                    large_loss_name_list.remove(data['image_name'])
            
            inputs, gt, labels = data['image'], data['gt'], data['label']
            depth, depth_large = data['depth'], data['depth_large']
            
            if args.device != 'cpu':
                inputs_v = Variable(inputs.to(device), requires_grad=False)
                gt_v = Variable(gt.to(device), requires_grad=False)
                labels_v = Variable(labels.to(device), requires_grad=False)
                depth_v = Variable(depth.to(device), requires_grad=False)
                depth_large_v = Variable(depth_large.to(device), requires_grad=False)
            else:
                inputs_v = Variable(inputs, requires_grad=False)
                gt_v = Variable(gt, requires_grad=False)
                labels_v = Variable(labels, requires_grad=False)
                depth_v = Variable(depth, requires_grad=False)
                depth_large_v = Variable(depth_large, requires_grad=False)
            
            with autocast():
                outputs = model(inputs_v, depth_v, gt_v, depth_large_v)
                pred, loss, target_loss = outputs[0], outputs[1], outputs[2]
                loss = loss / args.update_freq
            
            scaler.scale(loss).backward()
            iters += 1
            
            if iters % args.update_freq == 0 or iters == len(data_loader_train):    
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            with torch.no_grad():
                loss_list.append({"image_name": data['image_name'], 'loss': float(target_loss.cpu().detach())})
                epoch_loss += float(loss.cpu().detach().item() * args.update_freq)
                epoch_target_loss += float(target_loss.cpu().detach())
                iter_pbar.update()
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                
                if args.eval_metric == 'F1':
                    iter_pbar.set_description(f'Epoch: {epoch+1}/{EPOCH}, mean epoch time: {mean_epoch_time}s, rest time: {rest_time/3600:.2f}h, '
                                            + f'Train loss: {epoch_loss/(i+1):.4f}, '
                                            + f'target loss: {epoch_target_loss/(i+1):.4f}, '
                                            + '##'
                                            + f'Valid F1: {tmp_f1:.4f}, '
                                            + f'mae: {tmp_mae:.4f},'
                                            + '##'
                                            + f'best F1: {best_valid_f1:.4f}, '
                                            + f'LR:{lr:.8f}'
                                        )
                elif args.eval_metric == 'MAE':
                    iter_pbar.set_description(f'Epoch: {epoch+1}/{EPOCH}, mean epoch time: {mean_epoch_time}s, rest time: {rest_time/3600:.2f}h, '
                                            + f'Train loss: {epoch_loss/(i+1):.4f}, '
                                            + f'target loss: {epoch_target_loss/(i+1):.4f}, '
                                            + '##'
                                            + f'Valid F1: {tmp_f1:.4f}, '
                                            + f'mae: {tmp_mae:.4f},'
                                            + '##'
                                            + f'best MAE: {best_valid_mae:.4f}, '
                                            + f'LR:{lr:.8f}'
                                        )
                del outputs, loss, inputs_v, gt_v, labels_v, pred, target_loss
                gc.collect()
        
        torch.cuda.empty_cache()
        
        if args.eval:
            tmp_f1, tmp_mae, best_valid, inputs_k, gt_k, pred_grad_k = utiles.eval(
                this_checkpoints_dir, model, epoch, dataset_val, data_loader_val, 
                best_valid_f1, train_time, args)
            
            if args.eval_metric == 'F1':
                best_valid_f1 = best_valid
            elif args.eval_metric == 'MAE':
                best_valid_mae = best_valid
                
            if not args.DEBUG:
                writer.add_scalar('Train/loss', epoch_loss/(iters), epoch+1)
                writer.add_scalar('Train/target_loss', epoch_target_loss/(iters), epoch+1)
                writer.add_scalar('Valid/F1', tmp_f1, epoch+1)
                writer.add_scalar('Valid/mae', tmp_mae, epoch+1)
                writer.add_scalar('Lr', lr, epoch+1)
                writer.add_image('Image/image', np.array(inputs_k.cpu().detach().permute(1,2,0)), 
                               dataformats='HWC', global_step=epoch+1)
                writer.add_image('Image/GT', np.array(gt_k.cpu().detach().permute(1,2,0)), 
                               dataformats='HWC', global_step=epoch+1)
                writer.add_image('Image/Pred', np.array(pred_grad_k.cpu().detach().permute(1,2,0)), 
                               dataformats='HWC', global_step=epoch+1)
        else:
            if not args.DEBUG:
                writer.add_scalar('Train/loss', epoch_loss/(iters), epoch+1)
                writer.add_scalar('Train/target_loss', epoch_target_loss/(iters), epoch+1)
        
        epoch_endtime = datetime.datetime.now()
        mean_epoch_time = (epoch_endtime-epoch_starttime).seconds
        rest_time = (EPOCH-epoch-1)*mean_epoch_time
        iter_pbar.reset()
        lr_scheduler.step(epoch)
        torch.cuda.empty_cache()
        torch.save(model.state_dict(), f'{this_checkpoints_dir}/LAST.pth')
        large_loss_list = sorted(loss_list, key=lambda i: i['loss'], reverse=True)
        large_loss_name_list = [item['image_name'] for item in large_loss_list][:loss_list.__len__()//2]
    
    if not args.DEBUG:
        torch.save(model.state_dict(), 
                  f'{this_checkpoints_dir}/{epoch}_last_F1_{tmp_f1:.6f}_mae_{tmp_mae:.6f}_{args.model}.pth')
    iter_pbar.close()


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser('PDFNet training', parents=[get_args_parser()])
    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    train_main(args)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())