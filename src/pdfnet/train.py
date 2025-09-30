#!/usr/bin/env python3
"""
PDFNet Training Script

Main training script for PDFNet model using type-safe configuration.
"""

import torch
import numpy as np
from dataclasses import dataclass
from .dataloaders import dis_dataset
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
from .config import PDFNetConfig
import logging

logger = logging.getLogger(__name__)


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


def _config_to_args(config: PDFNetConfig):
    """
    Convert PDFNetConfig to a namespace object for compatibility with legacy training code.
    This creates a dataclass that mimics the old argument structure.
    """
    @dataclass
    class TrainingArgs:
        """Training arguments derived from config."""
        # Core parameters
        model: str
        batch_size: int
        epochs: int
        input_size: int
        device: str
        seed: int
        
        # Dataset
        data_path: str
        eval_data_path: str | None
        data_set: str
        nb_classes: int
        imagenet_default_mean_and_std: bool
        
        # Model architecture
        emb: int
        back_bone_channels_stage1: int
        back_bone_channels_stage2: int
        back_bone_channels_stage3: int
        back_bone_channels_stage4: int
        drop: float
        drop_path: float
        model_ema: bool
        model_ema_decay: float
        model_ema_force_cpu: bool
        
        # Optimizer
        opt: str
        opt_eps: float
        opt_betas: list[float] | None
        lr: float
        weight_decay: float
        momentum: float
        clip_grad: float | None
        
        # Learning rate schedule
        sched: str
        warmup_lr: float
        min_lr: float
        warmup_epochs: int
        decay_epochs: float
        cooldown_epochs: int
        patience_epochs: int
        decay_rate: float
        lr_noise: list[float] | None
        lr_noise_pct: float
        lr_noise_std: float
        
        # Augmentation
        color_jitter: float
        aa: str
        smoothing: float
        train_interpolation: str
        repeated_aug: bool
        reprob: float
        remode: str
        recount: int
        resplit: bool
        mixup: float
        cutmix: float
        cutmix_minmax: list[float] | None
        mixup_prob: float
        mixup_switch_prob: float
        mixup_mode: str
        
        # Training control
        update_freq: int
        update_half: bool
        COPY: bool
        
        # Checkpointing
        finetune: str
        finetune_epoch: int
        resume: str
        auto_resume: bool
        save_ckpt: bool
        start_epoch: int
        checkpoints_save_path: str
        
        # Output and evaluation
        output_dir: str
        log_dir: str | None
        eval_metric: str
        eval: bool
        dist_eval: bool
        
        # System
        num_workers: int
        pin_mem: bool
        world_size: int
        local_rank: int
        dist_on_itp: bool
        dist_url: str
        DEBUG: bool
    
    # Create args object from config
    args = TrainingArgs(
        # Core parameters
        model=config.model.name,
        batch_size=config.training.batch_size,
        epochs=config.training.epochs,
        input_size=config.data.input_size,
        device=config.device,
        seed=config.training.seed,
        
        # Dataset
        data_path=str(config.data.root_path),
        eval_data_path=None,
        data_set=config.data.dataset,
        nb_classes=1,
        imagenet_default_mean_and_std=True,
        
        # Model architecture
        emb=128,
        back_bone_channels_stage1=128,
        back_bone_channels_stage2=256,
        back_bone_channels_stage3=512,
        back_bone_channels_stage4=1024,
        drop=0.0,
        drop_path=config.model.drop_path,
        model_ema=True,
        model_ema_decay=0.99996,
        model_ema_force_cpu=False,
        
        # Optimizer
        opt=config.training.optimizer.type,
        opt_eps=config.training.optimizer.eps,
        opt_betas=list(config.training.optimizer.betas) if config.training.optimizer.betas else None,
        lr=config.training.optimizer.lr,
        weight_decay=config.training.optimizer.weight_decay,
        momentum=config.training.optimizer.momentum,
        clip_grad=config.training.gradient.clip_grad,
        
        # Learning rate schedule
        sched=config.training.scheduler.type,
        warmup_lr=config.training.scheduler.warmup_lr,
        min_lr=config.training.scheduler.min_lr,
        warmup_epochs=config.training.scheduler.warmup_epochs,
        decay_epochs=30.0,
        cooldown_epochs=10,
        patience_epochs=10,
        decay_rate=0.1,
        lr_noise=None,
        lr_noise_pct=0.67,
        lr_noise_std=1.0,
        
        # Augmentation
        color_jitter=config.training.augmentation.color_jitter,
        aa='rand-m9-mstd0.5-inc1',
        smoothing=0.1,
        train_interpolation='bicubic',
        repeated_aug=True,
        reprob=0.25,
        remode='pixel',
        recount=1,
        resplit=False,
        mixup=config.training.augmentation.mixup,
        cutmix=config.training.augmentation.cutmix,
        cutmix_minmax=None,
        mixup_prob=1.0,
        mixup_switch_prob=0.5,
        mixup_mode='batch',
        
        # Training control
        update_freq=config.training.gradient.update_freq,
        update_half=False,
        COPY=True,
        
        # Checkpointing
        finetune='',
        finetune_epoch=0,
        resume='',
        auto_resume=True,
        save_ckpt=True,
        start_epoch=0,
        checkpoints_save_path=str(config.output.checkpoint_dir),
        
        # Output and evaluation
        output_dir=str(config.output.save_dir),
        log_dir=str(config.output.log_dir) if config.output.log_dir else None,
        eval_metric=config.training.eval_metric,
        eval=False,
        dist_eval=False,
        
        # System
        num_workers=config.training.num_workers,
        pin_mem=True,
        world_size=1,
        local_rank=-1,
        dist_on_itp=False,
        dist_url='env://',
        DEBUG=config.debug
    )
    
    return args




def train_from_config(config: PDFNetConfig, resume: str = '', finetune: str = ''):
    """
    Main training function using type-safe configuration.
    
    :param config: PDFNet configuration object
    :param resume: Path to checkpoint to resume from
    :param finetune: Path to checkpoint to finetune from
    """
    # Convert config to args for compatibility
    args = _config_to_args(config)
    
    # Override resume/finetune if provided
    if resume:
        args.resume = resume
    if finetune:
        args.finetune = finetune
    
    # Call legacy training function
    return train_main(args)


def train_main(args):
    """
    Legacy training function using args namespace.
    
    DEPRECATED: Use train_from_config() with PDFNetConfig instead.
    This function is kept for backward compatibility.
    """
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    seed = args.seed
    setup_seed(seed)

    dataset_train = dis_dataset.build_dataset(is_train=True, args=args)
    dataset_val = dis_dataset.build_dataset(is_train=False, args=args)

    data_loader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, 
                                   num_workers=args.num_workers, persistent_workers=True)
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=args.batch_size, shuffle=True, 
                                 num_workers=args.num_workers, persistent_workers=True)

    logger.info(f"Creating model: {args.model}")
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
    logger.info(f'Model params: {round(n_parameters/1024/1024)}M')
    logger.info(f"LR: {args.lr:.8f}")
    logger.info(f"Batch size: {args.batch_size * args.update_freq}")
    logger.info(f"Update freq: {args.update_freq}")

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
        epoch_loss, _epoch_R_loss = 0, 0
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


if __name__ == '__main__':
    logger.info("=" * 70)
    logger.info("PDFNet Training")
    logger.info("=" * 70)
    logger.info("This module should be called through the tyro CLI:")
    logger.info("  uv run pdfnet.py train --config-file config/default.yaml")
    logger.info("  python -m pdfnet train --config-file config/default.yaml")
    logger.info("=" * 70)
    sys.exit(1)