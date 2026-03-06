"""
Training script for GeoSlot.

Usage:
    python train.py --dataset university1652 --backbone vim_tiny --epochs 100
    python train.py --dataset vigor --backbone vim_small --epochs 80
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.GeoSlot_geo import GeoSlot
from src.losses.joint_loss import JointLoss
from src.datasets.university1652 import University1652Dataset
from src.datasets.vigor import VIGORDataset
from src.datasets.cvusa import CVUSADataset
from src.datasets.cv_cities import CVCitiesDataset
from src.utils.metrics import compute_recall_at_k, extract_embeddings
from src.configs.default import ModelConfig, LossConfig, TrainConfig, KAGGLE_PATHS


def parse_args():
    parser = argparse.ArgumentParser(description='GeoSlot Training')
    parser.add_argument('--dataset', type=str, default='university1652',
                        choices=['university1652', 'vigor', 'cvusa', 'cv_cities'])
    parser.add_argument('--backbone', type=str, default='vim_tiny',
                        choices=['vim_tiny', 'vim_small'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--kaggle', action='store_true', help='Use Kaggle paths')
    return parser.parse_args()


def get_dataset(args, split='train'):
    """Create dataset based on args."""
    if args.kaggle:
        paths = KAGGLE_PATHS
    else:
        paths = {
            'university1652': './data/University-1652',
            'cvusa': './data/CVUSA',
            'cv_cities': './data/cv-cities',
            'vigor_chicago': './data/VIGOR/Chicago',
            'vigor_newyork': './data/VIGOR/NewYork',
            'vigor_sanfrancisco': './data/VIGOR/SanFrancisco',
            'vigor_seattle': './data/VIGOR/Seattle',
        }

    if args.dataset == 'university1652':
        return University1652Dataset(
            root=paths['university1652'],
            split=split,
            img_size=args.img_size,
        )
    elif args.dataset == 'vigor':
        vigor_dirs = [paths[f'vigor_{c.lower()}'] for c in ['chicago', 'newyork', 'sanfrancisco', 'seattle']]
        return VIGORDataset(
            root_dirs=vigor_dirs,
            split=split,
            img_size=args.img_size,
        )
    elif args.dataset == 'cvusa':
        return CVUSADataset(
            root=paths['cvusa'],
            split=split,
            img_size=args.img_size,
        )
    elif args.dataset == 'cv_cities':
        return CVCitiesDataset(
            root=paths['cv_cities'],
            split=split,
            img_size=args.img_size,
        )


def train_one_epoch(model, criterion, dataloader, optimizer, scaler, device, epoch, global_step):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        query = batch['query'].to(device)
        gallery = batch['gallery'].to(device)

        optimizer.zero_grad()

        with autocast(enabled=scaler is not None):
            model_output = model(query, gallery, global_step=global_step)
            loss_dict = criterion(model_output, epoch=epoch)
            loss = loss_dict['total_loss']

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        total_acc += loss_dict['accuracy'].item()
        num_batches += 1
        global_step += 1

        if batch_idx % 50 == 0:
            logging.info(
                f"  Batch [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} "
                f"Acc: {loss_dict['accuracy'].item():.2%} "
                f"Stage: {loss_dict['active_stage']}"
            )

    avg_loss = total_loss / max(num_batches, 1)
    avg_acc = total_acc / max(num_batches, 1)
    return avg_loss, avg_acc, global_step


def main():
    args = parse_args()

    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'{args.dataset}_{args.backbone}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Args: {args}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")

    # Model
    model_config = ModelConfig(backbone=args.backbone, img_size=args.img_size)
    model = GeoSlot(
        backbone=model_config.backbone,
        img_size=model_config.img_size,
        slot_dim=model_config.slot_dim,
        max_slots=model_config.max_slots,
        n_register=model_config.n_register,
        embed_dim_out=model_config.embed_dim_out,
    ).to(device)
    logging.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Loss
    loss_config = LossConfig()
    criterion = JointLoss(
        lambda_infonce=loss_config.lambda_infonce,
        lambda_dwbl=loss_config.lambda_dwbl,
        lambda_csm=loss_config.lambda_csm,
        lambda_dice=loss_config.lambda_dice,
        temperature=loss_config.temperature,
        stage2_epoch=loss_config.stage2_epoch,
        stage3_epoch=loss_config.stage3_epoch,
    ).to(device)

    # Dataset
    train_dataset = get_dataset(args, split='train')
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    logging.info(f"Train samples: {len(train_dataset)}")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # AMP Scaler
    scaler = GradScaler() if torch.cuda.is_available() else None

    # Resume
    start_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint.get('global_step', 0)
        logging.info(f"Resumed from epoch {start_epoch}")

    # Training loop
    logging.info("=" * 60)
    logging.info("Starting training...")
    logging.info("=" * 60)

    best_loss = float('inf')
    for epoch in range(start_epoch, args.epochs):
        logging.info(f"\nEpoch [{epoch+1}/{args.epochs}] lr={optimizer.param_groups[0]['lr']:.2e}")

        avg_loss, avg_acc, global_step = train_one_epoch(
            model, criterion, train_loader, optimizer, scaler,
            device, epoch, global_step
        )
        scheduler.step()

        logging.info(f"Epoch [{epoch+1}] Loss: {avg_loss:.4f} Acc: {avg_acc:.2%}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0 or avg_loss < best_loss:
            checkpoint = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            path = os.path.join(output_dir, f'checkpoint_epoch{epoch+1}.pth')
            torch.save(checkpoint, path)
            logging.info(f"Saved checkpoint: {path}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(output_dir, 'best_model.pth')
                torch.save(checkpoint, best_path)
                logging.info(f"New best model saved: {best_path}")

    logging.info("Training complete!")


if __name__ == '__main__':
    main()
