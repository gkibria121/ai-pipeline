
# ============================================================================
# FILE: scripts/train.py
# ============================================================================

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models import AASIST
from data_utils import Dataset_ASVspoof2019_train, genSpoof_list
from utils import create_optimizer, set_seed, seed_worker


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        _, outputs = model(inputs, freq_aug=True)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f'Batch [{batch_idx + 1}/{len(dataloader)}] '
                  f'Loss: {running_loss / (batch_idx + 1):.4f} '
                  f'Acc: {100. * correct / total:.2f}%')
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description='Train AASIST model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--seed', type=int, default=1234,
                       help='Random seed')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Set seed
    set_seed(args.seed, config)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Prepare data
    database_path = Path(config['database_path'])
    d_label_trn, file_train = genSpoof_list(
        database_path / 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt',
        is_train=True
    )
    
    print(f'Number of training samples: {len(file_train)}')
    
    # Create dataset and dataloader
    train_set = Dataset_ASVspoof2019_train(
        file_train, d_label_trn,
        database_path / 'ASVspoof2019_LA_train',
        cut=config['model_config']['nb_samp']
    )
    
    train_loader = DataLoader(
        train_set,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        worker_init_fn=seed_worker
    )
    
    # Create model
    model = AASIST(config['model_config']).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    config['optim_config']['epochs'] = config['num_epochs']
    config['optim_config']['steps_per_epoch'] = len(train_loader)
    
    optimizer, scheduler = create_optimizer(
        model.parameters(), config['optim_config']
    )
    
    # Training loop
    for epoch in range(config['num_epochs']):
        print(f'\nEpoch {epoch + 1}/{config["num_epochs"]}')
        print('-' * 60)
        
        loss, acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
                for _ in range(len(train_loader)):
                    scheduler.step()
            else:
                scheduler.step()
        
        print(f'Training Loss: {loss:.4f}, Accuracy: {acc:.2f}%')
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = Path(config['model_path']).parent / \
                            f'checkpoint_epoch_{epoch + 1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')
    
    # Save final model
    torch.save(model.state_dict(), config['model_path'])
    print(f'\nTraining complete! Model saved to {config["model_path"]}')


if __name__ == '__main__':
    main()