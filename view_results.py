#!/usr/bin/env python3
"""
Script to view and fix metrics from training results.
"""
import pandas as pd
from pathlib import Path
import sys

def view_results(exp_dir):
    """View results from experiment directory."""
    exp_dir = Path(exp_dir)
    metrics_file = exp_dir / "metrics" / "metrics.csv"
    
    if not metrics_file.exists():
        print(f"Error: Metrics file not found at {metrics_file}")
        return
    
    # Read the CSV file
    df = pd.read_csv(metrics_file)
    
    # Check if columns need to be renamed (old format with capital letters)
    if 'Dev_EER' in df.columns or 'Epoch' in df.columns:
        print("Converting old column names to new format...")
        # Create a mapping for column renaming
        rename_map = {
            'Epoch': 'epoch',
            'Train_Loss': 'train_loss',
            'Dev_EER': 'dev_eer',
            'Dev_tDCF': 'dev_tdcf',
            'Dev_Acc': 'dev_acc',
            'Eval_EER': 'eval_eer',
            'Eval_tDCF': 'eval_tdcf',
            'Eval_Acc': 'eval_acc',
            'Best_Dev_EER': 'best_dev_eer',
            'Best_Dev_tDCF': 'best_dev_tdcf'
        }
        df = df.rename(columns=rename_map)
        
        # Save the fixed CSV
        df.to_csv(metrics_file, index=False)
        print(f"✓ Fixed column names in {metrics_file}")
    
    # Display results
    print(f"\n{'='*70}")
    print(f"RESULTS: {exp_dir.name}")
    print(f"{'='*70}")
    print(f"Best Dev EER: {df['dev_eer'].min():.3f}%")
    print(f"Best Dev Accuracy: {df['dev_acc'].max():.2f}%")
    
    # Check if eval metrics are available
    if not df['eval_eer'].isna().all():
        last_eval_idx = df['eval_eer'].last_valid_index()
        if last_eval_idx is not None:
            print(f"Final Eval EER: {df['eval_eer'].iloc[last_eval_idx]:.3f}%")
            print(f"Final Eval Accuracy: {df['eval_acc'].iloc[last_eval_idx]:.2f}%")
    else:
        print("No evaluation metrics available yet")
    
    print(f"{'='*70}\n")
    
    # Display per-epoch breakdown
    print("Per-Epoch Metrics:")
    print("-" * 70)
    for idx, row in df.iterrows():
        epoch = int(row['epoch'])
        train_loss = row['train_loss']
        dev_eer = row['dev_eer']
        dev_acc = row['dev_acc']
        
        print(f"Epoch {epoch}: Loss={train_loss:.5f}, Dev_EER={dev_eer:.3f}%, Dev_Acc={dev_acc:.2f}%", end="")
        
        if not pd.isna(row['eval_eer']):
            eval_eer = row['eval_eer']
            eval_acc = row['eval_acc']
            print(f", Eval_EER={eval_eer:.3f}%, Eval_Acc={eval_acc:.2f}%")
        else:
            print()
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exp_dir = sys.argv[1]
    else:
        exp_dir = "./exp_result/FakeorReal_audio_SEResNet_ep1_bs32_feat1"
    
    view_results(exp_dir)
