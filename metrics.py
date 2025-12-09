"""
Metrics tracking, saving, and visualization module for ASVspoof detection.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')


class MetricsTracker:
    """Track and save training/validation metrics."""
    
    def __init__(self, save_dir: Path):
        """
        Initialize metrics tracker.
        
        Args:
            save_dir: Directory to save metrics files
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            'epochs': [],
            'train_loss': [],
            'dev_eer': [],
            'dev_tdcf': [],
            'eval_eer': [],
            'eval_tdcf': [],
            'best_dev_eer': [],
            'best_dev_tdcf': [],
        }
        
        self.csv_file = self.save_dir / "metrics.csv"
        self.json_file = self.save_dir / "metrics.json"
        
        # Create CSV file with headers
        self._init_csv()
    
    def _init_csv(self):
        """Initialize CSV file with headers."""
        if not self.csv_file.exists():
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'Train_Loss', 'Dev_EER', 'Dev_tDCF', 
                                'Eval_EER', 'Eval_tDCF', 'Best_Dev_EER', 'Best_Dev_tDCF'])
    
    def add_epoch(self, epoch: int, train_loss: float, dev_eer: float, 
                  dev_tdcf: float, eval_eer: Optional[float] = None, 
                  eval_tdcf: Optional[float] = None, best_dev_eer: float = None,
                  best_dev_tdcf: float = None):
        """
        Add metrics for an epoch.
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            dev_eer: Development EER
            dev_tdcf: Development t-DCF
            eval_eer: Evaluation EER (optional)
            eval_tdcf: Evaluation t-DCF (optional)
            best_dev_eer: Best development EER so far
            best_dev_tdcf: Best development t-DCF so far
        """
        self.metrics['epochs'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['dev_eer'].append(dev_eer)
        self.metrics['dev_tdcf'].append(dev_tdcf)
        self.metrics['eval_eer'].append(eval_eer if eval_eer is not None else np.nan)
        self.metrics['eval_tdcf'].append(eval_tdcf if eval_tdcf is not None else np.nan)
        self.metrics['best_dev_eer'].append(best_dev_eer if best_dev_eer is not None else np.nan)
        self.metrics['best_dev_tdcf'].append(best_dev_tdcf if best_dev_tdcf is not None else np.nan)
        
        # Save to CSV
        self._save_csv_row(epoch, train_loss, dev_eer, dev_tdcf, eval_eer, eval_tdcf, best_dev_eer, best_dev_tdcf)
        
        # Save to JSON
        self.save_json()
    
    def _save_csv_row(self, epoch, train_loss, dev_eer, dev_tdcf, eval_eer, eval_tdcf, best_dev_eer, best_dev_tdcf):
        """Save a single row to CSV."""
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.5f}",
                f"{dev_eer:.5f}",
                f"{dev_tdcf:.5f}",
                f"{eval_eer:.5f}" if eval_eer is not None and not np.isnan(eval_eer) else "",
                f"{eval_tdcf:.5f}" if eval_tdcf is not None and not np.isnan(eval_tdcf) else "",
                f"{best_dev_eer:.5f}" if best_dev_eer is not None and not np.isnan(best_dev_eer) else "",
                f"{best_dev_tdcf:.5f}" if best_dev_tdcf is not None and not np.isnan(best_dev_tdcf) else ""
            ])
    
    def save_json(self):
        """Save metrics to JSON file."""
        with open(self.json_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def get_metrics(self) -> Dict:
        """Get all metrics."""
        return self.metrics


def plot_training_metrics(metrics_dict: Dict, save_dir: Path):
    """
    Plot training and validation metrics.
    
    Args:
        metrics_dict: Dictionary containing metrics
        save_dir: Directory to save plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = metrics_dict['epochs']
    train_loss = metrics_dict['train_loss']
    dev_eer = metrics_dict['dev_eer']
    dev_tdcf = metrics_dict['dev_tdcf']
    best_dev_eer = metrics_dict['best_dev_eer']
    best_dev_tdcf = metrics_dict['best_dev_tdcf']
    eval_eer = metrics_dict['eval_eer']
    eval_tdcf = metrics_dict['eval_tdcf']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, 'b-o', linewidth=2, markersize=4, label='Training Loss')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Development EER
    ax = axes[0, 1]
    ax.plot(epochs, dev_eer, 'g-o', linewidth=2, markersize=4, label='Dev EER')
    ax.plot(epochs, best_dev_eer, 'r--', linewidth=2, label='Best Dev EER')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('EER (%)', fontsize=11)
    ax.set_title('Equal Error Rate (EER)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Development t-DCF
    ax = axes[1, 0]
    ax.plot(epochs, dev_tdcf, 'orange', marker='o', linewidth=2, markersize=4, label='Dev t-DCF')
    ax.plot(epochs, best_dev_tdcf, 'r--', linewidth=2, label='Best Dev t-DCF')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('t-DCF', fontsize=11)
    ax.set_title('Tandem Detection Cost Function (t-DCF)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 4: Eval vs Dev comparison
    ax = axes[1, 1]
    ax.plot(epochs, dev_eer, 'g-o', linewidth=2, markersize=4, label='Dev EER')
    
    # Plot eval metrics if available
    eval_epochs = [e for e, v in zip(epochs, eval_eer) if not np.isnan(v)]
    eval_eer_vals = [v for v in eval_eer if not np.isnan(v)]
    if eval_eer_vals:
        ax.plot(eval_epochs, eval_eer_vals, 'purple', marker='s', linewidth=2, markersize=6, label='Eval EER')
    
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('EER (%)', fontsize=11)
    ax.set_title('Development vs Evaluation EER', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    save_path = save_dir / "training_metrics.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training metrics plot saved to {save_path}")
    plt.close()


def plot_final_metrics(metrics_dict: Dict, final_eer: float, final_tdcf: float, save_dir: Path):
    """
    Create a summary plot of final metrics.
    
    Args:
        metrics_dict: Dictionary containing metrics
        final_eer: Final EER value
        final_tdcf: Final t-DCF value
        save_dir: Directory to save plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = metrics_dict['epochs']
    dev_eer = metrics_dict['dev_eer']
    dev_tdcf = metrics_dict['dev_tdcf']
    
    # Create summary figure
    fig = plt.figure(figsize=(14, 5))
    
    # Plot 1: Best metrics over time
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(epochs, dev_eer, 'g-o', linewidth=2, markersize=4, label='Dev EER')
    ax1.axhline(y=final_eer, color='r', linestyle='--', linewidth=2, label=f'Final EER: {final_eer:.3f}%')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('EER (%)', fontsize=12)
    ax1.set_title('Final Equal Error Rate (EER)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Plot 2: Best t-DCF over time
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(epochs, dev_tdcf, 'orange', marker='o', linewidth=2, markersize=4, label='Dev t-DCF')
    ax2.axhline(y=final_tdcf, color='r', linestyle='--', linewidth=2, label=f'Final t-DCF: {final_tdcf:.5f}')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('t-DCF', fontsize=12)
    ax2.set_title('Final Tandem Detection Cost Function (t-DCF)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    save_path = save_dir / "final_metrics.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Final metrics plot saved to {save_path}")
    plt.close()


def create_metrics_summary(metrics_dict: Dict, final_eer: float, final_tdcf: float, 
                          save_dir: Path, config: Dict = None):
    """
    Create a text summary of metrics.
    
    Args:
        metrics_dict: Dictionary containing metrics
        final_eer: Final EER value
        final_tdcf: Final t-DCF value
        save_dir: Directory to save summary
        config: Configuration dictionary (optional)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    summary_file = save_dir / "metrics_summary.txt"
    
    epochs = metrics_dict['epochs']
    train_loss = metrics_dict['train_loss']
    dev_eer = metrics_dict['dev_eer']
    dev_tdcf = metrics_dict['dev_tdcf']
    best_dev_eer = metrics_dict['best_dev_eer']
    best_dev_tdcf = metrics_dict['best_dev_tdcf']
    
    with open(summary_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("METRICS SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        if config:
            f.write("CONFIGURATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Epochs: {config.get('num_epochs', 'N/A')}\n")
            f.write(f"Batch Size: {config.get('batch_size', 'N/A')}\n")
            f.write(f"Feature Type: {config.get('feature_type', 0)}\n")
            f.write(f"Random Augmentation: {config.get('random_noise', False)}\n")
            f.write("\n")
        
        f.write("FINAL RESULTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Final EER: {final_eer:.4f}%\n")
        f.write(f"Final t-DCF: {final_tdcf:.6f}\n")
        f.write("\n")
        
        f.write("BEST DEVELOPMENT METRICS\n")
        f.write("-" * 70 + "\n")
        if best_dev_eer:
            best_eer_idx = np.nanargmin(best_dev_eer)
            best_eer_val = best_dev_eer[best_eer_idx]
            f.write(f"Best Dev EER: {best_eer_val:.4f}% (Epoch {epochs[best_eer_idx]})\n")
        
        if best_dev_tdcf:
            best_tdcf_idx = np.nanargmin(best_dev_tdcf)
            best_tdcf_val = best_dev_tdcf[best_tdcf_idx]
            f.write(f"Best Dev t-DCF: {best_tdcf_val:.6f} (Epoch {epochs[best_tdcf_idx]})\n")
        f.write("\n")
        
        f.write("TRAINING STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total Epochs: {len(epochs)}\n")
        f.write(f"Initial Train Loss: {train_loss[0]:.5f}\n")
        f.write(f"Final Train Loss: {train_loss[-1]:.5f}\n")
        f.write(f"Min Train Loss: {np.min(train_loss):.5f}\n")
        f.write(f"Average Dev EER: {np.mean(dev_eer):.4f}%\n")
        f.write(f"Average Dev t-DCF: {np.mean(dev_tdcf):.6f}\n")
        f.write("\n")
        
        f.write("=" * 70 + "\n")
    
    print(f"✓ Metrics summary saved to {summary_file}")


def save_all_metrics(metrics_dict: Dict, final_eer: float, final_tdcf: float, 
                     save_dir: Path, config: Dict = None):
    """
    Save and visualize all metrics.
    
    Args:
        metrics_dict: Dictionary containing metrics
        final_eer: Final EER value
        final_tdcf: Final t-DCF value
        save_dir: Directory to save all metrics and plots
        config: Configuration dictionary (optional)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("SAVING METRICS AND VISUALIZATIONS")
    print("=" * 70)
    
    # Save metrics to files
    tracker = MetricsTracker(save_dir)
    tracker.metrics = metrics_dict
    tracker.save_json()
    print(f"✓ Metrics saved to {tracker.json_file}")
    print(f"✓ Metrics CSV saved to {tracker.csv_file}")
    
    # Create visualizations
    plot_training_metrics(metrics_dict, save_dir)
    plot_final_metrics(metrics_dict, final_eer, final_tdcf, save_dir)
    create_metrics_summary(metrics_dict, final_eer, final_tdcf, save_dir, config)
    
    print("=" * 70 + "\n")
