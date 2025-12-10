"""
Metrics tracking, saving, and visualization module for ASVspoof detection.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


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
            'dev_acc': [],
            'eval_eer': [],
            'eval_tdcf': [],
            'eval_acc': [],
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
                writer.writerow(['epoch', 'train_loss', 'dev_eer', 'dev_tdcf', 'dev_acc',
                                'eval_eer', 'eval_tdcf', 'eval_acc', 'best_dev_eer', 'best_dev_tdcf'])
    
    def add_epoch(self, epoch: int, train_loss: float, dev_eer: float, 
                  dev_tdcf: float, dev_acc: float, eval_eer: Optional[float] = None, 
                  eval_tdcf: Optional[float] = None, eval_acc: Optional[float] = None,
                  best_dev_eer: float = None, best_dev_tdcf: float = None):
        """
        Add metrics for an epoch.
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            dev_eer: Development EER
            dev_tdcf: Development t-DCF
            dev_acc: Development accuracy
            eval_eer: Evaluation EER (optional)
            eval_tdcf: Evaluation t-DCF (optional)
            eval_acc: Evaluation accuracy (optional)
            best_dev_eer: Best development EER so far
            best_dev_tdcf: Best development t-DCF so far
        """
        self.metrics['epochs'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['dev_eer'].append(dev_eer)
        self.metrics['dev_tdcf'].append(dev_tdcf)
        self.metrics['dev_acc'].append(dev_acc)
        self.metrics['eval_eer'].append(eval_eer if eval_eer is not None else np.nan)
        self.metrics['eval_tdcf'].append(eval_tdcf if eval_tdcf is not None else np.nan)
        self.metrics['eval_acc'].append(eval_acc if eval_acc is not None else np.nan)
        self.metrics['best_dev_eer'].append(best_dev_eer if best_dev_eer is not None else np.nan)
        self.metrics['best_dev_tdcf'].append(best_dev_tdcf if best_dev_tdcf is not None else np.nan)
        
        # Save to CSV
        self._save_csv_row(epoch, train_loss, dev_eer, dev_tdcf, dev_acc, eval_eer, eval_tdcf, eval_acc, best_dev_eer, best_dev_tdcf)
        
        # Save to JSON
        self.save_json()
    
    def _save_csv_row(self, epoch, train_loss, dev_eer, dev_tdcf, dev_acc, eval_eer, eval_tdcf, eval_acc, best_dev_eer, best_dev_tdcf):
        """Save a single row to CSV."""
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.5f}",
                f"{dev_eer:.5f}",
                f"{dev_tdcf:.5f}",
                f"{dev_acc:.2f}" if dev_acc is not None and not np.isnan(dev_acc) else "",
                f"{eval_eer:.5f}" if eval_eer is not None and not np.isnan(eval_eer) else "",
                f"{eval_tdcf:.5f}" if eval_tdcf is not None and not np.isnan(eval_tdcf) else "",
                f"{eval_acc:.2f}" if eval_acc is not None and not np.isnan(eval_acc) else "",
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


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         save_path: Path, title: str = "Confusion Matrix",
                         labels: List[str] = None):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot
        title: Plot title
        labels: Class labels (default: ['Fake', 'Real'])
    """
    if labels is None:
        labels = ['Fake/Spoof', 'Real/Bonafide']
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add accuracy text
    accuracy = 100 * np.trace(cm) / np.sum(cm)
    plt.text(0.5, -0.15, f'Accuracy: {accuracy:.2f}%', 
             ha='center', va='center', transform=plt.gca().transAxes,
             fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {save_path}")
    plt.close()
    
    return cm


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, 
                   save_path: Path, title: str = "ROC Curve"):
    """
    Plot and save ROC curve.
    
    Args:
        y_true: True labels
        y_scores: Prediction scores (probabilities)
        save_path: Path to save the plot
        title: Plot title
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    
    # Find EER point
    eer_idx = np.nanargmin(np.abs(fpr - (1 - tpr)))
    eer = fpr[eer_idx]
    plt.plot(eer, 1-eer, 'ro', markersize=10, 
             label=f'EER = {eer*100:.2f}%')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ ROC curve saved to {save_path}")
    plt.close()
    
    return roc_auc, eer


def plot_accuracy_comparison(metrics_dict: Dict, save_path: Path):
    """
    Plot comparison of train, dev, and eval accuracies.
    
    Args:
        metrics_dict: Dictionary containing metrics
        save_path: Path to save the plot
    """
    epochs = metrics_dict['epochs']
    dev_acc = metrics_dict.get('dev_acc', [])
    eval_acc = metrics_dict.get('eval_acc', [])
    
    plt.figure(figsize=(12, 7))
    
    if dev_acc:
        plt.plot(epochs, dev_acc, 'g-o', linewidth=2, markersize=5, 
                label='Dev Accuracy', alpha=0.8)
    
    if eval_acc:
        # Plot eval accuracy only for epochs where it exists
        eval_epochs = [e for e, v in zip(epochs, eval_acc) if not np.isnan(v)]
        eval_acc_vals = [v for v in eval_acc if not np.isnan(v)]
        if eval_acc_vals:
            plt.plot(eval_epochs, eval_acc_vals, 'purple', marker='s', 
                    linewidth=2, markersize=7, label='Eval Accuracy', alpha=0.8)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Model Accuracy Over Training', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Accuracy comparison saved to {save_path}")
    plt.close()


def create_classification_report(y_true: np.ndarray, y_pred: np.ndarray, 
                                 save_path: Path, 
                                 labels: List[str] = None):
    """
    Create and save detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the report
        labels: Class labels
    """
    if labels is None:
        labels = ['Fake/Spoof', 'Real/Bonafide']
    
    report = classification_report(y_true, y_pred, target_names=labels, digits=4)
    
    with open(save_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(report)
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"✓ Classification report saved to {save_path}")
    return report


def generate_prediction_visualizations(y_true: np.ndarray, y_pred: np.ndarray, 
                                       y_scores: np.ndarray, save_dir: Path,
                                       split_name: str = "eval"):
    """
    Generate comprehensive prediction visualizations.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_scores: Prediction scores
        save_dir: Directory to save visualizations
        split_name: Name of the data split (e.g., 'train', 'dev', 'eval')
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 Generating {split_name} visualizations...")
    
    # Confusion Matrix
    cm_path = save_dir / f"confusion_matrix_{split_name}.png"
    plot_confusion_matrix(y_true, y_pred, cm_path, 
                         title=f"Confusion Matrix - {split_name.capitalize()}")
    
    # ROC Curve
    roc_path = save_dir / f"roc_curve_{split_name}.png"
    roc_auc, eer = plot_roc_curve(y_true, y_scores, roc_path,
                                   title=f"ROC Curve - {split_name.capitalize()}")
    
    # Classification Report
    report_path = save_dir / f"classification_report_{split_name}.txt"
    create_classification_report(y_true, y_pred, report_path)
    
    return {'roc_auc': roc_auc, 'eer': eer}


def display_final_summary(metrics_dict: Dict, final_eval_metrics: Dict, 
                         save_dir: Path):
    """
    Display and save final training summary with all metrics.
    
    Args:
        metrics_dict: Dictionary containing training metrics
        final_eval_metrics: Dictionary with final evaluation metrics
        save_dir: Directory to save summary
    """
    save_dir = Path(save_dir)
    summary_path = save_dir / "final_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(" " * 25 + "FINAL TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("📈 TRAINING METRICS\n")
        f.write("-" * 80 + "\n")
        if metrics_dict.get('train_loss'):
            f.write(f"  Initial Loss: {metrics_dict['train_loss'][0]:.5f}\n")
            f.write(f"  Final Loss:   {metrics_dict['train_loss'][-1]:.5f}\n")
            f.write(f"  Min Loss:     {min(metrics_dict['train_loss']):.5f}\n")
        f.write("\n")
        
        f.write("📊 DEVELOPMENT SET METRICS\n")
        f.write("-" * 80 + "\n")
        if metrics_dict.get('dev_eer'):
            f.write(f"  Best Dev EER:     {min(metrics_dict['dev_eer']):.4f}%\n")
        if metrics_dict.get('dev_acc'):
            dev_acc_clean = [x for x in metrics_dict['dev_acc'] if not np.isnan(x)]
            if dev_acc_clean:
                f.write(f"  Best Dev Accuracy: {max(dev_acc_clean):.2f}%\n")
        f.write("\n")
        
        f.write("🎯 EVALUATION SET METRICS\n")
        f.write("-" * 80 + "\n")
        if final_eval_metrics:
            if 'eer' in final_eval_metrics:
                f.write(f"  EER:       {final_eval_metrics['eer']*100:.4f}%\n")
            if 'roc_auc' in final_eval_metrics:
                f.write(f"  ROC AUC:   {final_eval_metrics['roc_auc']:.4f}\n")
            if 'accuracy' in final_eval_metrics:
                f.write(f"  Accuracy:  {final_eval_metrics['accuracy']:.2f}%\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
    
    # Also print to console
    with open(summary_path, 'r') as f:
        print(f.read())
    
    print(f"✓ Final summary saved to {summary_path}")

