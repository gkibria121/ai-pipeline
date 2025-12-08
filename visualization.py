"""
Visualization utilities for ASVspoof detection results
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns


def plot_roc_curve(y_true, y_scores, output_dir):
    """Plot and save ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Audio Anti-Spoofing Detection', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to {save_path}")
    return roc_auc


def plot_confusion_matrix(y_true, y_pred, output_dir):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Spoof', 'Bonafide'],
                yticklabels=['Spoof', 'Bonafide'],
                annot_kws={'size': 14, 'fontweight': 'bold'},
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix - Audio Anti-Spoofing Detection', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_score_distribution(bona_scores, spoof_scores, output_dir):
    """Plot and save score distributions"""
    plt.figure(figsize=(12, 8))
    
    plt.hist(bona_scores, bins=50, alpha=0.7, label='Bonafide', color='green', edgecolor='black')
    plt.hist(spoof_scores, bins=50, alpha=0.7, label='Spoof', color='red', edgecolor='black')
    
    plt.xlabel('Detection Score', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title('Score Distribution - Bonafide vs Spoof', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    
    save_path = os.path.join(output_dir, 'score_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Score distribution saved to {save_path}")


def plot_metrics_comparison(metrics_dict, output_dir):
    """Plot and save metrics comparison bar chart"""
    plt.figure(figsize=(12, 8))
    
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
    
    bars = plt.bar(metrics, values, color=colors[:len(metrics)], edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    plt.ylim([0, 1.0])
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Metrics comparison saved to {save_path}")


def plot_det_curve(fnr, fpr, thresholds, output_dir):
    """Plot and save DET curve"""
    plt.figure(figsize=(10, 8))
    plt.plot(fpr * 100, fnr * 100, color='darkblue', lw=2, label='DET Curve')
    plt.xlabel('False Acceptance Rate (%)', fontsize=12, fontweight='bold')
    plt.ylabel('False Rejection Rate (%)', fontsize=12, fontweight='bold')
    plt.title('DET Curve - Detection Error Tradeoff', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=11)
    plt.xlim([0.1, 100])
    plt.ylim([0.1, 100])
    plt.xscale('log')
    plt.yscale('log')
    
    save_path = os.path.join(output_dir, 'det_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"DET curve saved to {save_path}")