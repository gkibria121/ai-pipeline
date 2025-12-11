#!/usr/bin/env python3
"""
Visualize training results from saved metrics.

Usage:
    python visualize_results.py --path exp_result/RawNet3_FakeOrReal_2024-12-10_15-30-45/metrics
    python visualize_results.py --path exp_result/SEResNet_*/metrics
    python visualize_results.py --path exp_result/*/metrics --compare
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from glob import glob

# Detect if running in notebook
try:
    from IPython.display import display, Image as IPImage
    import io
    get_ipython()
    IN_NOTEBOOK = True
    get_ipython().run_line_magic('matplotlib', 'inline')
except:
    IN_NOTEBOOK = False
    matplotlib.use('Agg')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def show_plot_in_notebook(fig):
    """Display plot in notebook using IPython.display"""
    if IN_NOTEBOOK:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        display(IPImage(buf.read()))
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def load_metrics(metrics_path: Path) -> Optional[Dict]:
    """Load metrics from JSON file."""
    json_file = metrics_path / "metrics.json"
    if not json_file.exists():
        print(f"[ERROR] Metrics file not found: {json_file}")
        return None
    
    try:
        with open(json_file, 'r') as f:
            metrics = json.load(f)
        print(f"[OK] Loaded metrics from {json_file}")
        return metrics
    except Exception as e:
        print(f"[ERROR] Error loading metrics: {e}")
        return None


def plot_training_curves(metrics: Dict, save_path: Path, title_suffix: str = ""):
    """Plot training loss and dev EER curves."""
    epochs = metrics['epochs']
    train_loss = metrics['train_loss']
    dev_eer = metrics['dev_eer']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Training Loss
    ax = axes[0]
    ax.plot(epochs, train_loss, 'b-o', linewidth=2, markersize=4, label='Training Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Training Loss{title_suffix}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Development EER
    ax = axes[1]
    ax.plot(epochs, dev_eer, 'g-o', linewidth=2, markersize=4, label='Dev EER')
    if metrics.get('best_dev_eer'):
        best_eer = min([x for x in metrics['best_dev_eer'] if not np.isnan(x)])
        ax.axhline(y=best_eer, color='r', linestyle='--', linewidth=2, 
                   label=f'Best: {best_eer:.3f}%')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('EER (%)', fontsize=12)
    ax.set_title(f'Development EER{title_suffix}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved training curves to {save_path}")
    show_plot_in_notebook(fig)


def plot_accuracy_curves(metrics: Dict, save_path: Path, title_suffix: str = ""):
    """Plot accuracy curves for dev and eval sets."""
    epochs = metrics['epochs']
    dev_acc = metrics.get('dev_acc', [])
    eval_acc = metrics.get('eval_acc', [])
    
    fig = plt.figure(figsize=(10, 6))
    
    if dev_acc:
        dev_acc_clean = [x if not np.isnan(x) else None for x in dev_acc]
        plt.plot(epochs, dev_acc_clean, 'g-o', linewidth=2, markersize=5, 
                label='Dev Accuracy', alpha=0.8)
    
    if eval_acc:
        eval_epochs = [e for e, v in zip(epochs, eval_acc) if not np.isnan(v)]
        eval_acc_vals = [v for v in eval_acc if not np.isnan(v)]
        if eval_acc_vals:
            plt.plot(eval_epochs, eval_acc_vals, 'purple', marker='s', 
                    linewidth=2, markersize=7, label='Eval Accuracy', alpha=0.8)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'Model Accuracy{title_suffix}', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved accuracy curves to {save_path}")
    show_plot_in_notebook(fig)


def plot_eer_comparison(metrics: Dict, save_path: Path, title_suffix: str = ""):
    """Plot dev vs eval EER comparison."""
    epochs = metrics['epochs']
    dev_eer = metrics['dev_eer']
    eval_eer = metrics.get('eval_eer', [])
    
    fig = plt.figure(figsize=(10, 6))
    plt.plot(epochs, dev_eer, 'g-o', linewidth=2, markersize=5, 
            label='Dev EER', alpha=0.8)
    
    if eval_eer:
        eval_epochs = [e for e, v in zip(epochs, eval_eer) if not np.isnan(v)]
        eval_eer_vals = [v for v in eval_eer if not np.isnan(v)]
        if eval_eer_vals:
            plt.plot(eval_epochs, eval_eer_vals, 'purple', marker='s', 
                    linewidth=2, markersize=7, label='Eval EER', alpha=0.8)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('EER (%)', fontsize=12)
    plt.title(f'Equal Error Rate Comparison{title_suffix}', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved EER comparison to {save_path}")
    show_plot_in_notebook(fig)


def plot_all_metrics(metrics: Dict, save_path: Path, title_suffix: str = "", show: bool = False):
    """Create comprehensive 4-panel metrics plot."""
    epochs = metrics['epochs']
    train_loss = metrics['train_loss']
    dev_eer = metrics['dev_eer']
    dev_acc = metrics.get('dev_acc', [])
    dev_tdcf = metrics.get('dev_tdcf', [])
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Metrics{title_suffix}', fontsize=16, fontweight='bold')
    
    # Training Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, 'b-o', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Development EER
    ax = axes[0, 1]
    ax.plot(epochs, dev_eer, 'g-o', linewidth=2, markersize=4, label='Dev EER')
    if metrics.get('best_dev_eer'):
        best_eer = min([x for x in metrics['best_dev_eer'] if not np.isnan(x)])
        ax.axhline(y=best_eer, color='r', linestyle='--', linewidth=2, 
                   label=f'Best: {best_eer:.3f}%')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('EER (%)', fontsize=11)
    ax.set_title('Equal Error Rate', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Accuracy
    ax = axes[1, 0]
    if dev_acc:
        dev_acc_clean = [x if not np.isnan(x) else None for x in dev_acc]
        ax.plot(epochs, dev_acc_clean, 'orange', marker='o', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Development Accuracy', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # t-DCF
    ax = axes[1, 1]
    if dev_tdcf:
        dev_tdcf_clean = [x if not np.isnan(x) else None for x in dev_tdcf]
        ax.plot(epochs, dev_tdcf_clean, 'm-o', linewidth=2, markersize=4, label='Dev t-DCF')
        if metrics.get('best_dev_tdcf'):
            best_tdcf = min([x for x in metrics['best_dev_tdcf'] if not np.isnan(x)])
            ax.axhline(y=best_tdcf, color='r', linestyle='--', linewidth=2, 
                       label=f'Best: {best_tdcf:.5f}')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('t-DCF', fontsize=11)
    ax.set_title('Detection Cost Function', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved comprehensive metrics to {save_path}")
    if show or IN_NOTEBOOK:
        plt.show()
    if not (show or IN_NOTEBOOK):
        plt.close()


def plot_comparison(metrics_list: List[Dict], labels: List[str], save_path: Path, show: bool = False):
    """Compare multiple training runs."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_list)))
    
    # Training Loss Comparison
    ax = axes[0, 0]
    for metrics, label, color in zip(metrics_list, labels, colors):
        epochs = metrics['epochs']
        train_loss = metrics['train_loss']
        ax.plot(epochs, train_loss, '-o', linewidth=2, markersize=3, 
               label=label, color=color, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Training Loss Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Dev EER Comparison
    ax = axes[0, 1]
    for metrics, label, color in zip(metrics_list, labels, colors):
        epochs = metrics['epochs']
        dev_eer = metrics['dev_eer']
        ax.plot(epochs, dev_eer, '-o', linewidth=2, markersize=3, 
               label=label, color=color, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('EER (%)', fontsize=11)
    ax.set_title('Development EER Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Accuracy Comparison
    ax = axes[1, 0]
    for metrics, label, color in zip(metrics_list, labels, colors):
        epochs = metrics['epochs']
        dev_acc = metrics.get('dev_acc', [])
        if dev_acc:
            dev_acc_clean = [x if not np.isnan(x) else None for x in dev_acc]
            ax.plot(epochs, dev_acc_clean, '-o', linewidth=2, markersize=3, 
                   label=label, color=color, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Best Metrics Summary
    ax = axes[1, 1]
    best_eers = []
    best_accs = []
    for metrics in metrics_list:
        if metrics.get('best_dev_eer'):
            best_eer = min([x for x in metrics['best_dev_eer'] if not np.isnan(x)])
            best_eers.append(best_eer)
        else:
            best_eers.append(min(metrics['dev_eer']))
        
        if metrics.get('dev_acc'):
            best_acc = max([x for x in metrics['dev_acc'] if not np.isnan(x)])
            best_accs.append(best_acc)
        else:
            best_accs.append(0)
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax.bar(x - width/2, best_eers, width, label='Best EER (%)', alpha=0.8)
    ax2 = ax.twinx()
    ax2.bar(x + width/2, best_accs, width, label='Best Acc (%)', alpha=0.8, color='orange')
    
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Best EER (%)', fontsize=11)
    ax2.set_ylabel('Best Accuracy (%)', fontsize=11)
    ax.set_title('Best Metrics Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved comparison plot to {save_path}")
    if show or IN_NOTEBOOK:
        plt.show()
    if not (show or IN_NOTEBOOK):
        plt.close()


def print_summary(metrics: Dict, model_name: str = "Model"):
    """Print summary statistics."""
    print(f"\n{'='*70}")
    print(f"[SUMMARY] {model_name}")
    print(f"{'='*70}")
    
    epochs = metrics['epochs']
    train_loss = metrics['train_loss']
    dev_eer = metrics['dev_eer']
    dev_acc = metrics.get('dev_acc', [])
    
    print(f"Total Epochs: {len(epochs)}")
    print(f"\nTraining Loss:")
    print(f"  Initial: {train_loss[0]:.5f}")
    print(f"  Final:   {train_loss[-1]:.5f}")
    print(f"  Min:     {min(train_loss):.5f}")
    
    print(f"\nDevelopment EER:")
    print(f"  Best:    {min(dev_eer):.4f}%")
    print(f"  Final:   {dev_eer[-1]:.4f}%")
    
    if dev_acc:
        dev_acc_clean = [x for x in dev_acc if not np.isnan(x)]
        if dev_acc_clean:
            print(f"\nDevelopment Accuracy:")
            print(f"  Best:    {max(dev_acc_clean):.2f}%")
            print(f"  Final:   {dev_acc_clean[-1]:.2f}%")
    
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize training results from saved metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize single model
  python visualize_results.py --path exp_result/RawNet3_*/metrics
  
  # Compare multiple models
  python visualize_results.py --path "exp_result/*/metrics" --compare
  
  # Specify output directory
  python visualize_results.py --path exp_result/RawNet3_*/metrics --output ./plots
        """
    )
    
    parser.add_argument(
        '--path', 
        type=str, 
        required=True,
        help='Path to metrics directory (supports wildcards for comparison)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory to save plots (if not specified, plots are only displayed)'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Create comparison plots when multiple paths match'
    )
    
    parser.add_argument(
        '--show-summary',
        action='store_true',
        help='Print summary statistics'
    )
    
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display plots interactively instead of/in addition to saving'
    )
    
    args = parser.parse_args()
    
    # Find all matching paths
    paths = glob(args.path)
    if not paths:
        print(f"[ERROR] No paths found matching: {args.path}")
        sys.exit(1)
    
    paths = [Path(p) for p in paths]
    print(f"\n[OK] Found {len(paths)} path(s)")
    
    # Load metrics
    metrics_list = []
    labels = []
    
    for path in paths:
        metrics = load_metrics(path)
        if metrics:
            metrics_list.append(metrics)
            # Extract model name from path
            model_name = path.parent.name
            labels.append(model_name)
    
    if not metrics_list:
        print("[ERROR] No valid metrics found")
        sys.exit(1)
    
    print(f"\n[OK] Loaded {len(metrics_list)} metric file(s)")
    
    # Determine if we should save files
    save_plots = args.output is not None
    
    if save_plots:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[OK] Saving plots to: {output_dir}\n")
    else:
        output_dir = None
        if IN_NOTEBOOK:
            print(f"[INFO] Displaying plots in notebook\n")
        else:
            print(f"[INFO] Displaying plots only (use --output <dir> to save)\n")
    
    # Generate visualizations
    if len(metrics_list) == 1:
        # Single model visualization
        metrics = metrics_list[0]
        model_name = labels[0]
        
        print(f"📈 Generating visualizations for: {model_name}")
        
        save_path = output_dir if save_plots else None
        plot_training_curves(metrics, save_path / "training_curves.png" if save_path else None, f" - {model_name}")
        plot_accuracy_curves(metrics, save_path / "accuracy_curves.png" if save_path else None, f" - {model_name}")
        plot_eer_comparison(metrics, save_path / "eer_comparison.png" if save_path else None, f" - {model_name}")
        plot_all_metrics(metrics, save_path / "all_metrics.png" if save_path else None, f" - {model_name}")
        
        if args.show_summary:
            print_summary(metrics, model_name)
        
    else:
        # Multiple models
        if args.compare:
            print(f"[INFO] Generating comparison plots for {len(metrics_list)} models")
            save_path = output_dir / "model_comparison.png" if save_plots else None
            plot_comparison(metrics_list, labels, save_path)
        
        # Individual plots for each model
        for metrics, label in zip(metrics_list, labels):
            print(f"\n📈 Generating visualizations for: {label}")
            
            if save_plots:
                model_output_dir = output_dir / label
                model_output_dir.mkdir(parents=True, exist_ok=True)
                save_base = model_output_dir
            else:
                save_base = None
            
            plot_training_curves(metrics, save_base / "training_curves.png" if save_base else None, f" - {label}")
            plot_accuracy_curves(metrics, save_base / "accuracy_curves.png" if save_base else None, f" - {label}")
            plot_eer_comparison(metrics, save_base / "eer_comparison.png" if save_base else None, f" - {label}")
            plot_all_metrics(metrics, save_base / "all_metrics.png" if save_base else None, f" - {label}")
            
            if args.show_summary:
                print_summary(metrics, label)
    
    if save_plots:
        print(f"\n[OK] All visualizations displayed and saved to: {output_dir}")
    else:
        print(f"\n[OK] All visualizations displayed!")


if __name__ == "__main__":
    main()