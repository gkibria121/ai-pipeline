"""
Feature analysis and visualization module.
Provides insights into different audio feature representations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import soundfile as sf

# Detect if running in notebook
try:
    get_ipython()
    IN_NOTEBOOK = True
    # Use inline backend for notebooks
    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
    except:
        pass
except NameError:
    IN_NOTEBOOK = False
    # Use Agg backend for scripts (non-interactive)
    matplotlib.use('Agg')

try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


def analyze_and_visualize_features(
    audio_file: str,
    feature_type: int,
    save_dir: Path,
    sr: int = 16000,
    show: bool = True
):
    """
    Analyze and visualize audio features.
    
    Args:
        audio_file: Path to audio file
        feature_type: Type of feature (0-4)
        save_dir: Directory to save visualizations
        sr: Sample rate
        show: Whether to display plots (default True)
    """
    from data_utils import extract_feature, FEATURE_TYPES
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load audio
    waveform, file_sr = sf.read(audio_file)
    if file_sr != sr:
        if LIBROSA_AVAILABLE:
            waveform = librosa.resample(waveform, orig_sr=file_sr, target_sr=sr)
        else:
            print(f"Warning: Audio sample rate {file_sr} != {sr}, but librosa not available for resampling")
    
    # Limit to 4 seconds for visualization
    max_samples = sr * 4
    if len(waveform) > max_samples:
        waveform = waveform[:max_samples]
    
    feature_name = FEATURE_TYPES.get(feature_type, f"feature_{feature_type}")
    
    # Create visualization
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Waveform
    ax1 = plt.subplot(3, 2, 1)
    time_axis = np.arange(len(waveform)) / sr
    ax1.plot(time_axis, waveform, linewidth=0.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Audio Waveform', fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Extract feature
    feature = extract_feature(waveform, feature_type, sr)
    
    # Plot 2: Feature representation
    ax2 = plt.subplot(3, 2, 2)
    if feature_type == 0:
        # Raw waveform - show spectrum
        if LIBROSA_AVAILABLE:
            D = librosa.amplitude_to_db(np.abs(librosa.stft(waveform)), ref=np.max)
            img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax2)
            ax2.set_title(f'{feature_name.upper()} - Spectrogram', fontweight='bold', fontsize=12)
            plt.colorbar(img, ax=ax2, format='%+2.0f dB')
        else:
            ax2.plot(waveform[:1000])
            ax2.set_title(f'{feature_name.upper()} - Sample View', fontweight='bold', fontsize=12)
    else:
        # Time-frequency features
        if LIBROSA_AVAILABLE:
            img = librosa.display.specshow(feature, sr=sr, x_axis='time', ax=ax2, hop_length=160)
            ax2.set_title(f'{feature_name.upper()} Feature', fontweight='bold', fontsize=12)
            ax2.set_ylabel('Feature Dimension')
            plt.colorbar(img, ax=ax2)
        else:
            ax2.imshow(feature, aspect='auto', origin='lower')
            ax2.set_title(f'{feature_name.upper()} Feature', fontweight='bold', fontsize=12)
    
    # Plot 3: Feature statistics
    ax3 = plt.subplot(3, 2, 3)
    if feature_type == 0:
        # Waveform statistics
        stats_text = f"""
Feature Type: {feature_name.upper()}
Duration: {len(waveform)/sr:.2f} seconds
Sample Rate: {sr} Hz
Samples: {len(waveform)}
Min: {np.min(waveform):.4f}
Max: {np.max(waveform):.4f}
Mean: {np.mean(waveform):.4f}
Std: {np.std(waveform):.4f}
RMS Energy: {np.sqrt(np.mean(waveform**2)):.4f}
"""
    else:
        # Spectrogram-like features
        stats_text = f"""
Feature Type: {feature_name.upper()}
Duration: {len(waveform)/sr:.2f} seconds
Feature Shape: {feature.shape}
Dimensions: {feature.shape[0]}
Time Steps: {feature.shape[1]}
Min: {np.min(feature):.4f}
Max: {np.max(feature):.4f}
Mean: {np.mean(feature):.4f}
Std: {np.std(feature):.4f}
"""
    
    ax3.text(0.1, 0.5, stats_text, transform=ax3.transAxes, 
             fontsize=11, verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax3.axis('off')
    ax3.set_title('Feature Statistics', fontweight='bold', fontsize=12)
    
    # Plot 4: Feature distribution
    ax4 = plt.subplot(3, 2, 4)
    feature_flat = feature.flatten() if feature_type != 0 else waveform
    ax4.hist(feature_flat, bins=50, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Feature Value')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Feature Value Distribution', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Temporal dynamics
    ax5 = plt.subplot(3, 2, 5)
    if feature_type == 0:
        # Energy over time for raw waveform
        frame_length = 512
        hop_length = 160
        energy = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length)[0] if LIBROSA_AVAILABLE else np.array([np.sqrt(np.mean(waveform[i:i+frame_length]**2)) for i in range(0, len(waveform)-frame_length, hop_length)])
        time_frames = np.arange(len(energy)) * hop_length / sr
        ax5.plot(time_frames, energy, linewidth=2)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('RMS Energy')
        ax5.set_title('Energy Envelope', fontweight='bold', fontsize=12)
    else:
        # Mean feature value over time
        mean_over_time = np.mean(feature, axis=0)
        time_frames = np.arange(len(mean_over_time)) * 160 / sr
        ax5.plot(time_frames, mean_over_time, linewidth=2)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Mean Feature Value')
        ax5.set_title('Temporal Mean', fontweight='bold', fontsize=12)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Feature characteristics description
    ax6 = plt.subplot(3, 2, 6)
    
    descriptions = {
        0: """RAW WAVEFORM
• Direct audio signal
• 1D time series
• Captures all information
• Requires more parameters
• Best for simple models
• Sample rate dependent""",
        1: """MEL-SPECTROGRAM
• Perceptually motivated
• 128 mel frequency bins
• Time-frequency representation
• Good for speech tasks
• Mimics human hearing
• Widely used in audio ML""",
        2: """LFCC (Linear Frequency)
• Linear frequency scale
• 13 cepstral coefficients
• Decorrelated features
• Good for voice analysis
• Less computation than MFCC
• Simpler than mel scale""",
        3: """MFCC (Mel Frequency)
• Most popular audio feature
• 13 cepstral coefficients
• Mel frequency scale
• Compact representation
• Standard for ASR/ASV
• Robust to noise""",
        4: """CQT (Constant-Q Transform)
• Logarithmic frequency scale
• 84 frequency bins
• Better low-freq resolution
• Ideal for music/harmonics
• BEST FOR DEEPFAKE DETECTION
• Captures artifacts well"""
    }
    
    desc_text = descriptions.get(feature_type, "Unknown feature type")
    ax6.text(0.1, 0.5, desc_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
    ax6.axis('off')
    ax6.set_title('Feature Characteristics', fontweight='bold', fontsize=12)
    
    plt.suptitle(f'Feature Analysis: {feature_name.upper()}', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    output_path = save_dir / f'feature_analysis_{feature_name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Feature analysis saved to: {output_path}")
    
    # Display if requested or in notebook
    if show or IN_NOTEBOOK:
        plt.show()
    else:
        plt.close()
    
    return output_path
    
    return output_path


def create_feature_comparison(
    audio_file: str,
    save_dir: Path,
    sr: int = 16000,
    show: bool = True
):
    """
    Create a comparison visualization of all feature types.
    
    Args:
        audio_file: Path to audio file
        save_dir: Directory to save visualization
        sr: Sample rate
        show: Whether to display plots (default True)
    """
    from data_utils import extract_feature, FEATURE_TYPES
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load audio
    waveform, file_sr = sf.read(audio_file)
    if file_sr != sr:
        if LIBROSA_AVAILABLE:
            waveform = librosa.resample(waveform, orig_sr=file_sr, target_sr=sr)
    
    # Limit to 2 seconds for comparison
    max_samples = sr * 2
    if len(waveform) > max_samples:
        waveform = waveform[:max_samples]
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, (feat_type, feat_name) in enumerate(FEATURE_TYPES.items()):
        if idx >= 6:
            break
            
        ax = axes[idx]
        feature = extract_feature(waveform, feat_type, sr)
        
        if feat_type == 0:
            # Raw waveform
            time_axis = np.arange(len(feature)) / sr
            ax.plot(time_axis, feature, linewidth=0.8)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
        else:
            # Spectrogram-like features
            if LIBROSA_AVAILABLE:
                img = librosa.display.specshow(feature, sr=sr, x_axis='time', ax=ax, hop_length=160)
                plt.colorbar(img, ax=ax, format='%+2.0f')
            else:
                ax.imshow(feature, aspect='auto', origin='lower')
                ax.set_xlabel('Time')
                ax.set_ylabel('Frequency')
        
        ax.set_title(f'{feat_name.upper()}', fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    # Hide the 6th subplot if only 5 features
    if len(FEATURE_TYPES) < 6:
        axes[5].axis('off')
    
    plt.suptitle('Feature Type Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    output_path = save_dir / 'feature_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Feature comparison saved to: {output_path}")
    
    # Display if requested or in notebook
    if show or IN_NOTEBOOK:
        plt.show()
    else:
        plt.close()
    
    return output_path
    
    return output_path
