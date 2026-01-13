import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
from pathlib import Path
import random

# Cấu hình
DATA_DIR = "data"  # Thư mục chứa audio files
OUTPUT_DIR = "visual"
SAMPLE_RATE = 16000
DURATION = 2.5

# Các lệnh cần visualize
COMMANDS = [
    'tat_thong_bao_chay',
    'mo_cua',
    'tang_nhiet_do',
    'giam_nhiet_do',
    'bat_thong_bao_chay',
    'dong_cua'
]

def load_audio_file(command, sample_idx=0):
    """Load một file audio từ command"""
    command_folder = os.path.join(DATA_DIR, command)
    if not os.path.exists(command_folder):
        return None, None
    
    audio_files = list(Path(command_folder).glob('*.wav'))
    if len(audio_files) == 0:
        return None, None
    
    # Lấy file ngẫu nhiên hoặc theo index
    if sample_idx < len(audio_files):
        audio_file = audio_files[sample_idx]
    else:
        audio_file = random.choice(audio_files)
    
    # Load audio
    y, sr = librosa.load(str(audio_file), sr=SAMPLE_RATE, duration=DURATION)
    return y, sr

def plot_waveforms(num_samples_per_command=1, figsize=(18, 10)):
    """
    Vẽ waveforms của các lệnh voice command
    
    Parameters:
    -----------
    num_samples_per_command : int
        Số lượng samples hiển thị cho mỗi lệnh
    figsize : tuple
        Kích thước figure
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Tính số hàng và cột
    total_plots = len(COMMANDS) * num_samples_per_command
    cols = 3
    rows = (total_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if total_plots > 1 else [axes]
    
    plot_idx = 0
    
    for command in COMMANDS:
        for sample_idx in range(num_samples_per_command):
            if plot_idx >= len(axes):
                break
            
            # Load audio - random chọn sample
            y, sr = load_audio_file(command, sample_idx=random.randint(0, 1000))
            
            if y is not None:
                # Tính time axis
                time = np.linspace(0, len(y) / sr, len(y))
                
                # Vẽ waveform
                axes[plot_idx].fill_between(time, y, color='#1f77b4', alpha=0.8)
                axes[plot_idx].set_ylim(-1, 1)
                axes[plot_idx].set_xlim(0, DURATION)
                
                # Format
                axes[plot_idx].set_title(f"{command} — {DURATION:.2f}s", 
                                        fontsize=11, fontweight='normal')
                axes[plot_idx].set_xlabel('Time', fontsize=9)
                axes[plot_idx].set_yticks([])
                axes[plot_idx].grid(False)
                
                # Thêm khung bao
                for spine in axes[plot_idx].spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(1.5)
                    spine.set_edgecolor('black')
                
                print(f"✅ Plotted: {command} (sample {sample_idx + 1})")
            else:
                axes[plot_idx].text(0.5, 0.5, f'No data\n{command}', 
                                   ha='center', va='center', fontsize=10)
                axes[plot_idx].set_xticks([])
                axes[plot_idx].set_yticks([])
                print(f"⚠️  No data for: {command}")
            
            plot_idx += 1
    
    # Ẩn các subplot thừa
    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Lưu file
    output_path = os.path.join(OUTPUT_DIR, 'waveforms_comparison.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Đã lưu hình tại: {output_path}")
    plt.show()

def plot_random_waveforms(num_plots=6, figsize=(18, 10)):
    """
    Vẽ random waveforms từ các lệnh khác nhau
    
    Parameters:
    -----------
    num_plots : int
        Tổng số waveforms cần vẽ
    figsize : tuple
        Kích thước figure
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Tính số hàng và cột
    cols = 3
    rows = (num_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if num_plots > 1 else [axes]
    
    for plot_idx in range(num_plots):
        if plot_idx >= len(axes):
            break
        
        # Random chọn command
        command = random.choice(COMMANDS)
        
        # Load audio
        y, sr = load_audio_file(command)
        
        if y is not None:
            # Tính time axis
            time = np.linspace(0, len(y) / sr, len(y))
            
            # Vẽ waveform
            axes[plot_idx].fill_between(time, y, color='#1f77b4', alpha=0.8)
            axes[plot_idx].set_ylim(-1, 1)
            axes[plot_idx].set_xlim(0, DURATION)
            
            # Format
            axes[plot_idx].set_title(f"{command} — {DURATION:.2f}s", 
                                    fontsize=11, fontweight='normal')
            axes[plot_idx].set_xlabel('Time', fontsize=9)
            axes[plot_idx].set_yticks([])
            axes[plot_idx].grid(False)
            
            # Thêm khung bao
            for spine in axes[plot_idx].spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.5)
                spine.set_edgecolor('black')
            
            print(f"✅ Plotted {plot_idx + 1}/{num_plots}: {command}")
        else:
            axes[plot_idx].text(0.5, 0.5, f'No data\n{command}', 
                               ha='center', va='center', fontsize=10)
            axes[plot_idx].set_xticks([])
            axes[plot_idx].set_yticks([])
    
    # Ẩn các subplot thừa
    for idx in range(num_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Lưu file
    output_path = os.path.join(OUTPUT_DIR, 'waveforms_random.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Đã lưu hình tại: {output_path}")
    plt.show()

def plot_comparison_grid(commands=None, samples_per_command=2, figsize=(18, 12)):
    """
    Vẽ comparison grid với nhiều samples cho mỗi command
    
    Parameters:
    -----------
    commands : list
        Danh sách commands cần visualize (None = dùng COMMANDS mặc định)
    samples_per_command : int
        Số samples cho mỗi command
    figsize : tuple
        Kích thước figure
    """
    if commands is None:
        commands = COMMANDS
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    total_plots = len(commands) * samples_per_command
    cols = 3
    rows = (total_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if total_plots > 1 else [axes]
    
    plot_idx = 0
    
    for command in commands:
        for sample_idx in range(samples_per_command):
            if plot_idx >= len(axes):
                break
            
            y, sr = load_audio_file(command, sample_idx)
            
            if y is not None:
                time = np.linspace(0, len(y) / sr, len(y))
                
                axes[plot_idx].fill_between(time, y, color='#1f77b4', alpha=0.8)
                axes[plot_idx].set_ylim(-1, 1)
                axes[plot_idx].set_xlim(0, DURATION)
                axes[plot_idx].set_title(f"{command} — {DURATION:.2f}s", 
                                        fontsize=11, fontweight='normal')
                axes[plot_idx].set_xlabel('Time', fontsize=9)
                axes[plot_idx].set_yticks([])
                axes[plot_idx].grid(False)
                
                # Thêm khung bao
                for spine in axes[plot_idx].spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(1.5)
                    spine.set_edgecolor('black')
            else:
                axes[plot_idx].text(0.5, 0.5, f'No data\n{command}', 
                                   ha='center', va='center', fontsize=10)
                axes[plot_idx].set_xticks([])
                axes[plot_idx].set_yticks([])
            
            plot_idx += 1
    
    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'waveforms_grid.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Đã lưu hình tại: {output_path}")
    plt.show()

if __name__ == "__main__":
    print("="*70)
    print("VISUALIZE AUDIO WAVEFORMS")
    print("="*70)
    
    # Vẽ waveforms - mỗi lệnh 1 ô
    print("\n📊 Vẽ waveforms cho từng lệnh...")
    plot_waveforms(num_samples_per_command=1, figsize=(18, 10))
    
    print("\n" + "="*70)
    print("HOÀN THÀNH!")
    print("="*70)
