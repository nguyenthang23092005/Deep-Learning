import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import cv2
import os
from pathlib import Path
import random

# Cấu hình
DATA_DIR = "data"
OUTPUT_DIR = "visual"
SAMPLE_RATE = 16000
DURATION = 2.5

# Cấu hình spectrogram
N_FFT = 512
HOP_LENGTH = 256
N_MELS = 64
FMIN = 20
FMAX = 8000

# Kích thước resize cho NN model
RESIZE_SHAPE = (32, 32)

# Các lệnh cần phân loại (6 lệnh chính)
COMMANDS_TO_CLASSIFY = [
    'bat_thong_bao_chay',
    'tat_thong_bao_chay',
    'mo_cua',
    'dong_cua',
    'tang_nhiet_do',
    'giam_nhiet_do'
]

def load_random_sample():
    """Load một mẫu ngẫu nhiên từ dataset (chỉ từ 6 lệnh cần phân loại)"""
    # Random chọn command từ 6 lệnh cần phân loại
    command = random.choice(COMMANDS_TO_CLASSIFY)
    command_folder = os.path.join(DATA_DIR, command)
    
    # Lấy tất cả audio files
    audio_files = list(Path(command_folder).glob('*.wav'))
    if len(audio_files) == 0:
        return None, None
    
    # Random chọn file
    audio_file = random.choice(audio_files)
    
    # Load audio
    y, sr = librosa.load(str(audio_file), sr=SAMPLE_RATE, duration=DURATION)
    
    return y, sr, command, str(audio_file)

def create_spectrogram(y, sr):
    """Tạo log-mel spectrogram từ audio"""
    # Tạo mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX
    )
    
    # Convert to log scale (dB)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    return log_mel_spec

def visualize_model_input():
    """
    Visualize toàn bộ pipeline từ audio đến input của model
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*80)
    print("VISUALIZE MODEL INPUT PIPELINE")
    print("="*80)
    
    # Load random sample
    print("\n📂 Loading random sample...")
    y, sr, command, filepath = load_random_sample()
    
    if y is None:
        print("❌ Không tìm thấy audio file!")
        return
    
    print(f"✅ Loaded: {command}")
    print(f"   File: {Path(filepath).name}")
    
    # Tạo spectrogram
    print("\n🎵 Creating spectrogram...")
    log_mel_spec = create_spectrogram(y, sr)
    print(f"✅ Spectrogram shape: {log_mel_spec.shape}")
    
    # Resize
    print("\n📐 Resizing spectrogram...")
    resized_spec = cv2.resize(log_mel_spec, RESIZE_SHAPE, interpolation=cv2.INTER_LINEAR)
    print(f"✅ Resized shape: {resized_spec.shape}")
    
    # NO Normalization - keep original values
    print("\n🔢 Skipping normalization (using original values)...")
    print(f"   Value range: [{resized_spec.min():.2f}, {resized_spec.max():.2f}] dB")
    
    # Flatten
    print("\n🔄 Flattening...")
    flattened = resized_spec.flatten()
    print(f"✅ Flattened shape: {flattened.shape}")
    
    # Create visualization
    print("\n📊 Creating visualization...")
    fig = plt.figure(figsize=(20, 14))
    
    # Layout: 4 rows x 2 columns (no normalization step)
    gs = fig.add_gridspec(4, 2, hspace=0.7, wspace=0.6, height_ratios=[1, 1.4, 1, 0.3])
    
    # ========== ROW 1: Waveform (full width) ==========
    ax1 = fig.add_subplot(gs[0, :])
    time = np.linspace(0, len(y) / sr, len(y))
    ax1.fill_between(time, y, color='#1f77b4', alpha=0.7)
    ax1.set_xlim(0, DURATION)
    ax1.set_ylim(-1, 1)
    ax1.set_title(f'Step 1: Audio Waveform\nShape: ({len(y)},) samples | Duration: {DURATION}s | Sample Rate: {sr}Hz', 
                  fontsize=11, fontweight='bold', pad=8)
    ax1.set_xlabel('Time (s)', fontsize=9)
    ax1.set_ylabel('Amplitude', fontsize=9)
    ax1.grid(True, alpha=0.3)
    for spine in ax1.spines.values():
        spine.set_linewidth(2)
    
    # ========== ROW 2: Original Spectrogram ==========
    ax2 = fig.add_subplot(gs[1, 0])
    img2 = librosa.display.specshow(log_mel_spec, sr=sr, hop_length=HOP_LENGTH, 
                                     x_axis='time', y_axis='mel', ax=ax2, cmap='viridis')
    ax2.set_title(f'Step 2: Log-Mel Spectrogram\n{log_mel_spec.shape} | [{log_mel_spec.min():.0f}, {log_mel_spec.max():.0f}] dB', 
                  fontsize=9, fontweight='bold', pad=6)
    ax2.set_xlabel('Time (s)', fontsize=9)
    ax2.set_ylabel('Mel Frequency', fontsize=9)
    plt.colorbar(img2, ax=ax2, format='%+2.0f dB')
    for spine in ax2.spines.values():
        spine.set_linewidth(2)
    
    # ========== ROW 2: Resized Spectrogram (for NN input) ==========
    ax3 = fig.add_subplot(gs[1, 1:])
    img3 = ax3.imshow(resized_spec, aspect='auto', origin='lower', cmap='viridis')
    ax3.set_title(f'Step 3: Resized (NN Input)\n{resized_spec.shape} | [{resized_spec.min():.0f}, {resized_spec.max():.0f}] dB | NO normalization', 
                  fontsize=9, fontweight='bold', pad=6)
    ax3.set_xlabel('Time frames', fontsize=9)
    ax3.set_ylabel('Mel bins', fontsize=9)
    plt.colorbar(img3, ax=ax3, format='%+2.0f dB')
    for spine in ax3.spines.values():
        spine.set_linewidth(2)
    
    # ========== ROW 3: Flattened Vector (first 200 values) ==========
    ax5 = fig.add_subplot(gs[2, :])
    display_range = min(200, len(flattened))
    ax5.plot(flattened[:display_range], linewidth=0.8, color='#2ca02c')
    ax5.set_title(f'Step 4: Flattened Vector (Input to NN) | {flattened.shape} | [{flattened.min():.2f}, {flattened.max():.2f}] dB', 
                  fontsize=10, fontweight='bold', pad=6)
    ax5.set_xlabel('Index', fontsize=9)
    ax5.set_ylabel('Value (dB)', fontsize=9)
    ax5.grid(True, alpha=0.3)
    for spine in ax5.spines.values():
        spine.set_linewidth(2)
    
    # ========== ROW 4: Summary Info (full width, horizontal) ==========
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')
    
    summary_text = f"""LABEL: {command}  |  FILE: {Path(filepath).name}  |  MODEL INPUT: (1024,) in dB scale
1. Audio: ({len(y)},) {DURATION}s  |  2. Spectrogram: {log_mel_spec.shape} [{log_mel_spec.min():.0f},{log_mel_spec.max():.0f}]dB  |  3. Resized: {resized_spec.shape}  |  4. Flattened: {flattened.shape} (NO normalization)"""
    
    ax6.text(0.5, 0.5, summary_text, 
             transform=ax6.transAxes,
             fontsize=9,
             verticalalignment='center',
             horizontalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4, pad=0.6))
    
    # Main title
    fig.suptitle(f'NEURAL NETWORK INPUT PIPELINE - Sample: {command.upper()}', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, 'model_input_pipeline.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    
    print(f"\n✅ Đã lưu visualization tại: {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"📌 Label: {command}")
    print(f"📂 File: {Path(filepath).name}")
    print(f"\n📊 Pipeline:")
    print(f"   Step 1: Audio waveform       → Shape: ({len(y)},)")
    print(f"   Step 2: Log-Mel spectrogram  → Shape: {log_mel_spec.shape}")
    print(f"   Step 3: Resize               → Shape: {resized_spec.shape}")
    print(f"   Step 4: Flatten              → Shape: {flattened.shape}")
    print(f"\n🧠 Model Input: Vector {flattened.shape} (dB scale, NO normalization) với nhãn '{command}'")
    print("="*80)
    
    plt.show()

if __name__ == "__main__":
    visualize_model_input()
