import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf

# Tham số
SAMPLE_RATE = 16000
DURATION = 2.5

CONFIGS = {
    'Optimal (Ours)': {
        'n_fft': 512,
        'hop_length': 256,
        'n_mels': 64,
        'fmin': 20,
        'fmax': 8000,
    },
    'High Resolution': {
        'n_fft': 2048,
        'hop_length': 512,
        'n_mels': 128,
        'fmin': 20,
        'fmax': 8000,
    },
    'Low Resolution': {
        'n_fft': 256,
        'hop_length': 128,
        'n_mels': 40,
        'fmin': 20,
        'fmax': 8000,
    },
    'Google Speech': {
        'n_fft': 400,
        'hop_length': 160,
        'n_mels': 40,
        'fmin': 20,
        'fmax': 8000,
    }
}

def generate_test_signal():
    """Tạo tín hiệu test (sweep + voice-like frequencies)"""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION))
    
    # Fundamental frequency (giống giọng người: 100-300 Hz)
    f0 = 200  # Hz
    
    # Tạo signal với harmonics
    signal = np.zeros_like(t)
    for harmonic in range(1, 6):
        freq = f0 * harmonic
        if freq < 8000:  # Chỉ thêm harmonics trong range
            amplitude = 1.0 / harmonic  # Giảm amplitude theo harmonic
            signal += amplitude * np.sin(2 * np.pi * freq * t)
    
    # Normalize
    signal = signal / np.max(np.abs(signal))
    
    # Thêm một chút noise
    signal += 0.05 * np.random.randn(len(signal))
    
    return signal

def compare_configs(audio_signal=None, save_comparison=True):
    """So sánh các cấu hình spectrogram"""
    
    if audio_signal is None:
        print("📢 Tạo test signal...")
        audio_signal = generate_test_signal()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    print("\n" + "="*80)
    print("SO SÁNH CÁC CẤU HÌNH SPECTROGRAM")
    print("="*80)
    
    for idx, (config_name, config) in enumerate(CONFIGS.items()):
        print(f"\n{idx+1}. {config_name}:")
        print(f"   n_fft={config['n_fft']}, hop={config['hop_length']}, n_mels={config['n_mels']}")
        
        # Tạo mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_signal,
            sr=SAMPLE_RATE,
            **config,
            window='hann',
            power=2.0
        )
        
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Tính toán metrics
        shape = log_mel_spec.shape
        time_resolution = config['hop_length'] / SAMPLE_RATE * 1000  # ms
        freq_resolution = SAMPLE_RATE / config['n_fft']  # Hz
        n_params = shape[0] * shape[1]
        
        print(f"   ✓ Shape: {shape} ({shape[0]} mels × {shape[1]} frames)")
        print(f"   ✓ Time resolution: {time_resolution:.2f} ms/frame")
        print(f"   ✓ Freq resolution: {freq_resolution:.2f} Hz/bin")
        print(f"   ✓ Total parameters: {n_params:,}")
        print(f"   ✓ Memory (float32): {n_params * 4 / 1024:.2f} KB")
        
        # Visualize
        ax = axes[idx]
        librosa.display.specshow(
            log_mel_spec,
            sr=SAMPLE_RATE,
            hop_length=config['hop_length'],
            x_axis='time',
            y_axis='mel',
            fmin=config['fmin'],
            fmax=config['fmax'],
            cmap='viridis',
            ax=ax
        )
        ax.set_title(f"{config_name}\nShape: {shape}, Size: {n_params*4/1024:.1f}KB", 
                     fontsize=10)
        
    plt.tight_layout()
    
    if save_comparison:
        plt.savefig('config_comparison.png', dpi=150, bbox_inches='tight')
        print(f"\n✅ Đã lưu hình ảnh so sánh: config_comparison.png")
    
    plt.show()

def test_different_durations():
    """Test với các độ dài audio khác nhau"""
    durations = [1.0, 1.5, 2.0, 2.5, 3.0]
    config = CONFIGS['Optimal (Ours)']
    
    print("\n" + "="*80)
    print("TEST VỚI CÁC ĐỘ DÀI AUDIO KHÁC NHAU")
    print("="*80)
    print(f"\nCấu hình: n_fft={config['n_fft']}, hop={config['hop_length']}, n_mels={config['n_mels']}\n")
    
    fig, axes = plt.subplots(1, len(durations), figsize=(20, 4))
    
    for idx, duration in enumerate(durations):
        # Tạo signal
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
        signal = np.sin(2 * np.pi * 200 * t)  # 200 Hz tone
        
        # Tạo spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=signal,
            sr=SAMPLE_RATE,
            **config,
            window='hann',
            power=2.0
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        n_frames = log_mel_spec.shape[1]
        memory_kb = log_mel_spec.shape[0] * n_frames * 4 / 1024
        
        print(f"{duration}s → Shape: ({config['n_mels']}, {n_frames}) → {memory_kb:.2f} KB")
        
        # Visualize
        librosa.display.specshow(
            log_mel_spec,
            sr=SAMPLE_RATE,
            hop_length=config['hop_length'],
            x_axis='time',
            y_axis='mel',
            fmin=config['fmin'],
            fmax=config['fmax'],
            cmap='viridis',
            ax=axes[idx]
        )
        axes[idx].set_title(f'{duration}s\n{n_frames} frames', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('duration_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Đã lưu: duration_comparison.png")
    plt.show()

def analyze_frequency_range():
    """Phân tích frequency range tối ưu cho giọng nói"""
    print("\n" + "="*80)
    print("PHÂN TÍCH FREQUENCY RANGE")
    print("="*80)
    
    # Tạo signal với các frequencies khác nhau
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION))
    
    freq_ranges = {
        'Fundamental (Male)': 100,
        'Fundamental (Female)': 250,
        'Harmonic 2': 500,
        'Harmonic 3': 750,
        'Harmonic 4': 1000,
        'Harmonic 5': 1250,
        'High Formant': 3000,
        'Very High': 6000,
    }
    
    # Tạo composite signal
    signal = np.zeros_like(t)
    for name, freq in freq_ranges.items():
        signal += np.sin(2 * np.pi * freq * t)
    signal = signal / np.max(np.abs(signal))
    
    # Test các fmax khác nhau
    fmax_options = [4000, 6000, 8000, 11025]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    print("\nSo sánh fmax:")
    for idx, fmax in enumerate(fmax_options):
        config = CONFIGS['Optimal (Ours)'].copy()
        config['fmax'] = fmax
        
        mel_spec = librosa.feature.melspectrogram(
            y=signal,
            sr=SAMPLE_RATE,
            **config,
            window='hann',
            power=2.0
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        print(f"  fmax={fmax}Hz → Captures up to {fmax}Hz")
        
        librosa.display.specshow(
            log_mel_spec,
            sr=SAMPLE_RATE,
            hop_length=config['hop_length'],
            x_axis='time',
            y_axis='mel',
            fmin=config['fmin'],
            fmax=fmax,
            cmap='viridis',
            ax=axes[idx]
        )
        axes[idx].set_title(f'fmax = {fmax} Hz', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('frequency_range_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Đã lưu: frequency_range_comparison.png")
    plt.show()
    
    print("\n📝 Khuyến nghị:")
    print("   - fmax=8000Hz: Tối ưu cho voice commands (capture hết harmonics)")
    print("   - fmax=4000Hz: Đủ cho fundamental + vài harmonics (giảm noise)")
    print("   - fmax>8000Hz: Không cần thiết (Nyquist limit của 16kHz là 8kHz)")

if __name__ == "__main__":
    print("🔬 KIỂM TRA VÀ SO SÁNH CẤU HÌNH SPECTROGRAM")
    print("="*80)
    
    while True:
        print("\n📋 Chọn test:")
        print("  1. So sánh các cấu hình khác nhau")
        print("  2. Test với các độ dài audio khác nhau")
        print("  3. Phân tích frequency range")
        print("  4. Chạy tất cả")
        print("  5. Thoát")
        
        choice = input("\nChọn (1-5): ").strip()
        
        if choice == '1':
            compare_configs()
        elif choice == '2':
            test_different_durations()
        elif choice == '3':
            analyze_frequency_range()
        elif choice == '4':
            compare_configs()
            test_different_durations()
            analyze_frequency_range()
            print("\n✅ Hoàn thành tất cả tests!")
            break
        elif choice == '5':
            print("👋 Tạm biệt!")
            break
        else:
            print("❌ Lựa chọn không hợp lệ!")
