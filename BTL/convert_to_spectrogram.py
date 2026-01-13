import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Cấu hình giống với file thu âm
SAMPLE_RATE = 16000
SPECTROGRAM_CONFIG = {
    'n_fft': 512,
    'hop_length': 256,
    'n_mels': 64,
    'fmin': 20,
    'fmax': 8000,
    'window': 'hann',
    'power': 2.0,
}

# Cấu hình kích thước cố định cho spectrogram
MAX_TIME_FRAMES = 128  # Số frames tối đa (có thể điều chỉnh tùy thuộc vào độ dài audio)

def audio_to_log_mel_spectrogram(audio_path, sr=SAMPLE_RATE, max_frames=MAX_TIME_FRAMES):
    """
    Chuyển đổi file audio thành Log-Mel Spectrogram với kích thước cố định
    
    Parameters:
    -----------
    audio_path : str
        Đường dẫn đến file audio
    sr : int
        Sample rate
    max_frames : int
        Số lượng time frames tối đa (padding/truncating)
        
    Returns:
    --------
    log_mel_spec : numpy.ndarray
        Log-Mel Spectrogram với shape (n_mels, max_frames)
    """
    # Load audio file
    audio, _ = librosa.load(audio_path, sr=sr)
    
    # Tạo Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=SPECTROGRAM_CONFIG['n_fft'],
        hop_length=SPECTROGRAM_CONFIG['hop_length'],
        n_mels=SPECTROGRAM_CONFIG['n_mels'],
        fmin=SPECTROGRAM_CONFIG['fmin'],
        fmax=SPECTROGRAM_CONFIG['fmax'],
        window=SPECTROGRAM_CONFIG['window'],
        power=SPECTROGRAM_CONFIG['power']
    )
    
    # Convert sang log scale (dB)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Padding hoặc truncating để có kích thước cố định
    current_frames = log_mel_spec.shape[1]
    
    if current_frames < max_frames:
        # Padding với giá trị nhỏ nhất (silence)
        pad_width = max_frames - current_frames
        log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode='constant', constant_values=-80)
    elif current_frames > max_frames:
        # Truncating (cắt bớt)
        log_mel_spec = log_mel_spec[:, :max_frames]
    
    return log_mel_spec

def process_dataset(data_folder='data', output_folder='data_process', save_images=False):
    """
    Xử lý toàn bộ dataset, chuyển đổi tất cả audio thành spectrograms
    
    Parameters:
    -----------
    data_folder : str
        Thư mục chứa audio files
    output_folder : str
        Thư mục lưu spectrograms (.npy files)
    save_images : bool
        Có lưu ảnh visualization không
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Lấy danh sách tất cả các lệnh
    commands = [d for d in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, d))]
    
    all_spectrograms = []
    all_labels = []
    
    print(f"🔄 Đang xử lý {len(commands)} lệnh...")
    
    for command_idx, command in enumerate(commands):
        command_path = os.path.join(data_folder, command)
        audio_files = [f for f in os.listdir(command_path) if f.endswith('.wav')]
        
        print(f"\n📁 Xử lý lệnh '{command}': {len(audio_files)} files")
        
        # Tạo thư mục output cho lệnh này
        command_output = os.path.join(output_folder, command)
        if not os.path.exists(command_output):
            os.makedirs(command_output)
        
        for audio_file in tqdm(audio_files, desc=f"  {command}"):
            audio_path = os.path.join(command_path, audio_file)
            
            try:
                # Chuyển đổi thành spectrogram
                log_mel_spec = audio_to_log_mel_spectrogram(audio_path)
                
                # Lưu spectrogram dưới dạng numpy array
                spec_filename = audio_file.replace('.wav', '.npy')
                spec_path = os.path.join(command_output, spec_filename)
                np.save(spec_path, log_mel_spec)
                
                # Thêm vào dataset
                all_spectrograms.append(log_mel_spec)
                all_labels.append(command_idx)
                
                # Lưu ảnh visualization nếu cần
                if save_images:
                    img_filename = audio_file.replace('.wav', '.png')
                    img_path = os.path.join(command_output, img_filename)
                    
                    plt.figure(figsize=(10, 4))
                    librosa.display.specshow(
                        log_mel_spec,
                        sr=SAMPLE_RATE,
                        hop_length=SPECTROGRAM_CONFIG['hop_length'],
                        x_axis='time',
                        y_axis='mel',
                        fmin=SPECTROGRAM_CONFIG['fmin'],
                        fmax=SPECTROGRAM_CONFIG['fmax'],
                        cmap='viridis'
                    )
                    plt.colorbar(format='%+2.0f dB')
                    plt.title(f'{command}: {audio_file}')
                    plt.tight_layout()
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close()
                    
            except Exception as e:
                print(f"  ❌ Lỗi xử lý {audio_file}: {str(e)}")
    
    # Lưu toàn bộ dataset
    print(f"\n💾 Lưu toàn bộ dataset...")
    np.save(os.path.join(output_folder, 'X_spectrograms.npy'), np.array(all_spectrograms))
    np.save(os.path.join(output_folder, 'y_labels.npy'), np.array(all_labels))
    
    # Lưu mapping của labels
    label_mapping = {idx: cmd for idx, cmd in enumerate(commands)}
    np.save(os.path.join(output_folder, 'label_mapping.npy'), label_mapping)
    
    print(f"\n✅ Hoàn thành!")
    print(f"📊 Tổng số spectrograms: {len(all_spectrograms)}")
    print(f"📊 Shape của mỗi spectrogram: {all_spectrograms[0].shape}")
    print(f"📁 Đã lưu vào: {output_folder}")
    
    return all_spectrograms, all_labels, label_mapping

def save_sample_images(data_folder='data', output_folder='data_visualization', num_samples=10):
    """
    Lưu N samples đầu tiên của mỗi lệnh dưới dạng PNG để xem bằng mắt
    
    Parameters:
    -----------
    data_folder : str
        Thư mục chứa audio files
    output_folder : str
        Thư mục lưu PNG images
    num_samples : int
        Số lượng samples mỗi lệnh (default: 10)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    commands = [d for d in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, d))]
    
    print(f"🖼️  Đang lưu {num_samples} sample images cho mỗi lệnh...")
    print(f"📁 Thư mục output: {output_folder}\n")
    
    total_saved = 0
    
    for command in commands:
        command_path = os.path.join(data_folder, command)
        audio_files = [f for f in os.listdir(command_path) if f.endswith('.wav')][:num_samples]
        
        # Tạo thư mục cho lệnh này
        command_output = os.path.join(output_folder, command)
        if not os.path.exists(command_output):
            os.makedirs(command_output)
        
        print(f"📌 {command}: Lưu {len(audio_files)}/{num_samples} samples...")
        
        for idx, audio_file in enumerate(audio_files, 1):
            audio_path = os.path.join(command_path, audio_file)
            
            try:
                # Convert thành spectrogram
                log_mel_spec = audio_to_log_mel_spectrogram(audio_path)
                
                # Tạo figure với kích thước đẹp
                plt.figure(figsize=(12, 4))
                librosa.display.specshow(
                    log_mel_spec,
                    sr=SAMPLE_RATE,
                    hop_length=SPECTROGRAM_CONFIG['hop_length'],
                    x_axis='time',
                    y_axis='mel',
                    fmin=SPECTROGRAM_CONFIG['fmin'],
                    fmax=SPECTROGRAM_CONFIG['fmax'],
                    cmap='viridis'
                )
                plt.colorbar(format='%+2.0f dB')
                plt.title(f'{command.replace("_", " ").upper()} - Sample {idx}', fontsize=14, fontweight='bold')
                plt.xlabel('Time (s)', fontsize=12)
                plt.ylabel('Mel Frequency', fontsize=12)
                
                # Lưu với tên đẹp
                img_filename = f'{command}_sample_{idx:02d}.png'
                img_path = os.path.join(command_output, img_filename)
                
                plt.tight_layout()
                plt.savefig(img_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                total_saved += 1
                
            except Exception as e:
                print(f"   ❌ Lỗi xử lý {audio_file}: {str(e)}")
    
    print(f"\n✅ Hoàn thành! Đã lưu {total_saved} ảnh PNG")
    print(f"📂 Xem ảnh tại: {os.path.abspath(output_folder)}")

def visualize_samples(data_folder='data', num_samples=3):
    """
    Hiển thị một số mẫu spectrograms từ mỗi lệnh
    """
    commands = [d for d in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, d))]
    
    fig, axes = plt.subplots(len(commands), num_samples, figsize=(15, 3*len(commands)))
    
    for cmd_idx, command in enumerate(commands):
        command_path = os.path.join(data_folder, command)
        audio_files = [f for f in os.listdir(command_path) if f.endswith('.wav')][:num_samples]
        
        for sample_idx, audio_file in enumerate(audio_files):
            audio_path = os.path.join(command_path, audio_file)
            log_mel_spec = audio_to_log_mel_spectrogram(audio_path)
            
            ax = axes[cmd_idx, sample_idx] if len(commands) > 1 else axes[sample_idx]
            
            librosa.display.specshow(
                log_mel_spec,
                sr=SAMPLE_RATE,
                hop_length=SPECTROGRAM_CONFIG['hop_length'],
                ax=ax,
                x_axis='time',
                y_axis='mel',
                fmin=SPECTROGRAM_CONFIG['fmin'],
                fmax=SPECTROGRAM_CONFIG['fmax'],
                cmap='viridis'
            )
            
            if sample_idx == 0:
                ax.set_ylabel(command.replace('_', ' ').upper(), fontsize=10)
            else:
                ax.set_ylabel('')
            
            if cmd_idx == 0:
                ax.set_title(f'Sample {sample_idx + 1}', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('spectrogram_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Đã lưu visualization vào 'spectrogram_samples.png'")

if __name__ == "__main__":
    print("=" * 80)
    print("CHUYỂN ĐỔI AUDIO SANG LOG-MEL SPECTROGRAM")
    print("=" * 80)
    
    # Kiểm tra thư mục data có tồn tại không
    if not os.path.exists('data'):
        print("❌ Không tìm thấy thư mục 'data'. Vui lòng thu âm trước!")
        exit()
    
    print("\n📊 Cấu hình Spectrogram:")
    print(f"   - Sample Rate: {SAMPLE_RATE} Hz")
    print(f"   - N_FFT: {SPECTROGRAM_CONFIG['n_fft']}")
    print(f"   - Hop Length: {SPECTROGRAM_CONFIG['hop_length']}")
    print(f"   - N_Mels: {SPECTROGRAM_CONFIG['n_mels']}")
    print(f"   - Freq Range: {SPECTROGRAM_CONFIG['fmin']}-{SPECTROGRAM_CONFIG['fmax']} Hz")
    
    print("\n" + "=" * 80)
    print("BẮT ĐẦU XỬLÝ")
    print("=" * 80)
    
    # 1. Chuyển đổi toàn bộ dataset sang .npy
    print("\n[1/2] Chuyển đổi toàn bộ dataset sang .npy → data_process/")
    print("-" * 80)
    process_dataset(save_images=False)
    
    # 2. Lưu 10 sample PNG cho mỗi lệnh
    print("\n" + "=" * 80)
    print("[2/2] Lưu 10 sample PNG cho mỗi lệnh → data_visualization/")
    print("-" * 80)
    save_sample_images(num_samples=10)
    
    print("\n" + "=" * 80)
    print("✅ HOÀN THÀNH TẤT CẢ!")
    print("=" * 80)
    print(f"📁 Data .npy: {os.path.abspath('data_process')}")
    print(f"🖼️  Ảnh PNG: {os.path.abspath('data_visualization')}")
    print("=" * 80)
