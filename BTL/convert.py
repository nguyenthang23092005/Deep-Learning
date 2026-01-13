import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import random
from pathlib import Path

# Cấu hình
INPUT_DIR = "data"
OUTPUT_DIR = "data_GK"
VISUALIZATION_DIR = "visual"

# Các lệnh cần lấy
SELECTED_COMMANDS = [
    'tang_nhiet_do',
    'giam_nhiet_do',
    'mo_cua',
    'dong_cua',
    'bat_thong_bao_chay',
    'tat_thong_bao_chay'
]

# Tham số xử lý audio
SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 256  # Giảm để có temporal resolution tốt hơn
N_MELS = 256  # Tăng để capture chi tiết frequency tốt hơn
MAX_LENGTH = 256  # Tăng số frame để capture đủ thông tin

def create_directories():
    """Tạo các thư mục output nếu chưa tồn tại"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    
    for command in SELECTED_COMMANDS:
        os.makedirs(os.path.join(OUTPUT_DIR, command), exist_ok=True)
        os.makedirs(os.path.join(VISUALIZATION_DIR, command), exist_ok=True)

def load_and_preprocess_audio(file_path, focus_start=True, max_duration=1.5):
    """
    Load file audio và tiền xử lý - FOCUS VÀO PHẦN ĐẦU CÂU
    
    Arguments:
    file_path -- đường dẫn đến file audio
    focus_start -- nếu True, chỉ lấy phần đầu câu (từ khác nhau)
    max_duration -- thời gian tối đa giữ lại (giây)
    
    Returns:
    y -- audio signal
    sr -- sample rate
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Trim silence từ đầu và cuối
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # FOCUS VÀO PHẦN ĐẦU: Chỉ lấy 1.5s đầu (nơi có sự khác biệt lớn nhất)
        if focus_start:
            max_samples = int(max_duration * sr)
            if len(y) > max_samples:
                y = y[:max_samples]
        
        return y, sr
    except Exception as e:
        print(f"Lỗi khi load file {file_path}: {e}")
        return None, None

def augment_audio(y, sr, aug_type='original'):
    """
    Data augmentation để tăng diversity
    
    Arguments:
    y -- audio signal
    sr -- sample rate
    aug_type -- loại augmentation: 'original', 'pitch', 'speed', 'noise'
    
    Returns:
    y_aug -- augmented audio
    """
    if aug_type == 'original':
        return y
    
    elif aug_type == 'pitch':  # Pitch shifting
        # Shift pitch lên/xuống 2 semitones
        n_steps = np.random.choice([-2, -1, 1, 2])
        y_aug = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
        return y_aug
    
    elif aug_type == 'speed':  # Time stretching
        # Tăng/giảm tốc độ 10%
        rate = np.random.uniform(0.9, 1.1)
        y_aug = librosa.effects.time_stretch(y, rate=rate)
        return y_aug
    
    elif aug_type == 'noise':  # Add white noise
        # Thêm noise nhẹ
        noise = np.random.normal(0, 0.005, len(y))
        y_aug = y + noise
        return y_aug
    
    return y

def audio_to_log_spectrogram(y, sr):
    """
    Chuyển audio signal thành log-mel spectrogram
    
    Arguments:
    y -- audio signal
    sr -- sample rate
    
    Returns:
    log_spectrogram -- log-mel spectrogram, shape (N_MELS, MAX_LENGTH)
    """
    # Tạo mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    
    # Chuyển sang log scale (dB)
    log_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Chuẩn hóa về kích thước cố định
    if log_spectrogram.shape[1] < MAX_LENGTH:
        # Pad với giá trị min (silence)
        pad_width = MAX_LENGTH - log_spectrogram.shape[1]
        log_spectrogram = np.pad(log_spectrogram, ((0, 0), (0, pad_width)), 
                                mode='constant', constant_values=log_spectrogram.min())
    else:
        # Truncate nếu quá dài
        log_spectrogram = log_spectrogram[:, :MAX_LENGTH]
    
    return log_spectrogram

def save_spectrogram_visualization(spectrogram, output_path, title):
    """
    Lưu ảnh visualization của spectrogram
    
    Arguments:
    spectrogram -- log-mel spectrogram
    output_path -- đường dẫn lưu ảnh
    title -- tiêu đề của ảnh
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        spectrogram,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis='time',
        y_axis='mel',
        cmap='viridis'
    )
    plt.colorbar(format='%+2.0f')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def process_unknown_folder(max_samples=350):
    """
    Tạo folder 'unknown' từ các lệnh không được sử dụng và noise
    
    Arguments:
    max_samples -- số lượng mẫu tối đa (mặc định: 350)
    
    Returns:
    count -- số lượng file đã xử lý
    """
    print(f"\n📁 Tạo folder UNKNOWN từ các lệnh không sử dụng...")
    
    output_folder = os.path.join(OUTPUT_DIR, 'unknown')
    viz_folder = os.path.join(VISUALIZATION_DIR, 'unknown')
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(viz_folder, exist_ok=True)
    
    # Lấy tất cả các folder trong INPUT_DIR
    all_folders = [f for f in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, f))]
    
    # Tìm các folder không nằm trong SELECTED_COMMANDS (bao gồm cả noise)
    unused_commands = [f for f in all_folders if f not in SELECTED_COMMANDS]
    
    if not unused_commands:
        print("⚠️  Không tìm thấy lệnh nào không được sử dụng")
        return 0
    
    print(f"📋 Tìm thấy {len(unused_commands)} lệnh không sử dụng: {', '.join(unused_commands)}")
    
    # Thu thập tất cả file audio từ các lệnh không dùng
    all_audio_files = []
    for command in unused_commands:
        command_folder = os.path.join(INPUT_DIR, command)
        for ext in ['*.wav', '*.mp3', '*.m4a', '*.flac']:
            audio_files = list(Path(command_folder).glob(ext))
            all_audio_files.extend([(command, f) for f in audio_files])
    
    print(f"📊 Tổng số files có sẵn: {len(all_audio_files)}")
    
    # Random chọn max_samples files
    if len(all_audio_files) > max_samples:
        print(f"🎲 Random chọn {max_samples} files từ {len(all_audio_files)} files")
        selected_files = random.sample(all_audio_files, max_samples)
    else:
        print(f"⚠️  Chỉ có {len(all_audio_files)} files, lấy tất cả")
        selected_files = all_audio_files
    
    # Chọn 10 indices để visualization
    num_viz = min(10, len(selected_files))
    viz_indices = set(random.sample(range(len(selected_files)), num_viz))
    
    success_count = 0
    for idx, (command, audio_file) in enumerate(selected_files):
        # Load và preprocess audio
        y, sr = load_and_preprocess_audio(str(audio_file), focus_start=True, max_duration=1.5)
        
        if y is None:
            continue
        
        # Chuyển sang log-spectrogram
        log_spec = audio_to_log_spectrogram(y, sr)
        
        # Lưu với tên file có prefix command gốc
        output_filename = f"{command}_{audio_file.stem}.npy"
        output_path = os.path.join(output_folder, output_filename)
        np.save(output_path, log_spec)
        
        # Visualization cho 10 samples
        if idx in viz_indices:
            viz_filename = f"{command}_{audio_file.stem}.png"
            viz_path = os.path.join(viz_folder, viz_filename)
            save_spectrogram_visualization(
                log_spec,
                viz_path,
                f"UNKNOWN - {command} - Sample {idx+1}"
            )
        
        success_count += 1
        
        if (idx + 1) % 50 == 0:
            print(f"  ✓ Đã xử lý {idx + 1}/{len(selected_files)} files")
    
    print(f"✅ Hoàn thành UNKNOWN: {success_count}/{len(selected_files)} files")
    return success_count

def process_command(command):
    """
    Xử lý tất cả file audio của một lệnh
    
    Arguments:
    command -- tên lệnh (folder name)
    
    Returns:
    count -- số lượng file đã xử lý thành công
    """
    input_folder = os.path.join(INPUT_DIR, command)
    output_folder = os.path.join(OUTPUT_DIR, command)
    viz_folder = os.path.join(VISUALIZATION_DIR, command)
    
    # Đảm bảo thư mục output tồn tại
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(viz_folder, exist_ok=True)
    
    if not os.path.exists(input_folder):
        print(f"❌ Không tìm thấy folder: {input_folder}")
        return 0
    
    # Lấy danh sách file audio
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.m4a', '*.flac']:
        audio_files.extend(list(Path(input_folder).glob(ext)))
    
    if len(audio_files) == 0:
        print(f"⚠️  Không có file audio trong {command}")
        return 0
    
    print(f"\n📁 Đang xử lý: {command} ({len(audio_files)} files)")
    
    # Chọn ngẫu nhiên 10 indices để visualization
    num_viz = min(10, len(audio_files))
    selected_indices = set(random.sample(range(len(audio_files)), num_viz))
    
    success_count = 0
    
    for idx, audio_file in enumerate(audio_files):
        # Load và preprocess audio (FOCUS VÀO PHẦN ĐẦU)
        y, sr = load_and_preprocess_audio(str(audio_file), focus_start=True, max_duration=1.5)
        
        if y is None:
            continue
        
        # Chỉ dùng original (không augment) để so sánh rõ ràng
        y_aug = augment_audio(y, sr, aug_type='original')
        
        # Chuyển sang log-spectrogram
        log_spec = audio_to_log_spectrogram(y_aug, sr)
        
        # Lưu spectrogram dạng numpy
        output_filename = f"{audio_file.stem}.npy"
        output_path = os.path.join(output_folder, output_filename)
        np.save(output_path, log_spec)
        
        # Lưu 10 sample ngẫu nhiên dưới dạng ảnh visualization
        if idx in selected_indices:
            viz_filename = f"{audio_file.stem}.png"
            viz_path = os.path.join(viz_folder, viz_filename)
            save_spectrogram_visualization(
                log_spec,
                viz_path,
                f"{command} - Sample {idx+1} (Improved)"
            )
        
        success_count += 1
        
        if (idx + 1) % 10 == 0:
            print(f"  ✓ Đã xử lý {idx + 1}/{len(audio_files)} files")
    
    print(f"✅ Hoàn thành {command}: {success_count}/{len(audio_files)} files")
    return success_count

def main():
    """Hàm chính để chạy toàn bộ pipeline"""
    print("="*60)
    print("CHUYỂN ĐỔI AUDIO THÀNH LOG-MEL SPECTROGRAM")
    print("="*60)
    print(f"\n📂 Input folder: {INPUT_DIR}")
    print(f"📂 Output folder: {OUTPUT_DIR}")
    print(f"📂 Visualization folder: {VISUALIZATION_DIR}")
    print(f"\n🎯 Các lệnh được chọn:")
    for i, cmd in enumerate(SELECTED_COMMANDS, 1):
        print(f"   {i}. {cmd}")
    print(f"\n⚙️  Tham số:")
    print(f"   - Sample rate: {SAMPLE_RATE} Hz")
    print(f"   - N_MEL: {N_MELS}")
    print(f"   - Max length: {MAX_LENGTH} frames")
    print(f"   - Visualization: 10 samples/lệnh")
    
    # Tạo thư mục
    print("\n📁 Tạo thư mục...")
    create_directories()
    
    # Xử lý từng lệnh
    total_files = 0
    for command in SELECTED_COMMANDS:
        count = process_command(command)
        total_files += count
    
    # Xử lý folder UNKNOWN
    print("\n" + "="*60)
    print("TẠO FOLDER UNKNOWN")
    print("="*60)
    unknown_count = process_unknown_folder(max_samples=350)
    total_files += unknown_count
    
    print("\n" + "="*60)
    print(f"✅ HOÀN THÀNH!")
    print(f"📊 Tổng số file đã xử lý: {total_files}")
    print(f"   - Các lệnh chính: {total_files - unknown_count}")
    print(f"   - Unknown: {unknown_count}")
    print(f"📂 Dữ liệu đã lưu tại: {OUTPUT_DIR}")
    print(f"🖼️  Ảnh visualization tại: {VISUALIZATION_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
