import numpy as np
import sounddevice as sd
import librosa
import os
from datetime import datetime
from model_cnn import cnn_forward
import time

# =====================================================================
# CẤU HÌNH
# =====================================================================

# Audio config
SAMPLE_RATE = 16000
DURATION = 2.0  # Độ dài lệnh voice command (giây)
WAKE_WORD_DURATION = 1.5  # Độ dài wake word "hey siri"
CHANNELS = 1

# Spectrogram config (giống với training)
SPECTROGRAM_CONFIG = {
    'n_fft': 512,
    'hop_length': 256,
    'n_mels': 64,
    'fmin': 20,
    'fmax': 8000,
    'window': 'hann',
    'power': 2.0,
}
MAX_TIME_FRAMES = 128

# Model config
CHECKPOINT_DIR = 'checkpoint_cnn'
MODEL_RUN = None  # Sẽ tự động chọn run mới nhất
CHECKPOINT_FILE = 'final_model_cnn.npz'  # Sử dụng final model

# Wake word detection config
WAKE_WORD_THRESHOLD = 0.3  # Ngưỡng năng lượng để phát hiện "hey siri"
SILENCE_THRESHOLD = 0.02  # Ngưỡng silence

# =====================================================================
# AUDIO PROCESSING FUNCTIONS
# =====================================================================

def audio_to_log_mel_spectrogram(audio, sr=SAMPLE_RATE, max_frames=MAX_TIME_FRAMES):
    """
    Chuyển đổi audio array thành Log-Mel Spectrogram
    
    Parameters:
    -----------
    audio : numpy.ndarray
        Audio signal
    sr : int
        Sample rate
    max_frames : int
        Số lượng time frames tối đa
        
    Returns:
    --------
    log_mel_spec : numpy.ndarray
        Log-Mel Spectrogram với shape (n_mels, max_frames)
    """
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
        log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), 
                               mode='constant', constant_values=-80)
    elif current_frames > max_frames:
        # Truncating
        log_mel_spec = log_mel_spec[:, :max_frames]
    
    return log_mel_spec

def detect_wake_word(audio, threshold=WAKE_WORD_THRESHOLD):
    """
    Phát hiện wake word "hey siri" dựa trên năng lượng âm thanh
    
    Parameters:
    -----------
    audio : numpy.ndarray
        Audio signal
    threshold : float
        Ngưỡng năng lượng để phát hiện
        
    Returns:
    --------
    bool : True nếu phát hiện wake word
    """
    # Tính năng lượng (RMS - Root Mean Square)
    energy = np.sqrt(np.mean(audio**2))
    
    # Kiểm tra xem có vượt ngưỡng không
    return energy > threshold

def record_audio(duration=DURATION, sample_rate=SAMPLE_RATE):
    """
    Thu âm từ microphone
    
    Parameters:
    -----------
    duration : float
        Thời gian ghi âm (giây)
    sample_rate : int
        Sample rate
        
    Returns:
    --------
    audio : numpy.ndarray
        Audio signal
    """
    print(f"🎤 Đang ghi âm trong {duration}s...")
    audio = sd.rec(int(duration * sample_rate), 
                   samplerate=sample_rate, 
                   channels=CHANNELS,
                   dtype='float32')
    sd.wait()
    print("✅ Hoàn tất ghi âm")
    
    return audio.flatten()

# =====================================================================
# MODEL FUNCTIONS
# =====================================================================

def load_model(checkpoint_path):
    """
    Load model parameters từ checkpoint
    
    Parameters:
    -----------
    checkpoint_path : str
        Đường dẫn đến checkpoint file
        
    Returns:
    --------
    parameters : dict
        Model parameters
    """
    print(f"📥 Đang load model từ: {checkpoint_path}")
    data = np.load(checkpoint_path)
    parameters = {}
    for key in data.files:
        parameters[key] = data[key]
    print("✅ Đã load model thành công")
    return parameters

def load_label_mapping(run_dir):
    """
    Load label mapping từ run directory
    
    Parameters:
    -----------
    run_dir : str
        Đường dẫn đến run directory
        
    Returns:
    --------
    label_mapping : dict
        Mapping từ class index -> command name
    """
    label_mapping_path = os.path.join(run_dir, 'label_mapping.npy')
    if os.path.exists(label_mapping_path):
        label_mapping = np.load(label_mapping_path, allow_pickle=True).item()
        print(f"✅ Đã load label mapping: {len(label_mapping)} lệnh")
        return label_mapping
    else:
        print("⚠️ Không tìm thấy label_mapping.npy, sử dụng mapping mặc định")
        # Fallback: tạo mapping mặc định từ data_GK folder
        data_dir = 'D:/GitHub/Deep-Learning/BTL/data_GK'
        if os.path.exists(data_dir):
            command_folders = sorted([f for f in os.listdir(data_dir) 
                                    if os.path.isdir(os.path.join(data_dir, f))])
            label_mapping = {idx: cmd for idx, cmd in enumerate(command_folders)}
            return label_mapping
        else:
            raise ValueError("Không tìm thấy label mapping!")

def find_latest_run(checkpoint_dir):
    """
    Tìm run directory mới nhất
    
    Parameters:
    -----------
    checkpoint_dir : str
        Đường dẫn đến checkpoint directory
        
    Returns:
    --------
    str : Tên của run directory mới nhất
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    runs = [d for d in os.listdir(checkpoint_dir) 
            if os.path.isdir(os.path.join(checkpoint_dir, d)) and d.startswith('run_')]
    
    if not runs:
        return None
    
    # Sắp xếp theo thời gian (dựa trên tên)
    runs.sort(reverse=True)
    return runs[0]

def predict_command(audio, parameters, label_mapping):
    """
    Dự đoán lệnh từ audio
    
    Parameters:
    -----------
    audio : numpy.ndarray
        Audio signal
    parameters : dict
        Model parameters
    label_mapping : dict
        Label mapping
        
    Returns:
    --------
    command : str
        Tên lệnh được dự đoán
    confidence : float
        Độ tin cậy (probability)
    """
    # Chuyển audio thành spectrogram
    spectrogram = audio_to_log_mel_spectrogram(audio)
    
    # Reshape để phù hợp với input của CNN model
    # CNN expects: (N, C, H, W) where H=128, W=64
    # Resize spectrogram về 128x64 (giống với training)
    import cv2
    spectrogram_resized = cv2.resize(spectrogram, (64, 128), interpolation=cv2.INTER_AREA)
    
    # Reshape: (1, 1, 128, 64) - batch_size=1, channels=1
    X = spectrogram_resized.reshape(1, 1, 128, 64)
    
    # Forward pass
    Z, _ = cnn_forward(X, parameters, training=False)
    
    # Softmax để tính probability
    Z_shift = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_shift)
    probs = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    # Lấy prediction
    pred_idx = np.argmax(probs, axis=1)[0]
    confidence = probs[0, pred_idx]
    
    command = label_mapping.get(pred_idx, f"Unknown_{pred_idx}")
    
    return command, confidence

# =====================================================================
# MAIN TESTING LOOP
# =====================================================================

def continuous_voice_command_recognition():
    """
    Chạy continuous voice command recognition với wake word detection
    """
    print("=" * 70)
    print("🎯 VOICE COMMAND RECOGNITION SYSTEM (CNN)")
    print("=" * 70)
    
    # Tìm run mới nhất
    run_name = MODEL_RUN or find_latest_run(CHECKPOINT_DIR)
    
    if not run_name:
        print(f"❌ Không tìm thấy run nào trong {CHECKPOINT_DIR}")
        return
    
    run_dir = os.path.join(CHECKPOINT_DIR, run_name)
    checkpoint_path = os.path.join(run_dir, CHECKPOINT_FILE)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Không tìm thấy checkpoint: {checkpoint_path}")
        print("\n📋 Available checkpoints:")
        if os.path.exists(run_dir):
            for f in os.listdir(run_dir):
                if f.endswith('.npz'):
                    print(f"   - {f}")
        return
    
    # Load model
    parameters = load_model(checkpoint_path)
    label_mapping = load_label_mapping(run_dir)
    
    print(f"\n📁 Sử dụng run: {run_name}")
    print("\n📋 Các lệnh có thể nhận diện:")
    for idx, cmd in sorted(label_mapping.items()):
        print(f"   {idx}: {cmd}")
    
    print("\n" + "=" * 70)
    print("🎙️  BẮT ĐẦU NHẬN DIỆN GIỌNG NÓI")
    print("=" * 70)
    print("💡 Hướng dẫn:")
    print("   1. Nói 'Hey Siri' để kích hoạt")
    print("   2. Sau khi nghe beep, nói lệnh trong 2 giây")
    print("   3. Nhấn Ctrl+C để thoát")
    print("=" * 70)
    
    try:
        while True:
            print("\n👂 Đang lắng nghe 'Hey Siri'...")
            
            # Thu âm liên tục để phát hiện wake word
            wake_audio = record_audio(duration=WAKE_WORD_DURATION, sample_rate=SAMPLE_RATE)
            
            # Kiểm tra wake word
            if detect_wake_word(wake_audio):
                print("🔔 Đã phát hiện wake word! Sẵn sàng nhận lệnh...")
                
                # Phát beep sound (optional)
                try:
                    # Tạo beep sound ngắn
                    beep_duration = 0.1
                    beep_freq = 800
                    t = np.linspace(0, beep_duration, int(SAMPLE_RATE * beep_duration))
                    beep = 0.3 * np.sin(2 * np.pi * beep_freq * t)
                    sd.play(beep, SAMPLE_RATE)
                    sd.wait()
                except:
                    pass
                
                # Thu âm lệnh
                command_audio = record_audio(duration=DURATION, sample_rate=SAMPLE_RATE)
                
                # Dự đoán lệnh
                print("🔍 Đang xử lý...")
                command, confidence = predict_command(command_audio, parameters, label_mapping)
                
                # Hiển thị kết quả
                print("\n" + "=" * 70)
                print(f"✨ LỆNH: {command}")
                print(f"📊 Độ tin cậy: {confidence*100:.2f}%")
                print("=" * 70)
                
                # Thêm delay trước khi lắng nghe tiếp
                time.sleep(1)
            else:
                print("   (Chưa nghe thấy 'Hey Siri', thử lại...)")
                time.sleep(0.5)
                
    except KeyboardInterrupt:
        print("\n\n👋 Đã dừng chương trình")
        print("=" * 70)

def single_test():
    """
    Test một lần duy nhất (không cần wake word)
    """
    print("=" * 70)
    print("🎯 VOICE COMMAND RECOGNITION - SINGLE TEST (CNN)")
    print("=" * 70)
    
    # Tìm run mới nhất
    run_name = MODEL_RUN or find_latest_run(CHECKPOINT_DIR)
    
    if not run_name:
        print(f"❌ Không tìm thấy run nào trong {CHECKPOINT_DIR}")
        return
    
    run_dir = os.path.join(CHECKPOINT_DIR, run_name)
    checkpoint_path = os.path.join(run_dir, CHECKPOINT_FILE)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Không tìm thấy checkpoint: {checkpoint_path}")
        return
    
    # Load model
    parameters = load_model(checkpoint_path)
    label_mapping = load_label_mapping(run_dir)
    
    print(f"\n📁 Sử dụng run: {run_name}")
    print("\n📋 Các lệnh có thể nhận diện:")
    for idx, cmd in sorted(label_mapping.items()):
        print(f"   {idx}: {cmd}")
    
    print("\n" + "=" * 70)
    
    # Thu âm
    command_audio = record_audio(duration=DURATION, sample_rate=SAMPLE_RATE)
    
    # Dự đoán
    print("🔍 Đang xử lý...")
    command, confidence = predict_command(command_audio, parameters, label_mapping)
    
    # Hiển thị kết quả
    print("\n" + "=" * 70)
    print(f"✨ LỆNH: {command}")
    print(f"📊 Độ tin cậy: {confidence*100:.2f}%")
    print("=" * 70)

# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    import sys
    
    print("\n🎤 Voice Command Recognition System (CNN)")
    print("=" * 70)
    print("Chọn chế độ:")
    print("  1. Continuous mode (với wake word 'Hey Siri')")
    print("  2. Single test (test 1 lần, không cần wake word)")
    print("=" * 70)
    
    choice = input("\nNhập lựa chọn (1/2): ").strip()
    
    if choice == "1":
        continuous_voice_command_recognition()
    elif choice == "2":
        single_test()
    else:
        print("❌ Lựa chọn không hợp lệ!")
