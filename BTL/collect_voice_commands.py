import sounddevice as sd
import soundfile as sf
import numpy as np
import os
from datetime import datetime
import keyboard
import time
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy import signal

SAMPLE_RATE = 16000  
DURATION = 2.5  
CHANNELS = 1  
BIT_DEPTH = 'int16'  

SPECTROGRAM_CONFIG = {
    'n_fft': 512,           
    'hop_length': 256,    
    'n_mels': 64,           
    'fmin': 20,             
    'fmax': 8000,           
    'window': 'hann',       
    'power': 2.0,           
}


COMMANDS = {
    '1': 'bat_den',
    '2': 'tat_den',
    '3': 'bat_quat',
    '4': 'tat_quat',
    '5': 'bat_dieu_hoa',
    '6': 'tat_dieu_hoa',
    '7': 'mo_cua',
    '8': 'dong_cua',
    '9': 'bat_thong_bao_chay',
    '0': 'tat_thong_bao_chay',
    'q': 'bat_tat_ca',
    'w': 'tat_tat_ca',
    'e': 'tang_nhiet_do',
    'r': 'giam_nhiet_do',
    'n': 'noise',  
}

class VoiceCommandCollector:
    def __init__(self, data_folder='data'):
        self.data_folder = data_folder
        self.create_folders()
        self.is_recording = False
        
    def create_folders(self):
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
        
        for command in COMMANDS.values():
            command_folder = os.path.join(self.data_folder, command)
            if not os.path.exists(command_folder):
                os.makedirs(command_folder)
                print(f"Đã tạo thư mục: {command_folder}")
    
    def record_audio(self, command_name):
        if self.is_recording:
            print("Đang thu âm, vui lòng đợi...")
            return
        
        self.is_recording = True
        print(f"\n🎤 Đang thu âm lệnh '{command_name}' trong {DURATION} giây...")
        
        try:
            audio_data = sd.rec(int(DURATION * SAMPLE_RATE), 
                              samplerate=SAMPLE_RATE, 
                              channels=CHANNELS, 
                              dtype='float32')  
            sd.wait()  
            
            audio_data = audio_data / np.max(np.abs(audio_data) + 1e-10)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{command_name}_{timestamp}.wav"
            filepath = os.path.join(self.data_folder, command_name, filename)
            
            sf.write(filepath, audio_data, SAMPLE_RATE, subtype='PCM_16') 
            
            file_count = len([f for f in os.listdir(os.path.join(self.data_folder, command_name)) 
                            if f.endswith('.wav')])
            
            print(f"✅ Đã lưu: {filepath}")
            print(f"📊 Tổng số mẫu của '{command_name}': {file_count}")
            
        except Exception as e:
            print(f"❌ Lỗi khi thu âm: {str(e)}")
        
        finally:
            self.is_recording = False
    
    def preview_spectrogram(self, audio_file):
        """Hiển thị preview Log-Mel Spectrogram của một file"""
        try:
            audio, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
            
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
            
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(
                log_mel_spec,
                sr=sr,
                hop_length=SPECTROGRAM_CONFIG['hop_length'],
                x_axis='time',
                y_axis='mel',
                fmin=SPECTROGRAM_CONFIG['fmin'],
                fmax=SPECTROGRAM_CONFIG['fmax'],
                cmap='viridis'
            )
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Log-Mel Spectrogram: {os.path.basename(audio_file)}')
            plt.tight_layout()
            plt.show()
            
            print(f"📊 Shape: {log_mel_spec.shape} (n_mels={log_mel_spec.shape[0]}, frames={log_mel_spec.shape[1]})")
            
        except Exception as e:
            print(f"❌ Lỗi khi tạo spectrogram: {str(e)}")
    
    def print_instructions(self):
        print("=" * 80)
        print("HỆ THỐNG THU THẬP LỆNH GIỌNG NÓI CHO NHÀ THÔNG MINH")
        print("=" * 80)
        print("\n📋 DANH SÁCH LỆNH VÀ PHÍM TƯƠNG ỨNG:\n")
        
        # Nhóm lệnh theo chức năng
        groups = {
            "ĐÈN": ['1', '2'],
            "QUẠT": ['3', '4'],
            "ĐIỀU HÒA": ['5', '6', 'e', 'r'],
            "CỬA": ['7', '8'],
            "AN NINH": ['9', '0'],
            "TỔNG QUÁT": ['q', 'w'],
            "NHIỄU": ['n'],
        }
        
        for group_name, keys in groups.items():
            print(f"\n🏠 {group_name}:")
            for key in keys:
                if key in COMMANDS:
                    command_name = COMMANDS[key].replace('_', ' ').upper()
                    print(f"   [{key}] - {command_name}")
        
        print("\n" + "=" * 80)
        print("⚠️  HƯỚNG DẪN:")
        print(f"   - Bấm phím tương ứng để bắt đầu thu âm ({DURATION} giây)")
        print("   - Nói rõ ràng vào micro sau khi bấm phím")
        print("   - Bấm [ESC] để thoát chương trình")
        print("   - Bấm [SPACE] để xem preview spectrogram của file mới nhất")
        print("   - Thu ít nhất 50-100 mẫu cho mỗi lệnh để có kết quả tốt")
        print("\n📊 CẤU HÌNH LOG-SPECTROGRAM:")
        print(f"   - Sample Rate: {SAMPLE_RATE} Hz")
        print(f"   - FFT Size: {SPECTROGRAM_CONFIG['n_fft']}")
        print(f"   - Hop Length: {SPECTROGRAM_CONFIG['hop_length']}")
        print(f"   - Mel Bands: {SPECTROGRAM_CONFIG['n_mels']}")
        print(f"   - Frequency Range: {SPECTROGRAM_CONFIG['fmin']}-{SPECTROGRAM_CONFIG['fmax']} Hz")
        print("=" * 80 + "\n")
    
    def start(self):
        """Bắt đầu chương trình thu thập"""
        self.print_instructions()
        
        print("✅ Chương trình đã sẵn sàng! Bấm phím để bắt đầu thu âm...\n")
        
        last_recorded_file = None
        
        try:
            while True:
                if keyboard.is_pressed('esc'):
                    print("\n👋 Đã thoát chương trình!")
                    break
                
                if keyboard.is_pressed('space') and last_recorded_file:
                    print("\n📊 Đang tạo spectrogram preview...")
                    self.preview_spectrogram(last_recorded_file)
                    time.sleep(0.5)
                
                for key, command_name in COMMANDS.items():
                    if keyboard.is_pressed(key) and not self.is_recording:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = f"{command_name}_{timestamp}.wav"
                        last_recorded_file = os.path.join(self.data_folder, command_name, filename)
                        
                        self.record_audio(command_name)
                        time.sleep(0.3)  
                
                time.sleep(0.1)  
                
        except KeyboardInterrupt:
            print("\n👋 Đã thoát chương trình!")
    
    def show_statistics(self):
        print("\n📊 THỐNG KÊ SỐ LƯỢNG MẪU:")
        print("=" * 60)
        
        total = 0
        for command_name in COMMANDS.values():
            command_folder = os.path.join(self.data_folder, command_name)
            if os.path.exists(command_folder):
                count = len([f for f in os.listdir(command_folder) if f.endswith('.wav')])
                total += count
                status = "✅" if count >= 50 else "⚠️" if count >= 20 else "❌"
                print(f"{status} {command_name.replace('_', ' ').ljust(25)}: {count} mẫu")
        
        print("=" * 60)
        print(f"📈 TỔNG CỘNG: {total} mẫu")
        print("=" * 60)

if __name__ == "__main__":
    collector = VoiceCommandCollector(data_folder='data')
    
    collector.show_statistics()
    
    collector.start()
    
    collector.show_statistics()
