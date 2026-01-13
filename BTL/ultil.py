import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_data_from_folders(data_dir='data_GK', test_size=0.2, random_state=42):
    data_dir = os.path.normpath(data_dir)
    
    print(f"📂 Đang load dữ liệu từ: {data_dir}")
    
    command_folders = [f for f in os.listdir(data_dir) 
                      if os.path.isdir(os.path.join(data_dir, f))]
    
    if len(command_folders) == 0:
        raise ValueError(f"Không tìm thấy folder lệnh nào trong {data_dir}")
    
    print(f"📋 Tìm thấy {len(command_folders)} lệnh:")
    
    # Load tất cả spectrograms
    X_list = []
    Y_list = []
    label_mapping = {}
    
    for idx, command in enumerate(sorted(command_folders)):
        command_path = os.path.join(data_dir, command)
        npy_files = list(Path(command_path).glob('*.npy'))
        
        print(f"   {idx}: {command} - {len(npy_files)} samples")
        
        label_mapping[idx] = command
        
        for npy_file in npy_files:
            spec = np.load(str(npy_file))
            X_list.append(spec)
            Y_list.append(idx)
    
    # Chuyển thành numpy arrays
    X = np.array(X_list)
    Y = np.array(Y_list)
    
    print(f"\n📊 Thông tin dataset:")
    print(f"   Total samples: {X.shape[0]}")
    print(f"   Spectrogram shape: {X.shape[1:]}")
    print(f"   Số lệnh (classes): {len(label_mapping)}")
    
    # Reshape Y thành (1, n_samples)
    Y = Y.reshape(1, -1)
    
    # Shuffle và chia train/test
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y.T,
        test_size=test_size,
        random_state=random_state,
        stratify=Y.T,
        shuffle=True
    )
    
    # Reshape Y về dạng (1, n_samples)
    Y_train = Y_train.T
    Y_test = Y_test.T
    
    print(f"\n✅ Chia dữ liệu:")
    print(f"   Train set: {X_train.shape[0]} samples ({X_train.shape[0]/(X_train.shape[0]+X_test.shape[0])*100:.1f}%)")
    print(f"   Test set: {X_test.shape[0]} samples ({X_test.shape[0]/(X_train.shape[0]+X_test.shape[0])*100:.1f}%)")
    
    # Hiển thị phân bố class
    print(f"\n📊 Phân bố class trong Train set:")
    for class_idx in sorted(np.unique(Y_train)):
        count = np.sum(Y_train == class_idx)
        print(f"   Class {class_idx} ({label_mapping[class_idx]}): {count} samples")
    
    print(f"\n📊 Phân bố class trong Test set:")
    for class_idx in sorted(np.unique(Y_test)):
        count = np.sum(Y_test == class_idx)
        print(f"   Class {class_idx} ({label_mapping[class_idx]}): {count} samples")
    
    return X_train, Y_train, X_test, Y_test, label_mapping


def load_sample_data(data_dir='data_GK', samples_per_class=20, test_size=0.2, random_state=42):
    """
    Load MỘT SỐ MẪU từ mỗi lệnh để test nhanh
    
    Arguments:
    data_dir -- đường dẫn đến folder chứa data
    samples_per_class -- số mẫu lấy từ mỗi class (mặc định: 20)
    test_size -- tỷ lệ test set (mặc định: 0.2)
    random_state -- seed để shuffle
    
    Returns:
    X_train, Y_train, X_test, Y_test, label_mapping
    """
    data_dir = os.path.normpath(data_dir)
    
    print(f"📂 Đang load {samples_per_class} mẫu/lệnh từ: {data_dir}")
    
    # Lấy danh sách các folder (mỗi folder = 1 lệnh)
    command_folders = [f for f in os.listdir(data_dir) 
                      if os.path.isdir(os.path.join(data_dir, f))]
    
    if len(command_folders) == 0:
        raise ValueError(f"Không tìm thấy folder lệnh nào trong {data_dir}")
    
    print(f"📋 Tìm thấy {len(command_folders)} lệnh:")
    
    # Load samples
    X_list = []
    Y_list = []
    label_mapping = {}
    
    np.random.seed(random_state)
    
    for idx, command in enumerate(sorted(command_folders)):
        command_path = os.path.join(data_dir, command)
        npy_files = list(Path(command_path).glob('*.npy'))
        
        # Lấy ngẫu nhiên samples_per_class mẫu
        n_samples = min(samples_per_class, len(npy_files))
        selected_files = np.random.choice(npy_files, size=n_samples, replace=False)
        
        print(f"   {idx}: {command} - Lấy {n_samples}/{len(npy_files)} samples")
        
        label_mapping[idx] = command
        
        for npy_file in selected_files:
            spec = np.load(str(npy_file))
            X_list.append(spec)
            Y_list.append(idx)
    
    # Chuyển thành numpy arrays
    X = np.array(X_list)
    Y = np.array(Y_list)
    
    print(f"\n📊 Thông tin dataset (sample):")
    print(f"   Total samples: {X.shape[0]}")
    print(f"   Spectrogram shape: {X.shape[1:]}")
    print(f"   Số lệnh (classes): {len(label_mapping)}")
    
    # Reshape Y thành (1, n_samples)
    Y = Y.reshape(1, -1)
    
    # Shuffle và chia train/test
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y.T,
        test_size=test_size,
        random_state=random_state,
        stratify=Y.T,
        shuffle=True
    )
    
    # Reshape Y về dạng (1, n_samples)
    Y_train = Y_train.T
    Y_test = Y_test.T
    
    print(f"\n✅ Chia dữ liệu:")
    print(f"   Train set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    return X_train, Y_train, X_test, Y_test, label_mapping

