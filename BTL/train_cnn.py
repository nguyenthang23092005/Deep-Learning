import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
from ultil import load_data_from_folders, load_sample_data
from model_cnn import (
    init_cnn_params,
    cnn_forward,
    cnn_backward,
    softmax_cross_entropy
)

def save_checkpoint(parameters, iteration, checkpoint_dir='checkpoint_cnn'):
    """Lưu checkpoint của model"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_iter_{iteration}.npz')
    
    # Lưu tất cả parameters vào file .npz
    np.savez(checkpoint_path, **parameters)
    print(f"✅ Đã lưu checkpoint tại: {checkpoint_path}")

def load_checkpoint(checkpoint_path):
    """Load checkpoint từ file"""
    data = np.load(checkpoint_path)
    parameters = {}
    for key in data.files:
        parameters[key] = data[key]
    print(f"✅ Đã load checkpoint từ: {checkpoint_path}")
    return parameters

def update_parameters(params, grads, learning_rate):
    """Cập nhật parameters"""
    for key in params.keys():
        if 'd' + key in grads:
            params[key] -= learning_rate * grads['d' + key]
    return params

def predict_cnn(X, params):
    """
    Dự đoán với CNN
    X: (N, C, H, W)
    Returns: predictions
    """
    Z, _ = cnn_forward(X, params, training=False)
    preds = np.argmax(Z, axis=1)
    return preds

def train_cnn(X_train, Y_train, X_test, Y_test, label_mapping,
              learning_rate=0.001, num_iterations=3000,
              checkpoint_interval=500, checkpoint_dir='checkpoint_cnn'):
    """
    Training CNN model với checkpointing
    
    Arguments:
    X_train -- training data, shape (n_samples, height, width)
    Y_train -- training labels, shape (1, n_samples) hoặc (n_samples,)
    X_test -- test data, shape (n_samples, height, width)
    Y_test -- test labels, shape (1, n_samples) hoặc (n_samples,)
    label_mapping -- dict mapping từ class index -> command name
    learning_rate -- learning rate
    num_iterations -- số iterations
    checkpoint_interval -- lưu checkpoint sau mỗi bao nhiêu iterations
    checkpoint_dir -- folder lưu checkpoint
    
    Returns:
    params -- trained parameters
    costs -- list of costs during training
    """
    
    print("\n" + "="*70)
    print("🚀 BẮT ĐẦU TRAINING CNN")
    print("="*70)
    
    # Resize spectrograms về 128×64
    print(f"\n📏 Resize spectrograms từ {X_train.shape[1:]} về (128, 64)...")
    X_train_resized = np.array([cv2.resize(img, (64, 128), interpolation=cv2.INTER_AREA) 
                                for img in X_train])
    X_test_resized = np.array([cv2.resize(img, (64, 128), interpolation=cv2.INTER_AREA) 
                               for img in X_test])
    
    print(f"   X_train: {X_train.shape} → {X_train_resized.shape}")
    print(f"   X_test: {X_test.shape} → {X_test_resized.shape}")
    
    # Reshape X từ (n_samples, H, W) -> (n_samples, 1, H, W) cho CNN
    X_train_cnn = X_train_resized.reshape(X_train_resized.shape[0], 1, X_train_resized.shape[1], X_train_resized.shape[2])
    X_test_cnn = X_test_resized.reshape(X_test_resized.shape[0], 1, X_test_resized.shape[1], X_test_resized.shape[2])
    
    # Flatten Y labels
    Y_train_flat = Y_train.flatten().astype(int)
    Y_test_flat = Y_test.flatten().astype(int)
    
    print(f"\n📐 Shape sau khi reshape cho CNN:")
    print(f"   X_train: {X_train_cnn.shape} (N, C, H, W)")
    print(f"   Y_train: {Y_train_flat.shape}")
    print(f"   X_test: {X_test_cnn.shape} (N, C, H, W)")
    print(f"   Y_test: {Y_test_flat.shape}")
    
    # Initialize parameters
    num_classes = len(label_mapping)
    print(f"\n🔧 Khởi tạo CNN model:")
    print(f"   Architecture:")
    print(f"      Input: (N, 1, 128, 64)")
    print(f"      Conv1 (3x3) + BN + ReLU: 1 -> 16")
    print(f"      MaxPool (2x2): -> (N, 16, 64, 32)")
    print(f"      Conv2 (3x3) + BN + ReLU: 16 -> 32")
    print(f"      MaxPool (2x2): -> (N, 32, 32, 16)")
    print(f"      Depthwise Conv (3x3) + BN + ReLU: 32")
    print(f"      GlobalAvgPool: -> (N, 32)")
    print(f"      Dense1: 32 -> 64 + Dropout(0.5)")
    print(f"      Dense2: 64 -> {num_classes} + Softmax")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Iterations: {num_iterations}")
    print(f"   Checkpoint interval: {checkpoint_interval}")
    
    np.random.seed(42)
    params = init_cnn_params(num_classes=num_classes)
    
    costs = []
    
    # Tạo subfolder với timestamp cho mỗi lần train
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(checkpoint_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    checkpoint_dir = run_dir  # Sử dụng run_dir thay vì checkpoint_dir gốc
    
    print(f"📁 Checkpoint sẽ được lưu tại: {checkpoint_dir}")
    
    print(f"\n🏋️ Training...")
    print("-" * 70)
    
    # Training loop
    for i in range(num_iterations):
        # Forward propagation
        Z, caches = cnn_forward(X_train_cnn, params, training=True)
        
        # Compute cost and gradient
        cost, dZ = softmax_cross_entropy(Z, Y_train_flat)
        
        # Backward propagation
        grads = cnn_backward(dZ, caches, params)
        
        # Update parameters
        params = update_parameters(params, grads, learning_rate)
        
        # Log cost mỗi 100 iterations
        if i % 100 == 0:
            costs.append(cost)
            print(f"Iteration {i:5d} | Cost: {cost:.6f}")
        
        # Lưu checkpoint
        if (i > 0 and i % checkpoint_interval == 0) or i == num_iterations - 1:
            save_checkpoint(params, i, checkpoint_dir)
    
    print("-" * 70)
    print("✅ Hoàn thành training!\n")
    
    # Evaluate trên train set
    print("📊 ĐÁNH GIÁ TRÊN TRAIN SET:")
    print("-" * 70)
    train_preds = predict_cnn(X_train_cnn, params)
    train_acc = np.mean(train_preds == Y_train_flat)
    print(f"Accuracy: {train_acc:.6f}")
    
    # Evaluate trên test set
    print("\n📊 ĐÁNH GIÁ TRÊN TEST SET:")
    print("-" * 70)
    test_preds = predict_cnn(X_test_cnn, params)
    test_acc = np.mean(test_preds == Y_test_flat)
    print(f"Accuracy: {test_acc:.6f}")
    
    # Chi tiết accuracy cho từng class
    print("\n📋 Accuracy chi tiết cho từng lệnh (Test set):")
    print("-" * 70)
    for class_idx in sorted(np.unique(Y_test_flat)):
        mask = Y_test_flat == class_idx
        class_acc = np.mean(test_preds[mask] == Y_test_flat[mask])
        class_name = label_mapping[class_idx]
        total = np.sum(mask)
        correct = np.sum(test_preds[mask] == Y_test_flat[mask])
        print(f"   Class {class_idx} ({class_name:20s}): {class_acc*100:5.2f}% ({correct}/{total})")
    
    # Plot learning curve
    print("\n📈 Vẽ learning curve...")
    plt.figure(figsize=(12, 7))
    
    iterations = np.arange(0, num_iterations, 100)
    
    # Plot chính
    plt.plot(iterations, costs, 'g-', linewidth=2, label='Training Cost', marker='s', 
             markersize=4, markevery=5)
    
    # Thêm grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Labels và title
    plt.xlabel('Iterations', fontsize=12, fontweight='bold')
    plt.ylabel('Cost (Cross-Entropy Loss)', fontsize=12, fontweight='bold')
    plt.title(f'CNN Learning Curve\n(lr={learning_rate}, final cost={costs[-1]:.4f})', 
              fontsize=14, fontweight='bold')
    
    # Annotation cho cost thấp nhất
    min_cost_idx = np.argmin(costs)
    min_cost = costs[min_cost_idx]
    min_iter = iterations[min_cost_idx]
    plt.annotate(f'Min: {min_cost:.4f}\n@iter {min_iter}', 
                xy=(min_iter, min_cost), 
                xytext=(min_iter + num_iterations*0.1, min_cost + (max(costs)-min_cost)*0.1),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10, color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    # Annotation cho cost đầu và cuối
    plt.text(iterations[0], costs[0], f'{costs[0]:.2f}', 
             fontsize=9, ha='left', va='bottom', color='green')
    plt.text(iterations[-1], costs[-1], f'{costs[-1]:.4f}', 
             fontsize=9, ha='right', va='top', color='green')
    
    plt.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, 'learning_curve_cnn.png'), dpi=150, bbox_inches='tight')
    print(f"✅ Đã lưu learning curve tại: {os.path.join(checkpoint_dir, 'learning_curve_cnn.png')}")
    plt.close()
    
    # Lưu model cuối cùng
    final_model_path = os.path.join(checkpoint_dir, 'final_model_cnn.npz')
    np.savez(final_model_path, **params)
    print(f"✅ Đã lưu model cuối cùng tại: {final_model_path}")
    
    # Lưu loss history và metrics cho visualization
    metrics_path = os.path.join(checkpoint_dir, 'training_metrics.npz')
    
    # Tính class-wise accuracy
    class_accuracies_train = {}
    class_accuracies_test = {}
    for class_idx in sorted(np.unique(Y_test_flat)):
        # Train accuracy per class
        mask_train = Y_train_flat == class_idx
        class_accuracies_train[class_idx] = np.mean(train_preds[mask_train] == Y_train_flat[mask_train])
        
        # Test accuracy per class
        mask_test = Y_test_flat == class_idx
        class_accuracies_test[class_idx] = np.mean(test_preds[mask_test] == Y_test_flat[mask_test])
    
    np.savez(
        metrics_path,
        costs=np.array(costs),
        iterations=np.arange(0, num_iterations, 100),
        train_accuracy=train_acc,
        test_accuracy=test_acc,
        class_accuracies_train=np.array([class_accuracies_train[i] for i in sorted(class_accuracies_train.keys())]),
        class_accuracies_test=np.array([class_accuracies_test[i] for i in sorted(class_accuracies_test.keys())]),
        learning_rate=learning_rate,
        num_iterations=num_iterations,
        train_samples=X_train.shape[0],
        test_samples=X_test.shape[0],
        num_classes=num_classes
    )
    print(f"✅ Đã lưu training metrics tại: {metrics_path}")
    
    # Lưu thông tin training
    info_path = os.path.join(checkpoint_dir, 'training_info_cnn.txt')
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("THÔNG TIN TRAINING CNN\n")
        f.write("=" * 70 + "\n\n")
        f.write("Architecture:\n")
        f.write("   Conv: 1 -> 8 filters (3x3)\n")
        f.write("   MaxPool: 2x2\n")
        f.write("   FC1: 8*127*127 -> 128\n")
        f.write(f"   FC2: 128 -> {num_classes}\n")
        f.write(f"\nLearning rate: {learning_rate}\n")
        f.write(f"Iterations: {num_iterations}\n")
        f.write(f"Train samples: {X_train.shape[0]}\n")
        f.write(f"Test samples: {X_test.shape[0]}\n")
        f.write(f"\nTrain Accuracy: {train_acc*100:.2f}%\n")
        f.write(f"Test Accuracy: {test_acc*100:.2f}%\n")
        f.write(f"\nFinal Cost: {costs[-1]:.6f}\n")
        f.write(f"Min Cost: {min(costs):.6f} @ iteration {np.argmin(costs)*100}\n")
        f.write(f"\nLabel Mapping:\n")
        for idx, name in label_mapping.items():
            f.write(f"   {idx}: {name}\n")
        f.write(f"\nClass-wise Test Accuracy:\n")
        for class_idx in sorted(np.unique(Y_test_flat)):
            mask = Y_test_flat == class_idx
            class_acc = np.mean(test_preds[mask] == Y_test_flat[mask])
            class_name = label_mapping[class_idx]
            total = np.sum(mask)
            correct = np.sum(test_preds[mask] == Y_test_flat[mask])
            f.write(f"   Class {class_idx} ({class_name:20s}): {class_acc*100:5.2f}% ({correct}/{total})\n")
    print(f"✅ Đã lưu thông tin training tại: {info_path}")
    
    # Lưu label mapping riêng để dễ load
    label_mapping_path = os.path.join(checkpoint_dir, 'label_mapping.npy')
    np.save(label_mapping_path, label_mapping)
    print(f"✅ Đã lưu label mapping tại: {label_mapping_path}")
    
    print("\n" + "="*70)
    print("🎉 HOÀN TẤT!")
    print("="*70)
    
    return params, costs


if __name__ == "__main__":
    # =====================================================================
    # CẤU HÌNH
    # =====================================================================
    
    # Chọn 1 trong 2 options:
    USE_FULL_DATA = True  # True: dùng toàn bộ data, False: dùng sample data
    
    # Hyperparameters
    LEARNING_RATE = 0.001
    NUM_ITERATIONS = 100
    CHECKPOINT_INTERVAL = 10
    CHECKPOINT_DIR = 'checkpoint_cnn'
    
    # =====================================================================
    # LOAD DỮ LIỆU
    # =====================================================================
    
    if USE_FULL_DATA:
        print("📂 Loading TOÀN BỘ dữ liệu từ data_GK...")
        X_train, Y_train, X_test, Y_test, label_mapping = load_data_from_folders(
            data_dir='data_GK',
            test_size=0.2,
            random_state=42
        )
    else:
        print("📂 Loading SAMPLE dữ liệu (20 mẫu/lệnh)...")
        X_train, Y_train, X_test, Y_test, label_mapping = load_sample_data(
            data_dir='data_GK',
            samples_per_class=20,
            test_size=0.2,
            random_state=42
        )
    
    # =====================================================================
    # TRAINING
    # =====================================================================
    
    params, costs = train_cnn(
        X_train, Y_train, X_test, Y_test, label_mapping,
        learning_rate=LEARNING_RATE,
        num_iterations=NUM_ITERATIONS,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        checkpoint_dir=CHECKPOINT_DIR
    )
    
    print("\n💾 Các file đã được lưu trong folder:", CHECKPOINT_DIR)
    print("   - checkpoint_iter_500.npz")
    print("   - checkpoint_iter_1000.npz")
    print("   - checkpoint_iter_1500.npz")
    print("   - ...")
    print("   - final_model_cnn.npz")
    print("   - learning_curve_cnn.png")
    print("   - training_info_cnn.txt")
