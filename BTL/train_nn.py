import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
from ultil import load_data_from_folders, load_sample_data
from model_nn import (
    initialize_parameters_deep,
    L_model_forward,
    compute_cost,
    L_model_backward,
    update_parameters,
    predict
)

def save_checkpoint(parameters, iteration, checkpoint_dir='checkpoint'):
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

def train_model(X_train, Y_train, X_test, Y_test, label_mapping,
                layers_dims, learning_rate=0.01, num_iterations=3000,
                checkpoint_interval=500, checkpoint_dir='checkpoint'):
    """
    Training model với checkpointing
    
    Arguments:
    X_train -- training data, shape (n_samples, height, width)
    Y_train -- training labels, shape (1, n_samples)
    X_test -- test data, shape (n_samples, height, width)
    Y_test -- test labels, shape (1, n_samples)
    label_mapping -- dict mapping từ class index -> command name
    layers_dims -- list kích thước các layer [input_size, hidden1, hidden2, ..., output_size]
    learning_rate -- learning rate
    num_iterations -- số iterations
    checkpoint_interval -- lưu checkpoint sau mỗi bao nhiêu iterations
    checkpoint_dir -- folder lưu checkpoint
    
    Returns:
    parameters -- trained parameters
    costs -- list of costs during training
    """
    
    print("\n" + "="*70)
    print("🚀 BẮT ĐẦU TRAINING")
    print("="*70)
    
    # Resize spectrograms về 32×32
    print(f"\n📏 Resize spectrograms từ {X_train.shape[1:]} về (32, 32)...")
    X_train_resized = np.array([cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA) 
                                for img in X_train])
    X_test_resized = np.array([cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA) 
                               for img in X_test])
    
    print(f"   X_train: {X_train.shape} → {X_train_resized.shape}")
    print(f"   X_test: {X_test.shape} → {X_test_resized.shape}")
    
    # Flatten X từ (n_samples, height, width) -> (height*width, n_samples)
    X_train_flatten = X_train_resized.reshape(X_train_resized.shape[0], -1).T
    X_test_flatten = X_test_resized.reshape(X_test_resized.shape[0], -1).T
    
    print(f"\n📐 Shape sau khi flatten:")
    print(f"   X_train: {X_train_flatten.shape}")
    print(f"   Y_train: {Y_train.shape}")
    print(f"   X_test: {X_test_flatten.shape}")
    print(f"   Y_test: {Y_test.shape}")
    
    # Initialize parameters
    print(f"\n🔧 Khởi tạo model:")
    print(f"   Architecture: {layers_dims}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Iterations: {num_iterations}")
    print(f"   Checkpoint interval: {checkpoint_interval}")
    
    np.random.seed(1)
    parameters = initialize_parameters_deep(layers_dims)
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
        Z, caches = L_model_forward(X_train_flatten, parameters)
        
        # Compute cost
        cost = compute_cost(Z, Y_train)
        
        # Backward propagation
        grads = L_model_backward(Z, Y_train, caches)
        
        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Log cost mỗi 100 iterations
        if i % 100 == 0:
            costs.append(cost)
            print(f"Iteration {i:5d} | Cost: {cost:.6f}")
        
        # Lưu checkpoint
        if (i > 0 and i % checkpoint_interval == 0) or i == num_iterations - 1:
            save_checkpoint(parameters, i, checkpoint_dir)
    
    print("-" * 70)
    print("✅ Hoàn thành training!\n")
    
    # Evaluate trên train set
    print("📊 ĐÁNH GIÁ TRÊN TRAIN SET:")
    print("-" * 70)
    train_preds, train_acc = predict(X_train_flatten, Y_train, parameters)
    
    # Evaluate trên test set
    print("\n📊 ĐÁNH GIÁ TRÊN TEST SET:")
    print("-" * 70)
    test_preds, test_acc = predict(X_test_flatten, Y_test, parameters)
    
    # Chi tiết accuracy cho từng class
    print("\n📋 Accuracy chi tiết cho từng lệnh (Test set):")
    print("-" * 70)
    Y_test_flat = Y_test.flatten()
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
    plt.plot(iterations, costs, 'b-', linewidth=2, label='Training Cost', marker='o', 
             markersize=4, markevery=5)
    
    # Thêm grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Labels và title
    plt.xlabel('Iterations', fontsize=12, fontweight='bold')
    plt.ylabel('Cost (Cross-Entropy Loss)', fontsize=12, fontweight='bold')
    plt.title(f'Neural Network Learning Curve\n(lr={learning_rate}, final cost={costs[-1]:.4f})', 
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
             fontsize=9, ha='left', va='bottom', color='blue')
    plt.text(iterations[-1], costs[-1], f'{costs[-1]:.4f}', 
             fontsize=9, ha='right', va='top', color='blue')
    
    plt.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, 'learning_curve.png'), dpi=150, bbox_inches='tight')
    print(f"✅ Đã lưu learning curve tại: {os.path.join(checkpoint_dir, 'learning_curve.png')}")
    plt.close()
    
    # Lưu model cuối cùng
    final_model_path = os.path.join(checkpoint_dir, 'final_model.npz')
    np.savez(final_model_path, **parameters)
    print(f"✅ Đã lưu model cuối cùng tại: {final_model_path}")
    
    # Lưu loss history và metrics cho visualization
    metrics_path = os.path.join(checkpoint_dir, 'training_metrics.npz')
    
    # Tính class-wise accuracy
    class_accuracies_train = {}
    class_accuracies_test = {}
    for class_idx in sorted(np.unique(Y_test_flat)):
        # Train accuracy per class
        mask_train = Y_train.flatten() == class_idx
        class_accuracies_train[class_idx] = np.mean(train_preds[mask_train] == Y_train.flatten()[mask_train])
        
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
        layers_dims=np.array(layers_dims),
        train_samples=X_train.shape[0],
        test_samples=X_test.shape[0],
        num_classes=len(label_mapping)
    )
    print(f"✅ Đã lưu training metrics tại: {metrics_path}")
    
    # Lưu thông tin training
    info_path = os.path.join(checkpoint_dir, 'training_info.txt')
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("THÔNG TIN TRAINING\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Architecture: {layers_dims}\n")
        f.write(f"Learning rate: {learning_rate}\n")
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
    
    return parameters, costs


if __name__ == "__main__":
    # =====================================================================
    # CẤU HÌNH
    # =====================================================================
    
    # Chọn 1 trong 2 options:
    USE_FULL_DATA = True  # True: dùng toàn bộ data, False: dùng sample data
    
    # Hyperparameters
    LEARNING_RATE = 0.01
    NUM_ITERATIONS = 5000
    CHECKPOINT_INTERVAL = 500
    CHECKPOINT_DIR = 'checkpoint_nn'
    
    # =====================================================================
    # LOAD DỮ LIỆU
    # =====================================================================
    
    if USE_FULL_DATA:
        print("📂 Loading TOÀN BỘ dữ liệu từ data_GK...")
        X_train, Y_train, X_test, Y_test, label_mapping = load_data_from_folders(
            data_dir='D:/DL/BTL/data_GK',
            test_size=0.2,
            random_state=42
        )
    else:
        print("📂 Loading SAMPLE dữ liệu (20 mẫu/lệnh)...")
        X_train, Y_train, X_test, Y_test, label_mapping = load_sample_data(
            data_dir='D:/DL/BTL/data_GK',
            samples_per_class=20,
            test_size=0.2,
            random_state=42
        )
    
    # =====================================================================
    # ĐỊNH NGHĨA KIẾN TRÚC
    # =====================================================================
    
    # Input size = 32 * 32 = 1024 (sau khi resize)
    input_size = 32 * 32
    output_size = len(label_mapping)
    
    # Architecture: Input -> 256 -> 128 -> 64 -> Output
    layers_dims = [input_size, 256, 128, 64, output_size]
    
    print(f"\n🏗️  Kiến trúc mô hình:")
    print(f"   Input: {input_size} (32x32 spectrogram)")
    print(f"   Hidden layers: 256 -> 128 -> 64")
    print(f"   Output: {output_size} classes")
    
    # =====================================================================
    # TRAINING
    # =====================================================================
    
    parameters, costs = train_model(
        X_train, Y_train, X_test, Y_test, label_mapping,
        layers_dims=layers_dims,
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
    print("   - final_model.npz")
    print("   - learning_curve.png")
    print("   - training_info.txt")