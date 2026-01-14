import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import cv2
from ultil import load_data_from_folders
from model_nn import predict, compute_cost, L_model_forward

# Cấu hình
DATA_DIR = "data_GK"
CHECKPOINT_DIR_NORMALIZED = "checkpoint_nn"  # Có chuẩn hóa 0-1
CHECKPOINT_DIR_NO_NORM = "checkpoint_nn_1"   # Không có chuẩn hóa
OUTPUT_DIR = "visual"
RESIZE_SHAPE = (32, 32)

def load_model(run_path):
    """Load model từ checkpoint"""
    try:
        model_path = os.path.join(run_path, 'final_model.npz')
        label_path = os.path.join(run_path, 'label_mapping.npy')
        metrics_path = os.path.join(run_path, 'training_metrics.npz')
        
        if not os.path.exists(model_path):
            return None
        
        # Load model parameters
        data = np.load(model_path)
        parameters = {}
        for key in data.files:
            parameters[key] = data[key]
        
        # Load label mapping
        label_mapping = np.load(label_path, allow_pickle=True).item()
        
        # Load training metrics
        metrics = None
        if os.path.exists(metrics_path):
            metrics = np.load(metrics_path)
        
        return {
            'parameters': parameters,
            'label_mapping': label_mapping,
            'metrics': metrics,
            'run_name': os.path.basename(run_path)
        }
    except Exception as e:
        print(f"❌ Error loading model from {run_path}: {e}")
        return None

def evaluate_model(parameters, X, Y):
    """Đánh giá model trên dataset"""
    Y_reshaped = Y.reshape(1, -1)
    Y_pred, accuracy = predict(X, Y_reshaped, parameters)
    accuracy = accuracy * 100  # Convert to percentage
    
    Z, _ = L_model_forward(X, parameters)
    cost = compute_cost(Z, Y_reshaped)
    
    return accuracy, cost

def get_training_info(run_path):
    """Đọc thông tin từ training_info.txt"""
    info_path = os.path.join(run_path, 'training_info.txt')
    info = {}
    
    if os.path.exists(info_path):
        with open(info_path, 'r', encoding='utf-8') as f:
            for line in f:
                if 'Learning Rate:' in line:
                    info['learning_rate'] = float(line.split(':')[1].strip())
                elif 'Architecture:' in line:
                    info['architecture'] = line.split(':')[1].strip()
    
    return info

def find_best_model(checkpoint_dir):
    """Tìm model tốt nhất trong thư mục checkpoint"""
    run_dirs = []
    if os.path.exists(checkpoint_dir):
        for item in os.listdir(checkpoint_dir):
            item_path = os.path.join(checkpoint_dir, item)
            if os.path.isdir(item_path) and item.startswith('run_'):
                run_dirs.append(item_path)
    
    if len(run_dirs) == 0:
        print(f"❌ No models found in {checkpoint_dir}")
        return None
    
    best_model = None
    best_accuracy = -1
    
    X_train, Y_train, X_test, Y_test, label_mapping = load_data_from_folders(
        DATA_DIR, test_size=0.2, random_state=42
    )
    
    is_normalized = 'checkpoint_nn' in checkpoint_dir and 'checkpoint_nn_1' not in checkpoint_dir
    
    print(f"\n📂 {checkpoint_dir}")
    
    X_test_resized = np.array([cv2.resize(img, RESIZE_SHAPE, interpolation=cv2.INTER_AREA) 
                               for img in X_test])
    
    if is_normalized:
        X_min = X_test_resized.min()
        X_max = X_test_resized.max()
        X_test_eval = (X_test_resized - X_min) / (X_max - X_min + 1e-8)
    else:
        X_test_eval = X_test_resized
    
    X_test_eval = X_test_eval.reshape(X_test_eval.shape[0], -1).T
    Y_test_eval = Y_test.flatten()
    
    for run_dir in run_dirs:
        model_data = load_model(run_dir)
        if model_data is None:
            continue
        
        accuracy, loss = evaluate_model(model_data['parameters'], X_test_eval, Y_test_eval)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            train_info = get_training_info(run_dir)
            best_model = {
                'run_name': model_data['run_name'],
                'accuracy': accuracy,
                'loss': loss,
                'learning_rate': train_info.get('learning_rate', 'N/A'),
                'architecture': train_info.get('architecture', 'N/A')
            }
    
    return best_model, label_mapping

def compare_normalization():
    """So sánh mô hình tốt nhất giữa 2 phương pháp"""
    print("="*80)
    print("COMPARE NORMALIZATION METHODS")
    print("="*80)
    
    print("\n🔍 Finding best model with normalization...")
    model_normalized, label_mapping = find_best_model(CHECKPOINT_DIR_NORMALIZED)
    
    print("\n🔍 Finding best model without normalization...")
    model_no_norm, _ = find_best_model(CHECKPOINT_DIR_NO_NORM)
    
    if model_normalized is None or model_no_norm is None:
        print("❌ Could not load models!")
        return
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print(f"\n🟦 WITH NORMALIZATION: {model_normalized['accuracy']:.2f}%")
    print(f"🟧 WITHOUT NORMALIZATION: {model_no_norm['accuracy']:.2f}%")
    
    diff = model_normalized['accuracy'] - model_no_norm['accuracy']
    print(f"\n📊 Difference: {diff:+.2f}%")
    
    visualize_comparison(model_normalized, model_no_norm)

def visualize_comparison(model_norm, model_no_norm):
    """Tạo biểu đồ so sánh accuracy"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    models = ['With Normalization\n(Min-Max [0,1])', 'Without Normalization\n(Original Spectrogram)']
    accuracies = [model_norm['accuracy'], model_no_norm['accuracy']]
    colors = ['#4472C4', '#ED7D31']
    
    bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Accuracy Comparison:\nNormalization vs No Normalization', 
                  fontsize=15, fontweight='bold', pad=20)
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%', ha='center', va='bottom', 
                fontsize=12, fontweight='bold')
    
    diff = model_norm['accuracy'] - model_no_norm['accuracy']
    mid_y = (model_norm['accuracy'] + model_no_norm['accuracy']) / 2
    ax.annotate('', xy=(1, model_no_norm['accuracy']), xytext=(0, model_norm['accuracy']),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(0.5, mid_y, f'Δ = {diff:+.2f}%', 
            ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    output_path = os.path.join(OUTPUT_DIR, 'normalization_comparison_accuracy.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    compare_normalization()
