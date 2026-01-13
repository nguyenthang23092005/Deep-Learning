import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import cv2
from ultil import load_data_from_folders
from model_nn import predict, compute_cost, L_model_forward

# Cấu hình
DATA_DIR = "data_GK"
CHECKPOINT_DIR = "checkpoint_nn_1"
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
        
        # Load model parameters - they are saved as **parameters (unpacked)
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

def evaluate_model(parameters, X, Y, label_mapping):
    """Đánh giá model trên dataset (7 classes: 6 lệnh + 1 unknown/noise)"""
    # DEBUG: Check input data
    print(f"   [DEBUG] X shape: {X.shape}, min: {X.min():.4f}, max: {X.max():.4f}")
    print(f"   [DEBUG] Y shape: {Y.shape}, unique values: {np.unique(Y)}")
    
    # Predict - predict() expects (X, y, parameters)
    Y_reshaped = Y.reshape(1, -1)  # Reshape to (1, n_samples) for predict
    Y_pred, accuracy = predict(X, Y_reshaped, parameters)
    
    # DEBUG: Check predictions
    unique_preds, counts = np.unique(Y_pred, return_counts=True)
    print(f"   [DEBUG] Predictions - unique classes: {unique_preds}")
    print(f"   [DEBUG] Prediction distribution: {dict(zip(unique_preds, counts))}")
    
    # Accuracy from predict() (already calculated)
    accuracy = accuracy * 100  # Convert to percentage
    
    # Loss - using L_model_forward and compute_cost
    Z, _ = L_model_forward(X, parameters)
    cost = compute_cost(Z, Y_reshaped)
    
    # Get confidence scores (softmax probabilities)
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    probabilities = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    max_confidences = np.max(probabilities, axis=0)
    
    # Confusion matrix
    Y_flat = Y.flatten()
    n_classes = len(label_mapping)
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(Y_flat, Y_pred):
        if true < n_classes and pred < n_classes:
            confusion_matrix[true, pred] += 1
    
    # Per-class accuracy
    per_class_acc = {}
    for i in range(n_classes):
        if confusion_matrix[i, :].sum() > 0:
            per_class_acc[i] = (confusion_matrix[i, i] / confusion_matrix[i, :].sum()) * 100
        else:
            per_class_acc[i] = 0.0
    
    # Identify unknown/noise class
    unknown_class_idx = None
    unknown_class_name = None
    for idx, name in label_mapping.items():
        if 'noise' in name.lower() or 'unknown' in name.lower():
            unknown_class_idx = idx
            unknown_class_name = name
            break
    
    # Unknown/noise class statistics
    unknown_stats = {
        'has_unknown_class': unknown_class_idx is not None,
        'unknown_class_idx': unknown_class_idx,
        'unknown_class_name': unknown_class_name,
        'avg_confidence': np.mean(max_confidences),
        'min_confidence': np.min(max_confidences),
        'max_confidence': np.max(max_confidences)
    }
    
    if unknown_class_idx is not None:
        # Unknown class recall (얼마나 nhiễu được phát hiện đúng)
        unknown_mask = Y_flat == unknown_class_idx
        if np.sum(unknown_mask) > 0:
            unknown_correct = np.sum(Y_pred[unknown_mask] == unknown_class_idx)
            unknown_total = np.sum(unknown_mask)
            unknown_recall = (unknown_correct / unknown_total) * 100
            unknown_stats['unknown_samples'] = unknown_total
            unknown_stats['unknown_correct'] = unknown_correct
            unknown_stats['unknown_recall'] = unknown_recall
        
        # False unknown rate (얼마나 lệnh thực bị nhầm thành nhiễu)
        known_mask = Y_flat != unknown_class_idx
        if np.sum(known_mask) > 0:
            false_unknown = np.sum(Y_pred[known_mask] == unknown_class_idx)
            known_total = np.sum(known_mask)
            false_unknown_rate = (false_unknown / known_total) * 100
            unknown_stats['false_unknown_count'] = false_unknown
            unknown_stats['false_unknown_rate'] = false_unknown_rate
            
        # Confidence cho unknown class
        if np.sum(unknown_mask) > 0:
            unknown_confidences = max_confidences[unknown_mask]
            unknown_stats['unknown_avg_confidence'] = np.mean(unknown_confidences)
            unknown_stats['unknown_min_confidence'] = np.min(unknown_confidences)
    
    return {
        'accuracy': accuracy,
        'loss': cost,
        'predictions': Y_pred,
        'probabilities': probabilities,
        'confidences': max_confidences,
        'confusion_matrix': confusion_matrix,
        'per_class_accuracy': per_class_acc,
        'unknown_stats': unknown_stats
    }

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
                    arch_str = line.split(':')[1].strip()
                    info['architecture'] = arch_str
                elif 'Total Parameters:' in line:
                    info['total_params'] = int(line.split(':')[1].strip().replace(',', ''))
    
    return info

def evaluate_all_models():
    """Đánh giá tất cả các model đã train"""
    print("="*80)
    print("EVALUATE ALL TRAINED MODELS")
    print("="*80)
    
    # Load data với cùng split như lúc train (test_size=0.2, random_state=42)
    print("\n📂 Loading data (same split as training)...")
    X_train, Y_train, X_test, Y_test, label_mapping = load_data_from_folders(
        DATA_DIR, test_size=0.2, random_state=42
    )
    
    print(f"\n📊 Dataset info:")
    print(f"   Train set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    print(f"   Classes: {len(label_mapping)}")
    for idx, name in label_mapping.items():
        train_count = np.sum(Y_train.flatten() == idx)
        test_count = np.sum(Y_test.flatten() == idx)
        print(f"      Class {idx} ({name}): Train={train_count}, Test={test_count}")
    
    # Preprocess TEST SET - MUST use same normalization as training!
    print("\n📐 Preprocessing TEST set...")
    X_test_resized = np.array([cv2.resize(img, RESIZE_SHAPE, interpolation=cv2.INTER_AREA) 
                               for img in X_test])
    print(f"   Before normalization: min={X_test_resized.min():.2f}, max={X_test_resized.max():.2f}")
    
    # Preprocess TRAIN SET to get normalization parameters (same as training)
    print("\n📐 Preprocessing TRAIN set to get normalization parameters...")
    X_train_resized = np.array([cv2.resize(img, RESIZE_SHAPE, interpolation=cv2.INTER_AREA) 
                                for img in X_train])
    
    # Get train set's min/max
    X_train_min = X_train_resized.min()
    X_train_max = X_train_resized.max()
    print(f"   Train set: min={X_train_min:.2f}, max={X_train_max:.2f}")
    
    # Normalize TEST set using TRAIN set's statistics (CRITICAL: same as training!)
    X_test_resized = (X_test_resized - X_train_min) / (X_train_max - X_train_min + 1e-8)
    print(f"   After normalization: min={X_test_resized.min():.2f}, max={X_test_resized.max():.2f}")
    
    # Preprocess TRAIN SET to get normalization parameters (same as training)
    print("\n📐 Preprocessing TRAIN set to get normalization parameters...")
    X_train_resized = np.array([cv2.resize(img, RESIZE_SHAPE, interpolation=cv2.INTER_AREA) 
                                for img in X_train])
    
    # Get train set's min/max
    X_train_min = X_train_resized.min()
    X_train_max = X_train_resized.max()
    print(f"   Train set: min={X_train_min:.2f}, max={X_train_max:.2f}")
    
    # Normalize TEST set using TRAIN set's statistics (CRITICAL: same as training!)
    X_test_resized = (X_test_resized - X_train_min) / (X_train_max - X_train_min + 1e-8)
    print(f"   After normalization: min={X_test_resized.min():.2f}, max={X_test_resized.max():.2f}")
    
    # Flatten
    X_test_eval = X_test_resized.reshape(X_test_resized.shape[0], -1).T
    Y_test_eval = Y_test.flatten()
    
    print(f"✅ Test data ready: X={X_test_eval.shape}, Y={Y_test_eval.shape}")
    
    # Find all run directories
    run_dirs = []
    if os.path.exists(CHECKPOINT_DIR):
        for item in os.listdir(CHECKPOINT_DIR):
            item_path = os.path.join(CHECKPOINT_DIR, item)
            if os.path.isdir(item_path) and item.startswith('run_'):
                run_dirs.append(item_path)
    
    run_dirs = sorted(run_dirs)
    print(f"\n📊 Found {len(run_dirs)} trained models")
    
    if len(run_dirs) == 0:
        print("❌ No trained models found!")
        return
    
    # Evaluate each model
    results = []
    for run_dir in run_dirs:
        print(f"\n{'='*60}")
        print(f"Evaluating: {os.path.basename(run_dir)}")
        print(f"{'='*60}")
        
        model_data = load_model(run_dir)
        if model_data is None:
            continue
        
        # Get training info
        train_info = get_training_info(run_dir)
        
        # Evaluate on TEST set (same as used during training)
        eval_results = evaluate_model(model_data['parameters'], X_test_eval, Y_test_eval, label_mapping)
        
        # Get training accuracy from metrics
        train_accuracy = None
        test_accuracy_from_training = None
        if model_data['metrics'] is not None:
            if 'train_accuracy' in model_data['metrics']:
                train_accuracy = float(model_data['metrics']['train_accuracy']) * 100
            if 'test_accuracy' in model_data['metrics']:
                test_accuracy_from_training = float(model_data['metrics']['test_accuracy']) * 100
        
        # Combine results
        result = {
            'run_name': model_data['run_name'],
            'accuracy': eval_results['accuracy'],
            'train_accuracy': train_accuracy,
            'test_accuracy_from_training': test_accuracy_from_training,
            'loss': eval_results['loss'],
            'confusion_matrix': eval_results['confusion_matrix'],
            'per_class_accuracy': eval_results['per_class_accuracy'],
            'unknown_stats': eval_results['unknown_stats'],
            'metrics': model_data['metrics'],
            'learning_rate': train_info.get('learning_rate', 'N/A'),
            'architecture': train_info.get('architecture', 'N/A'),
            'total_params': train_info.get('total_params', 'N/A')
        }
        
        results.append(result)
        
        print(f"✅ Test Accuracy (current eval): {eval_results['accuracy']:.2f}%")
        if train_accuracy is not None:
            print(f"   Train Accuracy (from training): {train_accuracy:.2f}%")
            overfitting = train_accuracy - eval_results['accuracy']
            if overfitting > 5:
                print(f"   ⚠️  Potential overfitting: {overfitting:.2f}% gap")
        if test_accuracy_from_training is not None:
            print(f"   Test Accuracy (from training): {test_accuracy_from_training:.2f}%")
        print(f"   Loss: {eval_results['loss']:.4f}")
        if result['learning_rate'] != 'N/A':
            print(f"   Learning Rate: {result['learning_rate']}")
        if result['architecture'] != 'N/A':
            print(f"   Architecture: {result['architecture']}")
        
        # Print unknown/noise class info
        unknown_stats = eval_results['unknown_stats']
        if unknown_stats['has_unknown_class']:
            print(f"\n   🔊 Unknown/Noise Class: '{unknown_stats['unknown_class_name']}'")
            if 'unknown_samples' in unknown_stats:
                print(f"      Total samples: {unknown_stats['unknown_samples']}")
                print(f"      Correctly detected: {unknown_stats['unknown_correct']}/{unknown_stats['unknown_samples']} ({unknown_stats['unknown_recall']:.2f}%)")
            if 'false_unknown_rate' in unknown_stats:
                print(f"      Known commands misclassified as noise: {unknown_stats['false_unknown_count']} ({unknown_stats['false_unknown_rate']:.2f}%)")
            if 'unknown_avg_confidence' in unknown_stats:
                print(f"      Avg confidence (on noise samples): {unknown_stats['unknown_avg_confidence']:.4f}")
        
        print(f"\n   📊 Overall Confidence Stats:")
        print(f"      Mean: {unknown_stats['avg_confidence']:.4f}")
        print(f"      Range: [{unknown_stats['min_confidence']:.4f}, {unknown_stats['max_confidence']:.4f}]")
    
    # Visualize results
    if len(results) > 0:
        visualize_all_results(results, label_mapping)
    
    return results

def visualize_all_results(results, label_mapping):
    """Visualize tất cả kết quả trên cùng 1 figure"""
    print("\n" + "="*80)
    print("CREATING COMPREHENSIVE VISUALIZATION")
    print("="*80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    n_models = len(results)
    
    # ========== FIGURE 1: Training Curves ==========
    fig1 = plt.figure(figsize=(20, 12))
    gs1 = fig1.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Training Loss
    ax1 = fig1.add_subplot(gs1[0, 0])
    for i, result in enumerate(results):
        if result['metrics'] is not None:
            costs = result['metrics']['costs']
            iterations = result['metrics']['iterations']
            lr = result['learning_rate']
            label = f"{result['run_name']} ({lr})" if lr != 'N/A' else result['run_name']
            ax1.plot(iterations, costs, marker='o', markersize=3, 
                    label=label, linewidth=1.5)
    
    ax1.set_xlabel('Iterations', fontsize=11)
    ax1.set_ylabel('Cost (Loss)', fontsize=11)
    ax1.set_title('Training Loss Curves - All Models', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final Loss Comparison (Bar Chart)
    ax2 = fig1.add_subplot(gs1[0, 1])
    run_names = [f"Run {i+1}\n{r['learning_rate']}" if r['learning_rate'] != 'N/A' else f"Run {i+1}" for i, r in enumerate(results)]
    final_losses = [r['loss'] for r in results]
    colors = plt.cm.viridis(np.linspace(0, 1, n_models))
    
    bars = ax2.bar(range(n_models), final_losses, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Model', fontsize=11)
    ax2.set_ylabel('Final Loss', fontsize=11)
    ax2.set_title('Final Loss Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(n_models))
    ax2.set_xticklabels(run_names, fontsize=8, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, loss) in enumerate(zip(bars, final_losses)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Accuracy Comparison (Bar Chart)
    ax3 = fig1.add_subplot(gs1[1, 0])
    accuracies = [r['accuracy'] for r in results]
    
    bars = ax3.bar(range(n_models), accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Model', fontsize=11)
    ax3.set_ylabel('Accuracy (%)', fontsize=11)
    ax3.set_title('Test Accuracy Comparison', fontsize=13, fontweight='bold')
    ax3.set_xticks(range(n_models))
    ax3.set_xticklabels(run_names, fontsize=8, rotation=45, ha='right')
    ax3.set_ylim([0, 100])
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=100, color='green', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Summary Table
    ax4 = fig1.add_subplot(gs1[1, 1])
    ax4.axis('off')
    
    # Check if any model has valid LR
    has_valid_lr = any(r['learning_rate'] != 'N/A' for r in results)
    
    # Create table data
    table_data = []
    if has_valid_lr:
        table_data.append(['Model', 'LR', 'Accuracy', 'Loss', 'Architecture'])
    else:
        table_data.append(['Model', 'Accuracy', 'Loss', 'Architecture'])
    
    for i, result in enumerate(results):
        arch_short = str(result['architecture'])[:20] + '...' if len(str(result['architecture'])) > 20 else str(result['architecture'])
        if has_valid_lr:
            table_data.append([
                f"Run {i+1}",
                f"{result['learning_rate']}" if result['learning_rate'] != 'N/A' else '-',
                f"{result['accuracy']:.2f}%",
                f"{result['loss']:.4f}",
                arch_short
            ])
        else:
            table_data.append([
                f"Run {i+1}",
                f"{result['accuracy']:.2f}%",
                f"{result['loss']:.4f}",
                arch_short
            ])
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    num_cols = 5 if has_valid_lr else 4
    for i in range(num_cols):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(num_cols):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
    
    fig1.suptitle('MODEL TRAINING COMPARISON - LOSS & ACCURACY', 
                  fontsize=16, fontweight='bold', y=0.98)
    
    output_path1 = os.path.join(OUTPUT_DIR, 'model_comparison_training.png')
    plt.savefig(output_path1, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved training comparison: {output_path1}")
    
    # ========== FIGURE 2: Per-Class Performance ==========
    fig2 = plt.figure(figsize=(20, 10))
    
    n_classes = len(label_mapping)
    class_names = [label_mapping[i] for i in range(n_classes)]
    
    # Prepare per-class accuracy data
    per_class_data = np.zeros((n_models, n_classes))
    for i, result in enumerate(results):
        for class_idx, acc in result['per_class_accuracy'].items():
            per_class_data[i, class_idx] = acc
    
    # Grouped bar chart
    x = np.arange(n_classes)
    width = 0.8 / n_models
    
    for i in range(n_models):
        offset = (i - n_models/2) * width + width/2
        label = f"Run {i+1} ({results[i]['learning_rate']})" if results[i]['learning_rate'] != 'N/A' else f"Run {i+1}"
        plt.bar(x + offset, per_class_data[i], width, 
               label=label,
               alpha=0.7, edgecolor='black')
    
    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Per-Class Accuracy Comparison Across All Models', 
             fontsize=15, fontweight='bold', pad=20)
    plt.xticks(x, class_names, rotation=45, ha='right', fontsize=10)
    plt.ylim([0, 105])
    plt.legend(fontsize=9, loc='upper right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.axhline(y=100, color='green', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    output_path2 = os.path.join(OUTPUT_DIR, 'model_comparison_per_class.png')
    plt.savefig(output_path2, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved per-class comparison: {output_path2}")
    
    # ========== FIGURE 3: Best Model Confusion Matrix ==========
    best_idx = np.argmax([r['accuracy'] for r in results])
    best_result = results[best_idx]
    
    fig3, ax = plt.subplots(figsize=(12, 10))
    
    cm = best_result['confusion_matrix']
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', fontsize=11)
    
    # Set ticks
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    
    # Add text annotations
    for i in range(n_classes):
        for j in range(n_classes):
            text = ax.text(j, i, cm[i, j],
                          ha="center", va="center",
                          color="white" if cm[i, j] > cm.max()/2 else "black",
                          fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    lr_text = f' | LR: {best_result["learning_rate"]}' if best_result['learning_rate'] != 'N/A' else ''
    ax.set_title(f'Confusion Matrix - Best Model: {best_result["run_name"]}\n'
                f'Accuracy: {best_result["accuracy"]:.2f}%{lr_text}',
                fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    output_path3 = os.path.join(OUTPUT_DIR, 'best_model_confusion_matrix.png')
    plt.savefig(output_path3, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved confusion matrix: {output_path3}")
    
    plt.show()
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"\n🏆 Best Model: {best_result['run_name']}")
    print(f"   Accuracy: {best_result['accuracy']:.2f}%")
    print(f"   Loss: {best_result['loss']:.4f}")
    print(f"   Learning Rate: {best_result['learning_rate']}")
    print(f"   Architecture: {best_result['architecture']}")
    print("\n📊 All Models:")
    for i, result in enumerate(results):
        lr_text = f" ({result['learning_rate']})" if result['learning_rate'] != 'N/A' else ""
        print(f"   {i+1}. {result['run_name']}: {result['accuracy']:.2f}%{lr_text}")
        if result['train_accuracy'] is not None:
            gap = result['train_accuracy'] - result['accuracy']
            print(f"       Train: {result['train_accuracy']:.2f}% | Gap: {gap:.2f}%")
    print("="*80)

if __name__ == "__main__":
    evaluate_all_models()
