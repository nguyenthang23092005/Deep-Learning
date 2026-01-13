import numpy as np
import cv2
from ultil import load_data_from_folders
from model_nn import initialize_parameters_deep, L_model_forward, compute_cost, L_model_backward, update_parameters

print("="*80)
print("DEBUG TRAINING PROCESS")
print("="*80)

# Load data
X_train, Y_train, X_test, Y_test, label_mapping = load_data_from_folders(
    'data_GK', test_size=0.2, random_state=42
)

# Resize
print(f"\n📏 Resizing to (32, 32)...")
X_train_resized = np.array([cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA) 
                            for img in X_train])

# Normalize
print(f"\n🔧 Normalizing...")
print(f"Before: min={X_train_resized.min():.2f}, max={X_train_resized.max():.2f}")
X_train_min = X_train_resized.min()
X_train_max = X_train_resized.max()
X_train_resized = (X_train_resized - X_train_min) / (X_train_max - X_train_min + 1e-8)
print(f"After: min={X_train_resized.min():.2f}, max={X_train_resized.max():.2f}")

# Flatten
X_train_flatten = X_train_resized.reshape(X_train_resized.shape[0], -1).T
print(f"\nX_train_flatten: {X_train_flatten.shape}")
print(f"Y_train: {Y_train.shape}")

# Initialize model
layers_dims = [1024, 256, 128, 64, 7]
np.random.seed(1)
parameters = initialize_parameters_deep(layers_dims)

print(f"\n🔧 Model initialized: {layers_dims}")
print(f"Number of parameters:")
L = len(layers_dims) - 1
for l in range(1, L + 1):
    W_shape = parameters[f'W{l}'].shape
    b_shape = parameters[f'b{l}'].shape
    n_params = np.prod(W_shape) + np.prod(b_shape)
    print(f"   Layer {l}: W{W_shape} + b{b_shape} = {n_params:,} params")

# Check initial predictions
print(f"\n📊 BEFORE TRAINING:")
Z, _ = L_model_forward(X_train_flatten, parameters)
preds = np.argmax(Z, axis=0)
unique, counts = np.unique(preds, return_counts=True)
print(f"Initial predictions distribution:")
for c, n in zip(unique, counts):
    print(f"   Class {c} ({label_mapping[c]}): {n} samples ({n/len(preds)*100:.1f}%)")

initial_cost = compute_cost(Z, Y_train)
print(f"Initial cost: {initial_cost:.6f}")

# Check Y_train distribution
Y_flat = Y_train.flatten()
unique_y, counts_y = np.unique(Y_flat, return_counts=True)
print(f"\nTrue labels distribution:")
for c, n in zip(unique_y, counts_y):
    print(f"   Class {c} ({label_mapping[c]}): {n} samples ({n/len(Y_flat)*100:.1f}%)")

# Train for 500 iterations
print(f"\n🏋️ Training for 500 iterations with LR=0.1...")
learning_rate = 0.1

for i in range(500):
    # Forward
    Z, caches = L_model_forward(X_train_flatten, parameters)
    
    # Cost
    cost = compute_cost(Z, Y_train)
    
    # Backward
    grads = L_model_backward(Z, Y_train, caches)
    
    # Check gradient magnitudes
    if i == 0:
        print(f"\nGradient magnitudes at iteration 0:")
        for key in sorted(grads.keys()):
            grad_norm = np.linalg.norm(grads[key])
            grad_mean = np.abs(grads[key]).mean()
            print(f"   {key}: norm={grad_norm:.6f}, mean={grad_mean:.6f}")
    
    # Update
    parameters = update_parameters(parameters, grads, learning_rate)
    
    if i % 100 == 0:
        preds = np.argmax(Z, axis=0)
        unique, counts = np.unique(preds, return_counts=True)
        pred_str = ", ".join([f"C{c}:{n}" for c, n in zip(unique, counts)])
        print(f"Iter {i:3d} | Cost: {cost:.6f} | Preds: {pred_str}")

print(f"\n📊 AFTER TRAINING:")
Z, _ = L_model_forward(X_train_flatten, parameters)
preds = np.argmax(Z, axis=0)
unique, counts = np.unique(preds, return_counts=True)
print(f"Final predictions distribution:")
for c, n in zip(unique, counts):
    print(f"   Class {c} ({label_mapping[c]}): {n} samples ({n/len(preds)*100:.1f}%)")

final_cost = compute_cost(Z, Y_train)
print(f"Final cost: {final_cost:.6f}")

# Calculate accuracy
Y_flat = Y_train.flatten()
accuracy = np.mean(preds == Y_flat)
print(f"Training accuracy: {accuracy*100:.2f}%")

print(f"\nPer-class accuracy:")
for class_idx in sorted(np.unique(Y_flat)):
    mask = Y_flat == class_idx
    if np.sum(mask) > 0:
        class_acc = np.mean(preds[mask] == Y_flat[mask])
        correct = np.sum(preds[mask] == Y_flat[mask])
        total = np.sum(mask)
        print(f"   Class {class_idx} ({label_mapping[class_idx]}): {class_acc*100:.1f}% ({correct}/{total})")
