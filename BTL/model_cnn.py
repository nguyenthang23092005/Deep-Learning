import numpy as np
import matplotlib.pyplot as plt

def batch_norm_forward(X, gamma, beta, eps=1e-5):
    """
    Batch Normalization forward (simplified - training mode only)
    X: (N, C, H, W)
    """
    N, C, H, W = X.shape
    mean = np.mean(X, axis=(0, 2, 3), keepdims=True)
    var = np.var(X, axis=(0, 2, 3), keepdims=True)
    X_norm = (X - mean) / np.sqrt(var + eps)
    out = gamma.reshape(1, C, 1, 1) * X_norm + beta.reshape(1, C, 1, 1)
    cache = (X, X_norm, mean, var, gamma, beta, eps)
    return out, cache

def batch_norm_backward(dout, cache):
    """BN backward - simplified"""
    X, X_norm, mean, var, gamma, beta, eps = cache
    N, C, H, W = X.shape
    
    dgamma = np.sum(dout * X_norm, axis=(0, 2, 3))
    dbeta = np.sum(dout, axis=(0, 2, 3))
    
    dX_norm = dout * gamma.reshape(1, C, 1, 1)
    dvar = np.sum(dX_norm * (X - mean) * -0.5 * (var + eps)**(-1.5), axis=(0, 2, 3), keepdims=True)
    dmean = np.sum(dX_norm * -1 / np.sqrt(var + eps), axis=(0, 2, 3), keepdims=True)
    
    dX = dX_norm / np.sqrt(var + eps) + dvar * 2 * (X - mean) / (N*H*W) + dmean / (N*H*W)
    
    return dX, dgamma, dbeta

def conv_forward(X, W, b, stride=1, padding=0):
    """
    X: (N, C, H, W)
    W: (F, C, HH, WW)
    b: (F,)
    """
    N, C, H, W_ = X.shape
    F, _, HH, WW = W.shape

    H_out = (H + 2*padding - HH) // stride + 1
    W_out = (W_ + 2*padding - WW) // stride + 1

    if padding > 0:
        X_pad = np.pad(X, ((0,0),(0,0),(padding,padding),(padding,padding)))
    else:
        X_pad = X
        
    out = np.zeros((N, F, H_out, W_out))

    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    h = i * stride
                    w = j * stride
                    out[n,f,i,j] = np.sum(X_pad[n,:,h:h+HH,w:w+WW] * W[f]) + b[f]

    cache = (X, W, b, stride, padding)
    return out, cache

def conv_backward(dout, cache):
    """Simplified conv backward"""
    X, W, b, stride, padding = cache
    N, C, H, W_ = X.shape
    F, _, HH, WW = W.shape
    N, F, H_out, W_out = dout.shape
    
    dW = np.zeros_like(W)
    db = np.sum(dout, axis=(0, 2, 3))
    dX = np.zeros_like(X)
    
    if padding > 0:
        X_pad = np.pad(X, ((0,0),(0,0),(padding,padding),(padding,padding)))
        dX_pad = np.zeros_like(X_pad)
    else:
        X_pad = X
        dX_pad = dX
    
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    h = i * stride
                    w = j * stride
                    dW[f] += X_pad[n,:,h:h+HH,w:w+WW] * dout[n,f,i,j]
                    dX_pad[n,:,h:h+HH,w:w+WW] += W[f] * dout[n,f,i,j]
    
    if padding > 0:
        dX = dX_pad[:,:,padding:-padding,padding:-padding]
    else:
        dX = dX_pad
        
    return dX, dW, db

def relu_forward(Z):
    A = np.maximum(0, Z)
    return A, Z

def relu_backward(dA, cache):
    Z = cache
    dZ = dA.copy()
    dZ[Z <= 0] = 0
    return dZ

def maxpool_forward(X, size=2, stride=2):
    N, C, H, W = X.shape
    H_out = (H - size) // stride + 1
    W_out = (W - size) // stride + 1

    out = np.zeros((N, C, H_out, W_out))
    cache_mask = np.zeros_like(X)

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h = i * stride
                    w = j * stride
                    window = X[n,c,h:h+size,w:w+size]
                    out[n,c,i,j] = np.max(window)
                    
                    # Store mask for backward
                    max_idx = np.unravel_index(np.argmax(window), window.shape)
                    cache_mask[n,c,h+max_idx[0],w+max_idx[1]] = 1

    cache = (X, size, stride, cache_mask)
    return out, cache

def maxpool_backward(dout, cache):
    X, size, stride, mask = cache
    dX = np.zeros_like(X)
    N, C, H_out, W_out = dout.shape
    
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h = i * stride
                    w = j * stride
                    dX[n,c,h:h+size,w:w+size] += dout[n,c,i,j] * mask[n,c,h:h+size,w:w+size]
    
    return dX

def depthwise_conv_forward(X, W, b, stride=1, padding=0):
    """
    Depthwise convolution
    X: (N, C, H, W)
    W: (C, 1, HH, WW) - one filter per channel
    b: (C,)
    """
    N, C, H, W_ = X.shape
    _, _, HH, WW = W.shape
    
    H_out = (H + 2*padding - HH) // stride + 1
    W_out = (W_ + 2*padding - WW) // stride + 1
    
    if padding > 0:
        X_pad = np.pad(X, ((0,0),(0,0),(padding,padding),(padding,padding)))
    else:
        X_pad = X
    
    out = np.zeros((N, C, H_out, W_out))
    
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h = i * stride
                    w = j * stride
                    out[n,c,i,j] = np.sum(X_pad[n,c,h:h+HH,w:w+WW] * W[c,0]) + b[c]
    
    cache = (X, W, b, stride, padding)
    return out, cache

def global_avgpool_forward(X):
    """Global Average Pooling"""
    N, C, H, W = X.shape
    out = np.mean(X, axis=(2, 3))  # (N, C)
    cache = X.shape
    return out, cache

def global_avgpool_backward(dout, cache):
    N, C, H, W = cache
    dX = np.zeros((N, C, H, W))
    dX = dout.reshape(N, C, 1, 1) / (H * W)
    dX = np.repeat(np.repeat(dX, H, axis=2), W, axis=3)
    return dX

def dropout_forward(X, p=0.5, training=True):
    if not training:
        return X, None
    mask = (np.random.rand(*X.shape) > p) / (1 - p)
    out = X * mask
    return out, mask

def dropout_backward(dout, mask):
    if mask is None:
        return dout
    return dout * mask

def linear_forward(A, W, b):
    Z = A @ W.T + b.T
    cache = (A, W, b)
    return Z, cache

def linear_backward(dZ, cache):
    A, W, b = cache
    dA = dZ @ W
    dW = dZ.T @ A
    db = np.sum(dZ, axis=0, keepdims=True).T
    return dA, dW, db

def softmax_cross_entropy(Z, y):
    Z -= np.max(Z, axis=1, keepdims=True)
    log_probs = Z - np.log(np.sum(np.exp(Z), axis=1, keepdims=True))
    loss = -np.mean(log_probs[np.arange(len(y)), y])

    probs = np.exp(log_probs)
    dZ = probs
    dZ[np.arange(len(y)), y] -= 1
    dZ /= len(y)

    return loss, dZ

def cnn_forward(X, params, training=True):
    """
    New CNN architecture:
    Input: (N, 1, 128, 64)
    Conv3x3 + BN + ReLU -> (N, 16, 128, 64)
    MaxPool -> (N, 16, 64, 32)
    Conv3x3 + BN + ReLU -> (N, 32, 64, 32)
    MaxPool -> (N, 32, 32, 16)
    Depthwise Conv + BN + ReLU -> (N, 32, 32, 16)
    GlobalAvgPool -> (N, 32)
    Dense 64 + Dropout -> (N, 64)
    Dense 6 + Softmax -> (N, 6)
    """
    caches = {}
    
    # Conv1 + BN + ReLU
    Z1, caches["conv1"] = conv_forward(X, params["W1"], params["b1"], stride=1, padding=1)
    BN1, caches["bn1"] = batch_norm_forward(Z1, params["gamma1"], params["beta1"])
    A1, caches["relu1"] = relu_forward(BN1)
    
    # MaxPool1
    P1, caches["pool1"] = maxpool_forward(A1, size=2, stride=2)
    
    # Conv2 + BN + ReLU
    Z2, caches["conv2"] = conv_forward(P1, params["W2"], params["b2"], stride=1, padding=1)
    BN2, caches["bn2"] = batch_norm_forward(Z2, params["gamma2"], params["beta2"])
    A2, caches["relu2"] = relu_forward(BN2)
    
    # MaxPool2
    P2, caches["pool2"] = maxpool_forward(A2, size=2, stride=2)
    
    # Depthwise Conv + BN + ReLU
    Z3, caches["dwconv"] = depthwise_conv_forward(P2, params["Wdw"], params["bdw"], stride=1, padding=1)
    BN3, caches["bn3"] = batch_norm_forward(Z3, params["gamma3"], params["beta3"])
    A3, caches["relu3"] = relu_forward(BN3)
    
    # Global Average Pooling
    GAP, caches["gap"] = global_avgpool_forward(A3)
    
    # Dense1 + Dropout
    Z4, caches["fc1"] = linear_forward(GAP, params["Wfc1"], params["bfc1"])
    A4, caches["relu4"] = relu_forward(Z4)
    D4, caches["dropout"] = dropout_forward(A4, p=0.5, training=training)
    
    # Dense2 (output)
    Z5, caches["fc2"] = linear_forward(D4, params["Wfc2"], params["bfc2"])
    
    return Z5, caches

def cnn_backward(dZ5, caches, params):
    grads = {}
    
    # FC2 backward
    dD4, grads["dWfc2"], grads["dbfc2"] = linear_backward(dZ5, caches["fc2"])
    
    # Dropout backward
    dA4 = dropout_backward(dD4, caches["dropout"])
    
    # ReLU4 + FC1 backward
    dZ4 = relu_backward(dA4, caches["relu4"])
    dGAP, grads["dWfc1"], grads["dbfc1"] = linear_backward(dZ4, caches["fc1"])
    
    # Global Avg Pool backward
    dA3 = global_avgpool_backward(dGAP, caches["gap"])
    
    # ReLU3 + BN3 backward
    dBN3 = relu_backward(dA3, caches["relu3"])
    dZ3, grads["dgamma3"], grads["dbeta3"] = batch_norm_backward(dBN3, caches["bn3"])
    
    # Depthwise Conv backward (simplified - just pass gradient)
    dP2 = dZ3  # Simplified
    
    # Pool2 backward
    dA2 = maxpool_backward(dP2, caches["pool2"])
    
    # ReLU2 + BN2 + Conv2 backward
    dBN2 = relu_backward(dA2, caches["relu2"])
    dZ2, grads["dgamma2"], grads["dbeta2"] = batch_norm_backward(dBN2, caches["bn2"])
    dP1, grads["dW2"], grads["db2"] = conv_backward(dZ2, caches["conv2"])
    
    # Pool1 backward
    dA1 = maxpool_backward(dP1, caches["pool1"])
    
    # ReLU1 + BN1 + Conv1 backward
    dBN1 = relu_backward(dA1, caches["relu1"])
    dZ1, grads["dgamma1"], grads["dbeta1"] = batch_norm_backward(dBN1, caches["bn1"])
    dX, grads["dW1"], grads["db1"] = conv_backward(dZ1, caches["conv1"])
    
    # Depthwise conv gradients (simplified)
    grads["dWdw"] = np.zeros_like(params["Wdw"])
    grads["dbdw"] = np.zeros_like(params["bdw"])
    
    return grads

def init_cnn_params(num_classes=6):
    """
    Initialize parameters for new architecture
    Input: (N, 1, 128, 64)
    """
    params = {}
    
    # Conv1: 1 -> 16 filters, 3x3
    params["W1"] = np.random.randn(16, 1, 3, 3) * np.sqrt(2.0 / (1*3*3))
    params["b1"] = np.zeros(16)
    params["gamma1"] = np.ones(16)
    params["beta1"] = np.zeros(16)
    
    # Conv2: 16 -> 32 filters, 3x3
    params["W2"] = np.random.randn(32, 16, 3, 3) * np.sqrt(2.0 / (16*3*3))
    params["b2"] = np.zeros(32)
    params["gamma2"] = np.ones(32)
    params["beta2"] = np.zeros(32)
    
    # Depthwise Conv: 32 channels, 3x3
    params["Wdw"] = np.random.randn(32, 1, 3, 3) * np.sqrt(2.0 / (1*3*3))
    params["bdw"] = np.zeros(32)
    params["gamma3"] = np.ones(32)
    params["beta3"] = np.zeros(32)
    
    # FC1: 32 -> 64
    params["Wfc1"] = np.random.randn(64, 32) * np.sqrt(2.0 / 32)
    params["bfc1"] = np.zeros((64, 1))
    
    # FC2: 64 -> num_classes
    params["Wfc2"] = np.random.randn(num_classes, 64) * np.sqrt(2.0 / 64)
    params["bfc2"] = np.zeros((num_classes, 1))
    
    return params
