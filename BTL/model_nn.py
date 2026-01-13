import numpy as np
import matplotlib.pyplot as plt


def relu(Z):
    A = np.maximum(0,Z)    
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0    
    return dZ




def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)           

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))     
    return parameters

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    
    return Z, cache


def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2

    # Hidden layers (ReLU)
    for l in range(1, L):
        A_prev = A
        Z, linear_cache = linear_forward(
            A_prev, parameters['W' + str(l)], parameters['b' + str(l)]
        )
        A, activation_cache = relu(Z)
        caches.append((linear_cache, activation_cache))

    # Last layer: LINEAR ONLY (logits)
    ZL, linear_cache = linear_forward(
        A, parameters['W' + str(L)], parameters['b' + str(L)]
    )
    caches.append((linear_cache, None))

    return ZL, caches



def compute_cost(Z, Y):
    """
    Z: logits, shape (num_classes, m)
    Y: integer labels, shape (m,)
    """
    m = Y.shape[0] if len(Y.shape) == 1 else Y.shape[1]
    Y = Y.flatten().astype(int)

    Z_shift = Z - np.max(Z, axis=0, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(Z_shift), axis=0, keepdims=True))
    log_probs = Z_shift - log_sum_exp

    loss = -np.mean(log_probs[Y, np.arange(m)])
    return loss



def linear_backward(dZ, cache):

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def L_model_backward(Z, Y, caches):
    grads = {}
    L = len(caches)
    m = Y.shape[0] if len(Y.shape) == 1 else Y.shape[1]
    Y = Y.flatten().astype(int)

    # Softmax
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    A = expZ / np.sum(expZ, axis=0, keepdims=True)

    # dZ = softmax - one_hot
    dZ = A
    dZ[Y, np.arange(m)] -= 1
    dZ /= m

    # Last layer
    linear_cache = caches[-1][0]
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db
    grads["dA" + str(L - 1)] = dA_prev

    # Hidden layers (ReLU)
    for l in reversed(range(L - 1)):
        linear_cache, activation_cache = caches[l]
        dZ = relu_backward(grads["dA" + str(l + 1)], activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db
        grads["dA" + str(l)] = dA_prev

    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 

    for l in range(1, L + 1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
    return parameters

def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations=1000, print_cost=False):
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters_deep(layers_dims)
    
    for i in range(0, num_iterations):
        # Forward
        Z, caches = L_model_forward(X, parameters)

        # Cost
        cost = compute_cost(Z, Y)

        # Backward
        grads = L_model_backward(Z, Y, caches)

        
        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 10 == 0:
            print("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)
            
    # Plot cost
    if print_cost and len(costs) > 0:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    
    return parameters

def predict(X, y, parameters):
    Z, _ = L_model_forward(X, parameters)
    preds = np.argmax(Z, axis=0)

    y = y.flatten()
    acc = np.mean(preds == y)
    print("Accuracy:", acc)

    return preds, acc


