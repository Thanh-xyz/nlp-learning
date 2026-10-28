import numpy as np

def stable_softmax(x, axis=-1):
    max_vals = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - max_vals)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / sum_exp

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scale_factor = np.sqrt(d_k)
    
    if Q.ndim == 2:
        scores = np.matmul(Q, K.T) / scale_factor
    elif Q.ndim == 3:
        scores = np.matmul(Q, np.swapaxes(K, -1, -2)) / scale_factor
    else:
        scores = np.matmul(Q, np.swapaxes(K, -1, -2)) / scale_factor
        
    attention_weights = stable_softmax(scores, axis=-1)
    context = np.matmul(attention_weights, V)
    
    return scores, attention_weights, context

def compute_self_attention(X, W_Q, W_K, W_V):
    Q = np.matmul(X, W_Q)
    K = np.matmul(X, W_K)
    V = np.matmul(X, W_V)
    return scaled_dot_product_attention(Q, K, V)
