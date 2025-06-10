import numpy as np

# === Activation functions ===
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    x = np.array(x)
    if x.ndim == 1:
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)
    else:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)


# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    return x @ W + b

# === Forward pass ===
def nn_forward_h5(model_arch, weights, data):
    x = data
    w_idx = 0
    for layer in model_arch:
        ltype = layer["class_name"]
        cfg = layer["config"]

        if ltype == "Flatten":
            x = flatten(x)
        elif ltype == "Dense":
            W = weights[w_idx]
            b = weights[w_idx + 1]
            w_idx += 2
            x = dense(x, W, b)
            act = cfg.get("activation", "")
            if act == "relu":
                x = relu(x)
            elif act == "softmax":
                x = softmax(x)
    return x


def nn_inference(model_arch, weights, data):
    import numpy.lib.npyio
    print("==> weights type:", type(weights))
    
    # 修正點：如果是 npz 檔回傳的 NpzFile 類別 → 轉成 list
    if isinstance(weights, numpy.lib.npyio.NpzFile):
        weights = [weights[k] for k in sorted(weights.files)]
    
    return nn_forward_h5(model_arch, weights, data)

