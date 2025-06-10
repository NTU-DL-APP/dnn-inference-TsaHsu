import numpy as np
import json

# === Activation functions ===
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # 避免 overflow
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
    w_idx = 0  # 用來讀取 npz 權重順序

    for layer in model_arch:
        ltype = layer["class_name"]
        cfg = layer["config"]
        lname = cfg.get("name", "unknown")

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

# === 主程式示範使用 ===
if __name__ == "__main__":
    # 載入模型架構
    with open("fashion_mnist.json", "r") as f:
        model_json = json.load(f)
    model_arch = model_json["config"]["layers"]  # 重點：抓出 layers 陣列

    # 載入權重
    npz = np.load("fashion_mnist.npz")
    weights = [npz[key] for key in npz]

    # 載入測試資料（這裡只做簡單測試用，要和你訓練時格式一致）
    from tensorflow.keras.datasets import fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_test = x_test.astype("float32") / 255.0

    # 推論前 5 筆
    x_input = x_test[:5]
    predictions = nn_forward_h5(model_arch, weights.copy(), x_input)

    # 顯示結果
    print("預測機率：")
    print(predictions)
    print("預測類別：", np.argmax(predictions, axis=1))
    print("真實類別：", y_test[:5])
