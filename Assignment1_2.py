import numpy as np
import matplotlib.pyplot as plt

# 1. Define true values and model predictions

y_true = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

# Model's predicted probabilities
y_pred = np.array([
    [0.80, 0.10, 0.10],
    [0.20, 0.70, 0.10],
    [0.10, 0.20, 0.70]
])

# 2. Define MSE and Categorical Cross-Entropy

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def cce(y_true, y_pred):
    
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

mse_original = mse(y_true, y_pred)
cce_original = cce(y_true, y_pred)

print("Original Predictions: ")
print(y_pred)

print("Original MSE: ", mse_original)
print("Original CCE: ", cce_original)

# 3. Modify predictions slightly

y_pred_modified = np.array([
    [0.70, 0.20, 0.10],
    [0.20, 0.60, 0.20],
    [0.15, 0.20, 0.65]
])

mse_modified = mse(y_true, y_pred_modified)
cce_modified = cce(y_true, y_pred_modified)

print("Modified predictions:")
print(y_pred_modified)

print("Modified MSE:", mse_modified)
print("Modified CCE:", cce_modified)

# 4. Compare loss values

print("Change in losses: ")
print("MSE change:", mse_modified - mse_original)
print("CCE change:", cce_modified - cce_original)

# 5. Plot the loss values

loss_names = ["MSE", "CCE"]

original_losses = [mse_original, cce_original]
modified_losses = [mse_modified, cce_modified]

x = np.arange(len(loss_names))
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(x - width / 2, original_losses, width, label="Original")
plt.bar(x + width / 2, modified_losses, width, label="Modified")

plt.xticks(x, loss_names)
plt.ylabel("Loss")
plt.title("Comparison of MSE and Categorical Cross-Entropy")
plt.legend()
plt.tight_layout()
plt.show()