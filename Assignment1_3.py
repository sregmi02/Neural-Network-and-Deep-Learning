import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Load the MNIST dataset

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)


# 2. Create a model

def create_model(optimizer):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# 3. Train model with Adam
adam_model = create_model(
    tf.keras.optimizers.Adam(learning_rate=0.001)
)

adam_history = adam_model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)


# 4. Train model with SGD

sgd_model = create_model(
    tf.keras.optimizers.SGD(learning_rate=0.01)
)

sgd_history = sgd_model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)


# 5. Evaluate both models on the test dataset

adam_loss, adam_accuracy = adam_model.evaluate(x_test, y_test, verbose=0)
sgd_loss, sgd_accuracy = sgd_model.evaluate(x_test, y_test, verbose=0)

print("\nTest Accuracy")
print("-------------------------")
print(f"Adam: {adam_accuracy:.4f}")
print(f"SGD : {sgd_accuracy:.4f}")


# 6. Plot training and validation accuracy

epochs = range(1, 11)

plt.figure(figsize=(10, 6))

# Adam
plt.plot(
    epochs,
    adam_history.history["accuracy"],
    marker="o",
    label="Adam - Training Accuracy"
)

plt.plot(
    epochs,
    adam_history.history["val_accuracy"],
    marker="o",
    linestyle="--",
    label="Adam - Validation Accuracy"
)

# SGD
plt.plot(
    epochs,
    sgd_history.history["accuracy"],
    marker="s",
    label="SGD - Training Accuracy"
)

plt.plot(
    epochs,
    sgd_history.history["val_accuracy"],
    marker="s",
    linestyle="--",
    label="SGD - Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MNIST Accuracy: Adam vs SGD")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()