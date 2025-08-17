import numpy as np
from keras import layers, datasets, Sequential
import os

# Hard-disable GPU & XLA before TF initializes
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
# Reduce TensorFlow logging noise (0=all, 1=warnings, 2=errors only)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress TF info logs


def build_model():
    model = Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def load_data():
    # Loads MNIST, normalizes pixel values to [0,1] and adds a channel axis.
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    # scale to [0,1] and add channel dim
    x_train = (x_train.astype("float32") / 255.0)[..., None]  # (N,28,28,1)
    x_test = (x_test.astype("float32") / 255.0)[..., None]
    return (x_train, y_train), (x_test, y_test)


def main():
    # Load data, build the model, train, evaluate, and save the model file.
    (x_train, y_train), (x_test, y_test) = load_data()
    model = build_model()

    # Train for a few epochs. validation_split keeps 10% of training data
    # for quick validation during training. Verbose=2 prints one line per epoch.
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=2)

    # Evaluate on the held-out test set and print accuracy.
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    # Save the trained model to a single Keras file next to app.py so the
    # inference app can load it later.
    model.save("model.keras")  # single file, lives next to app.py
    print("Saved: model.keras")


if __name__ == "__main__":
    main()
