import time

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def build_cnn_model(n_blocks, filters_base, kernel_size, dropout):
    model = keras.Sequential(name="mnist_cnn")
    model.add(layers.Input(shape=(28, 28, 1), name="input_mnist"))

    for i in range(n_blocks):
        filters = filters_base * (2 ** i)
        model.add(
            layers.Conv2D(
                filters,
                kernel_size=kernel_size,
                padding="same",
                name=f"conv2d_block_{i + 1}",
            )
        )
        model.add(layers.ReLU(name=f"relu_block_{i + 1}"))
        model.add(layers.MaxPooling2D((2, 2), name=f"pool_block_{i + 1}"))

    model.add(layers.Flatten(name="flatten_features"))
    model.add(layers.Dense(128, name="dense_hidden"))
    model.add(layers.ReLU(name="relu_hidden"))
    model.add(layers.Dropout(dropout, name="dropout_hidden"))
    model.add(layers.Dense(10, activation="softmax", name="classifier"))
    return model


def build_cifar10_model(n_blocks, filters_base, kernel_size, dropout):
    model = keras.Sequential(name="cifar10_cnn")
    model.add(layers.Input(shape=(32, 32, 3), name="input_cifar10"))

    for i in range(n_blocks):
        filters = filters_base * (2 ** i)

        model.add(
            layers.Conv2D(
                filters,
                kernel_size=kernel_size,
                padding="same",
                name=f"conv2d_block_{i + 1}_a",
            )
        )
        model.add(layers.BatchNormalization(name=f"batchnorm_block_{i + 1}_a"))
        model.add(layers.ReLU(name=f"relu_block_{i + 1}_a"))

        model.add(
            layers.Conv2D(
                filters,
                kernel_size=kernel_size,
                padding="same",
                name=f"conv2d_block_{i + 1}_b",
            )
        )
        model.add(layers.BatchNormalization(name=f"batchnorm_block_{i + 1}_b"))
        model.add(layers.ReLU(name=f"relu_block_{i + 1}_b"))

        model.add(layers.MaxPooling2D(pool_size=2, name=f"pool_block_{i + 1}"))
        model.add(layers.Dropout(dropout * 0.5, name=f"dropout_block_{i + 1}"))

    model.add(layers.Flatten(name="flatten_features"))
    model.add(layers.Dense(256, name="dense_hidden"))
    model.add(layers.ReLU(name="relu_hidden"))
    model.add(layers.Dropout(dropout, name="dropout_hidden"))
    model.add(layers.Dense(10, activation="softmax", name="classifier"))
    return model


class MsperBatchLogger(keras.callbacks.Callback):
    def __init__(self, warmup_batches=10, max_batches=50):
        super().__init__()
        self.warmup_batches = warmup_batches
        self.max_batches = max_batches
        self.batch_times = []
        self.batch_start = None
        self.ms_per_batch = None

    def on_train_batch_begin(self, batch, logs=None):
        if self.warmup_batches <= batch < self.warmup_batches + self.max_batches:
            self.batch_start = time.perf_counter()

    def on_train_batch_end(self, batch, logs=None):
        if self.warmup_batches <= batch < self.warmup_batches + self.max_batches:
            batch_time = (time.perf_counter() - self.batch_start) * 1000.0
            self.batch_times.append(batch_time)

    def on_epoch_end(self, epoch, logs=None):
        if self.batch_times:
            self.ms_per_batch = float(np.mean(self.batch_times))
