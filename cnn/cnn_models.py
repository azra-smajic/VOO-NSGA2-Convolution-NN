from dataclasses import dataclass
import gc
import logging
import math
import os
import random
from typing import Dict, List, Tuple

import numpy as np

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_strict_conv_algorithm_picker=false")

from tensorflow import keras
import tensorflow as tf

from cnn.cnn import MsperBatchLogger, build_cifar10_model, build_cnn_model


LOGGER = logging.getLogger("cnn_models")

N_BLOCKS_CHOICES = [2, 3, 4]
FILTERS_BASE_CHOICES = [16, 32, 64]
KERNEL_CHOICES = [3, 5]
BATCH_CHOICES = [32, 64, 128]
LR_MIN, LR_MAX = 1e-4, 3e-2
DROP_MIN, DROP_MAX = 0.0, 0.5

DATASET_OPTIONS = {
    "mnist": {
        "label": "MNIST",
        "model_label": "MNIST CNN (Conv2D + ReLU + MaxPooling + Dense)",
        "num_classes": 10,
        "input_shape": (28, 28, 1),
        "builder": build_cnn_model,
    },
    "cifar10": {
        "label": "CIFAR-10",
        "model_label": "CIFAR-10 CNN (dva Conv2D po bloku + BatchNorm + Dropout)",
        "num_classes": 10,
        "input_shape": (32, 32, 3),
        "builder": build_cifar10_model,
    },
}

OBJECTIVE_1_OPTIONS = {
    "1-val_accuracy": {
        "label": "1 - validation accuracy",
        "direction": "min",
    },
    "val_loss": {
        "label": "Validation loss",
        "direction": "min",
    },
}

OBJECTIVE_2_OPTIONS = {
    "ms_per_batch": {
        "label": "Milliseconds per batch",
        "direction": "min",
    },
    "param_count": {
        "label": "Number of parameters",
        "direction": "min",
    },
}

EXPERIMENT_CONFIG = {
    "dataset": "mnist",
    "train_fraction": 0.8,
    "val_fraction": 0.1,
    "test_fraction": 0.1,
    "split_seed": 1,
    "split_mode": "random",
    "search_epochs": 2,
    "final_epochs": 15,
    "objective_1": "1-val_accuracy",
    "objective_2": "ms_per_batch",
    "execution_device": "cpu",
}

DATA_CACHE: Dict[str, np.ndarray] = {}
SPLIT_CACHE: Dict[str, object] = {
    "config_key": None,
    "splits": None,
    "summary": None,
}

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
LOGGER.info("TensorFlow GPU devices dostupni u cnn_models: %s", gpus)


def _dataset_cache_key(name: str) -> str:
    return name.lower()


def _load_raw_dataset(name: str):
    key = _dataset_cache_key(name)
    if key in DATA_CACHE:
        return DATA_CACHE[key]

    if key == "mnist":
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = (x_train.astype("float32") / 255.0)[..., None]
        x_test = (x_test.astype("float32") / 255.0)[..., None]
        y_train = y_train.astype("int64")
        y_test = y_test.astype("int64")
    elif key == "cifar10":
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        y_train = y_train.squeeze().astype("int64")
        y_test = y_test.squeeze().astype("int64")
    else:
        raise ValueError(f"Nepoznat dataset: {name}")

    DATA_CACHE[key] = (x_train, y_train, x_test, y_test)
    return DATA_CACHE[key]


def _split_arrays(x_all, y_all, train_fraction: float, val_fraction: float, split_seed: int, split_mode: str):
    total = len(x_all)
    indices = np.arange(total)
    if split_mode == "random":
        rng = np.random.default_rng(split_seed)
        rng.shuffle(indices)

    train_count = int(total * train_fraction)
    val_count = int(total * val_fraction)
    train_idx = indices[:train_count]
    val_idx = indices[train_count:train_count + val_count]
    test_idx = indices[train_count + val_count:]

    return (
        (x_all[train_idx], y_all[train_idx]),
        (x_all[val_idx], y_all[val_idx]),
        (x_all[test_idx], y_all[test_idx]),
    )


def _build_summary(dataset_name: str, train_split, val_split, test_split):
    return {
        "dataset": dataset_name,
        "model_label": DATASET_OPTIONS[dataset_name]["model_label"],
        "input_shape": DATASET_OPTIONS[dataset_name]["input_shape"],
        "num_classes": DATASET_OPTIONS[dataset_name]["num_classes"],
        "train_instances": int(len(train_split[0])),
        "val_instances": int(len(val_split[0])),
        "test_instances": int(len(test_split[0])),
        "train_shape": tuple(train_split[0].shape),
        "val_shape": tuple(val_split[0].shape),
        "test_shape": tuple(test_split[0].shape),
    }


def _config_cache_key() -> Tuple[object, ...]:
    return (
        EXPERIMENT_CONFIG["dataset"],
        EXPERIMENT_CONFIG["train_fraction"],
        EXPERIMENT_CONFIG["val_fraction"],
        EXPERIMENT_CONFIG["test_fraction"],
        EXPERIMENT_CONFIG["split_seed"],
        EXPERIMENT_CONFIG["split_mode"],
    )


def _refresh_splits():
    key = _config_cache_key()
    if SPLIT_CACHE["config_key"] == key and SPLIT_CACHE["splits"] is not None:
        return

    dataset_name = EXPERIMENT_CONFIG["dataset"]
    train_fraction = EXPERIMENT_CONFIG["train_fraction"]
    val_fraction = EXPERIMENT_CONFIG["val_fraction"]
    test_fraction = EXPERIMENT_CONFIG["test_fraction"]
    split_seed = EXPERIMENT_CONFIG["split_seed"]
    split_mode = EXPERIMENT_CONFIG["split_mode"]

    if abs((train_fraction + val_fraction + test_fraction) - 1.0) > 1e-8:
        raise ValueError("Train/validation/test split mora imati zbir 1.0")

    x_train_raw, y_train_raw, x_test_raw, y_test_raw = _load_raw_dataset(dataset_name)
    x_all = np.concatenate([x_train_raw, x_test_raw], axis=0)
    y_all = np.concatenate([y_train_raw, y_test_raw], axis=0)

    train_split, val_split, test_split = _split_arrays(
        x_all,
        y_all,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        split_seed=split_seed,
        split_mode=split_mode,
    )

    SPLIT_CACHE["config_key"] = key
    SPLIT_CACHE["splits"] = {
        "train": train_split,
        "val": val_split,
        "test": test_split,
    }
    SPLIT_CACHE["summary"] = _build_summary(dataset_name, train_split, val_split, test_split)
    LOGGER.info("Eksperiment konfigurisan: %s", EXPERIMENT_CONFIG)


def configure_experiment(
    dataset: str,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    split_seed: int,
    split_mode: str,
    search_epochs: int,
    final_epochs: int,
    objective_1: str,
    objective_2: str,
    execution_device: str,
):
    if dataset not in DATASET_OPTIONS:
        raise ValueError(f"Nepodrzan dataset: {dataset}")
    if objective_1 not in OBJECTIVE_1_OPTIONS:
        raise ValueError(f"Nepodrzana objective_1 mjera: {objective_1}")
    if objective_2 not in OBJECTIVE_2_OPTIONS:
        raise ValueError(f"Nepodrzana objective_2 mjera: {objective_2}")
    if execution_device not in {"cpu", "gpu"}:
        raise ValueError(f"Nepodrzan execution_device: {execution_device}")
    if train_fraction <= 0 or val_fraction <= 0 or test_fraction <= 0:
        raise ValueError("Sve split vrijednosti moraju biti vece od nule.")

    EXPERIMENT_CONFIG.update(
        {
            "dataset": dataset,
            "train_fraction": float(train_fraction),
            "val_fraction": float(val_fraction),
            "test_fraction": float(test_fraction),
            "split_seed": int(split_seed),
            "split_mode": split_mode,
            "search_epochs": int(search_epochs),
            "final_epochs": int(final_epochs),
            "objective_1": objective_1,
            "objective_2": objective_2,
            "execution_device": execution_device,
        }
    )
    SPLIT_CACHE["config_key"] = None
    _refresh_splits()


def get_experiment_config():
    return dict(EXPERIMENT_CONFIG)


def get_dataset_summary():
    _refresh_splits()
    return dict(SPLIT_CACHE["summary"])


def get_objective_labels():
    return {
        "objective_1": OBJECTIVE_1_OPTIONS[EXPERIMENT_CONFIG["objective_1"]]["label"],
        "objective_2": OBJECTIVE_2_OPTIONS[EXPERIMENT_CONFIG["objective_2"]]["label"],
    }


def get_available_options():
    return {
        "datasets": {key: value["label"] for key, value in DATASET_OPTIONS.items()},
        "objective_1": {key: value["label"] for key, value in OBJECTIVE_1_OPTIONS.items()},
        "objective_2": {key: value["label"] for key, value in OBJECTIVE_2_OPTIONS.items()},
        "execution_devices": {
            "cpu": "CPU only",
            "gpu": "GPU if available",
        },
        "split_modes": {
            "random": "Random shuffle split",
            "ordered": "Ordered split (bez mjesanja)",
        },
    }


def get_runtime_info():
    requested = EXPERIMENT_CONFIG["execution_device"]
    gpu_available = bool(gpus)
    effective = "gpu" if requested == "gpu" and gpu_available else "cpu"
    return {
        "requested_device": requested,
        "gpu_available": gpu_available,
        "gpu_count": len(gpus),
        "effective_device": effective,
    }


def _is_gpu_runtime_error(exc: Exception) -> bool:
    message = str(exc).lower()
    gpu_markers = [
        "cudnn",
        "cuda",
        "no algorithm worked",
        "graph execution error",
        "failed to determine best cudnn convolution algorithm",
        "cudnn_status",
        "profiling failure on cudnn engine",
        "unknown cudnn status",
        "relu",
    ]
    return any(marker in message for marker in gpu_markers)


def _build_model_for_dataset(cfg):
    dataset_name = EXPERIMENT_CONFIG["dataset"]
    builder = DATASET_OPTIONS[dataset_name]["builder"]
    return builder(cfg["n_blocks"], cfg["filters_base"], cfg["kernel_size"], cfg["dropout"])


def _compile_model(cfg):
    model = _build_model_for_dataset(cfg)
    opt = keras.optimizers.Adam(learning_rate=cfg["lr"])
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=False,
    )
    return model


def _fit_model(cfg, x_train, y_train, validation_data=None, epochs=2, force_cpu=False):
    requested_device = EXPERIMENT_CONFIG["execution_device"]
    use_cpu = force_cpu or requested_device == "cpu" or not gpus
    with tf.device("/CPU:0" if use_cpu else "/device:GPU:0"):
        model = _compile_model(cfg)
        ms_cb = MsperBatchLogger(warmup_batches=10, max_batches=50)
        history = model.fit(
            x_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=cfg["batch_size"],
            callbacks=[ms_cb],
            verbose=0,
        )
    return model, history, ms_cb


def _get_current_splits():
    _refresh_splits()
    return SPLIT_CACHE["splits"]


def _objective_values(cfg, model, history, ms_cb):
    objective_1 = EXPERIMENT_CONFIG["objective_1"]
    objective_2 = EXPERIMENT_CONFIG["objective_2"]

    if objective_1 == "1-val_accuracy":
        val_acc = float(history.history["val_accuracy"][-1])
        obj1 = 1.0 - val_acc
    elif objective_1 == "val_loss":
        val_acc = float(history.history["val_accuracy"][-1])
        obj1 = float(history.history["val_loss"][-1])
    else:
        raise ValueError(f"Nepodrzan objective_1: {objective_1}")

    if objective_2 == "ms_per_batch":
        obj2 = float(ms_cb.ms_per_batch) if ms_cb.ms_per_batch is not None else float("inf")
    elif objective_2 == "param_count":
        obj2 = float(model.count_params())
    else:
        raise ValueError(f"Nepodrzan objective_2: {objective_2}")

    return obj1, obj2, val_acc


def evaluate_genome(genome, epochs=None, seed=1):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    if epochs is None:
        epochs = EXPERIMENT_CONFIG["search_epochs"]

    splits = _get_current_splits()
    x_train, y_train = splits["train"]
    x_val, y_val = splits["val"]

    cfg = decode(genome)
    LOGGER.info(
        "Evaluacija genoma pokrenuta sa config=%s, requested_device=%s, gpu_available=%s",
        cfg,
        EXPERIMENT_CONFIG["execution_device"],
        bool(gpus),
    )
    try:
        model, history, ms_cb = _fit_model(
            cfg,
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            force_cpu=False,
        )
    except Exception as exc:
        if EXPERIMENT_CONFIG["execution_device"] != "gpu":
            tf.keras.backend.clear_session()
            raise
        LOGGER.warning("GPU evaluacija pala, prelazim na CPU fallback. Razlog: %s", exc)
        tf.keras.backend.clear_session()
        gc.collect()
        model, history, ms_cb = _fit_model(
            cfg,
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            force_cpu=True,
        )

    f1, f2, val_acc = _objective_values(cfg, model, history, ms_cb)
    if not np.isfinite(val_acc) or val_acc < 0.5:
        tf.keras.backend.clear_session()
        return (1.0, 1e9)

    LOGGER.info("Evaluacija zavrsena: f1=%.4f, f2=%.4f, val_acc=%.4f", f1, f2, val_acc)
    tf.keras.backend.clear_session()
    gc.collect()
    return (f1, f2)


def pick_representatives(front):
    fastest = min(front, key=lambda ind: ind.f[1])
    best_acc = min(front, key=lambda ind: ind.f[0])

    obj2_vals = [ind.f[1] for ind in front]
    obj1_vals = [ind.f[0] for ind in front]
    obj2_min, obj2_max = min(obj2_vals), max(obj2_vals)
    obj1_min, obj1_max = min(obj1_vals), max(obj1_vals)

    def balanced_score(ind):
        obj2_n = 0.0 if obj2_max == obj2_min else (ind.f[1] - obj2_min) / (obj2_max - obj2_min)
        obj1_n = 0.0 if obj1_max == obj1_min else (ind.f[0] - obj1_min) / (obj1_max - obj1_min)
        return obj1_n + obj2_n

    balanced = min(front, key=balanced_score)

    reps = []
    for label, candidate in [("FASTEST", fastest), ("BEST_ACC", best_acc), ("BALANCED", balanced)]:
        if all(candidate is not existing for _, existing in reps):
            reps.append((label, candidate))
    return reps


def final_eval_on_test(ind, epochs_final=None):
    if epochs_final is None:
        epochs_final = EXPERIMENT_CONFIG["final_epochs"]

    label, candidate = ind
    cfg = decode(candidate.x)
    splits = _get_current_splits()
    x_train, y_train = splits["train"]
    x_val, y_val = splits["val"]
    x_test, y_test = splits["test"]

    x_all = np.concatenate([x_train, x_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)

    try:
        requested_device = EXPERIMENT_CONFIG["execution_device"]
        with tf.device("/CPU:0" if requested_device == "cpu" or not gpus else "/device:GPU:0"):
            model = _compile_model(cfg)
            ms_cb = MsperBatchLogger(warmup_batches=10, max_batches=50)
            model.fit(x_all, y_all, epochs=epochs_final, batch_size=cfg["batch_size"], callbacks=[ms_cb], verbose=0)
            test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    except Exception as exc:
        if EXPERIMENT_CONFIG["execution_device"] != "gpu":
            tf.keras.backend.clear_session()
            raise
        LOGGER.warning("GPU finalna evaluacija pala, prelazim na CPU fallback. Razlog: %s", exc)
        tf.keras.backend.clear_session()
        gc.collect()
        with tf.device("/CPU:0"):
            model = _compile_model(cfg)
            ms_cb = MsperBatchLogger(warmup_batches=10, max_batches=50)
            model.fit(x_all, y_all, epochs=epochs_final, batch_size=cfg["batch_size"], callbacks=[ms_cb], verbose=0)
            test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    ms_per_batch = float(ms_cb.ms_per_batch) if ms_cb.ms_per_batch is not None else float("inf")
    tf.keras.backend.clear_session()
    gc.collect()

    return (
        {
            "label": label,
            "dataset": EXPERIMENT_CONFIG["dataset"],
            **cfg,
        },
        float(test_acc),
        float(ms_per_batch),
    )


def random_genome():
    return [
        random.randint(0, len(N_BLOCKS_CHOICES) - 1),
        random.randint(0, len(FILTERS_BASE_CHOICES) - 1),
        random.randint(0, len(KERNEL_CHOICES) - 1),
        random.randint(0, len(BATCH_CHOICES) - 1),
        random.random(),
        random.random(),
    ]


def lerp(a, b, u):
    return a + (b - a) * u


def log_uniform(min_v, max_v, u):
    lo = math.log10(min_v)
    hi = math.log10(max_v)
    return 10 ** (lo + (hi - lo) * u)


def decode(genome):
    g0, g1, g2, g3, u_lr, u_drop = genome
    n_blocks = N_BLOCKS_CHOICES[g0]
    filters_base = FILTERS_BASE_CHOICES[g1]
    kernel_size = KERNEL_CHOICES[g2]
    batch_size = BATCH_CHOICES[g3]
    lr = log_uniform(LR_MIN, LR_MAX, u_lr)
    dropout = lerp(DROP_MIN, DROP_MAX, u_drop)
    return {
        "n_blocks": n_blocks,
        "filters_base": filters_base,
        "kernel_size": kernel_size,
        "batch_size": batch_size,
        "lr": lr,
        "dropout": dropout,
    }


@dataclass
class Individual:
    x: List[float]
    f: Tuple[float, float] = (0.0, 0.0)
    rank: int = 0
    crowding_distance: float = 0.0
