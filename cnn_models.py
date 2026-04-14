
from dataclasses import dataclass
import logging
import math
import os
import random
from typing import Tuple, List
import gc
import numpy as np

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_strict_conv_algorithm_picker=false")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow import keras
import numpy as np
import tensorflow as tf
from cnn import MsperBatchLogger, build_cnn_model, load_mnist

LOGGER = logging.getLogger("cnn_models")
FORCE_CPU = True

(X_TR, Y_TR), (X_VAL, Y_VAL), (X_TE, Y_TE) = load_mnist()

N_BLOCKS_CHOICES = [2, 3, 4] 
FILTERS_BASE_CHOICES = [16, 32, 64] 
KERNEL_CHOICES = [3, 5] 
BATCH_CHOICES = [32, 64, 128] 
LR_MIN, LR_MAX = 1e-4, 3e-2
DROP_MIN, DROP_MAX = 0.0, 0.5

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu,True)
LOGGER.info("FORCE_CPU=%s, TensorFlow GPU devices dostupni u cnn_models: %s", FORCE_CPU, gpus)


def _is_gpu_runtime_error(exc: Exception) -> bool:
    message = str(exc).lower()
    gpu_markers = [
        "cudnn",
        "cuda",
        "no algorithm worked",
        "graph execution error",
        "failed to determine best cudnn convolution algorithm",
        "cudnn_status",
    ]
    return any(marker in message for marker in gpu_markers)


def _compile_model(cfg):
    model = build_cnn_model(cfg["n_blocks"], cfg["filters_base"], cfg["kernel_size"], cfg["dropout"])
    opt = keras.optimizers.Adam(learning_rate=cfg["lr"])
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=False,
    )
    return model


def _fit_model(cfg, x_train, y_train, validation_data=None, epochs=2, force_cpu=False):
    use_cpu = FORCE_CPU or force_cpu or not gpus
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
            verbose=0
        )
    return history, ms_cb


def evaluate_genome(genome, epochs=2, seed=1):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    cfg = decode(genome)
    LOGGER.info("Evaluacija genoma pokrenuta sa config=%s", cfg)
    try:
        history, ms_cb = _fit_model(
            cfg,
            X_TR,
            Y_TR,
            validation_data=(X_VAL, Y_VAL),
            epochs=epochs,
            force_cpu=False,
        )
    except Exception as exc:
        if not _is_gpu_runtime_error(exc):
            tf.keras.backend.clear_session()
            raise
        LOGGER.warning("GPU evaluacija pala, prelazim na CPU fallback. Razlog: %s", exc)
        tf.keras.backend.clear_session()
        gc.collect()
        history, ms_cb = _fit_model(
            cfg,
            X_TR,
            Y_TR,
            validation_data=(X_VAL, Y_VAL),
            epochs=epochs,
            force_cpu=True,
        )

    val_acc = float(history.history["val_accuracy"][-1])
    ms_per_batch = float(ms_cb.ms_per_batch) if ms_cb.ms_per_batch is not None else float("inf")
    if not np.isfinite(val_acc) or val_acc < 0.5:
        tf.keras.backend.clear_session()
        return (1.0, 1e9) 
    f1 = 1.0 - val_acc
    f2 = ms_per_batch
    LOGGER.info("Evaluacija zavrsena: val_acc=%.4f, ms_per_batch=%.2f", val_acc, ms_per_batch)
    tf.keras.backend.clear_session()
    gc.collect()
    return (f1, f2)


def pick_representatives(front):
    # 1) fastest (min ms/batch)
    fastest = min(front, key=lambda ind: ind.f[1])

    # 2) best accuracy (min 1-acc)
    best_acc = min(front, key=lambda ind: ind.f[0])

    # 3) balanced: normalizovana suma (ms + error)
    ms_vals = [ind.f[1] for ind in front]
    acc_vals = [1.0 - ind.f[0] for ind in front]

    ms_min, ms_max = min(ms_vals), max(ms_vals)
    acc_min, acc_max = min(acc_vals), max(acc_vals)

    def balanced_score(ind):
        ms = ind.f[1]
        acc = 1.0 - ind.f[0]
        ms_n = 0.0 if ms_max == ms_min else (ms - ms_min) / (ms_max - ms_min)     # 0..1 (manje bolje)
        err_n = 0.0 if acc_max == acc_min else (acc_max - acc) / (acc_max - acc_min)  # 0..1 (manje bolje)
        return ms_n + err_n

    balanced = min(front, key=balanced_score)

    # vrati jedinstveno (da se ne ponove)
    reps = []
    for label, candidate in [("FASTEST", fastest), ("BEST_ACC", best_acc), ("BALANCED", balanced)]:
        if all(candidate is not existing for _, existing in reps):
            reps.append((label, candidate))
    return reps

def final_eval_on_test(ind, epochs_final=15):
    label, candidate = ind
    cfg = decode(candidate.x)

    # Train + Val spojeno 
    X_ALL = np.concatenate([X_TR, X_VAL], axis=0)
    Y_ALL = np.concatenate([Y_TR, Y_VAL], axis=0)
    try:
        with tf.device("/CPU:0" if FORCE_CPU or not gpus else "/device:GPU:0"):
            model = _compile_model(cfg)
            ms_cb = MsperBatchLogger(warmup_batches=10, max_batches=50)
            model.fit(X_ALL, Y_ALL, epochs=epochs_final, batch_size=cfg["batch_size"], callbacks=[ms_cb], verbose=0)
            test_loss, test_acc = model.evaluate(X_TE, Y_TE, verbose=0)
    except Exception as exc:
        if not _is_gpu_runtime_error(exc):
            tf.keras.backend.clear_session()
            raise
        LOGGER.warning("GPU finalna evaluacija pala, prelazim na CPU fallback. Razlog: %s", exc)
        tf.keras.backend.clear_session()
        gc.collect()
        with tf.device("/CPU:0"):
            model = _compile_model(cfg)
            ms_cb = MsperBatchLogger(warmup_batches=10, max_batches=50)
            model.fit(X_ALL, Y_ALL, epochs=epochs_final, batch_size=cfg["batch_size"], callbacks=[ms_cb], verbose=0)
            test_loss, test_acc = model.evaluate(X_TE, Y_TE, verbose=0)
    # vrijeme po batchu 
    ms_per_batch = float(ms_cb.ms_per_batch) if ms_cb.ms_per_batch is not None else float("inf")

    # očisti
    tf.keras.backend.clear_session()
    gc.collect()

    return (
        {
            "label": label,
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
        random.random(),  # u in [0,1] -> lr
        random.random(),  # u in [0,1] -> dropout
    ]
# linearna interpolacija između a i b sa faktorom u, gdje u=0 daje a, a u=1 daje b, a vrijednosti između 0 i 1 daju vrijednosti između a i b, npr ako je a=10, b=20, i u=0.5, dobijemo 15
def lerp(a, b, u):
    return a + (b - a) * u

# learning rate biramo log uniform distribucijom, jer želimo da imamo više varijacija u manjim vrijednostima (npr 1e-4, 1e-3) nego u većim vrijednostima (npr 1e-2), što je često slučaj kod hiperparametara poput learning rate-a
def log_uniform(min_v, max_v, u):
    # mapiraj u∈[0,1] na log10 skalu između min_v i max_v, jer nam je bitna relativna promjena learning rate-a, npr da li je 1e-3 deset puta veći od 1e-4, a ne da li je veći za 0.0009, što nije toliko bitno, i zato koristimo log uniform distribuciju
    # ovako dobijamo realnu šansu da imamo i vrijednosti blizu 1e-4 i blizu 1e-2, a ne da su sve vrijednosti blizu 1e-2, što bi bilo slučaj da koristimo uniform distribuciju
    # biramo logaritamski, jer je kod lr bitan faktor a ne razlika
    lo = math.log10(min_v)
    hi = math.log10(max_v)
    # radimo linearnu interpolaciju između lo i hi sa faktorom u, a zatim vraćamo 10 na tu vrijednost da bismo dobili broj u originalnoj skali
    return 10 ** (lo + (hi - lo) * u)
    # ovdje se vraćamo nazad iz log prostora u normalan broj, jer želimo da dobijemo realan learning rate koji možemo koristiti za treniranje modela

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
    x: List[float]  # genom koji predstavlja arhitekturu CNN-a i hiperparametre
    f: Tuple[float, float] = (0.0, 0.0) # (accuracy, ms_per_batch)
    rank: int = 0  # rang u populaciji, niže je bolje
    crowding_distance: float = 0.0  # crowding distance, veće je bolje
 
