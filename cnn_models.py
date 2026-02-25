
from dataclasses import dataclass
import math
import random
from typing import Tuple, List

from tensorflow import keras
import numpy as np
import tensorflow as tf
from cnn import MsperBatchLogger, build_cnn_model, load_mnist

(X_TR, Y_TR), (X_VAL, Y_VAL), (X_TE, Y_TE) = load_mnist()

N_BLOCKS_CHOICES = [2, 3, 4]
FILTERS_BASE_CHOICES = [16, 32, 64]
KERNEL_CHOICES = [3, 5]
BATCH_CHOICES = [32, 64, 128]
LR_MIN, LR_MAX = 1e-4, 3e-2
DROP_MIN, DROP_MAX = 0.0, 0.5

def evaluate_genome(genome, epochs=3, seed=1):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    cfg = decode(genome)
    model = build_cnn_model(cfg["n_blocks"], cfg["filters_base"], cfg["kernel_size"], cfg["dropout"])

    opt = keras.optimizers.Adam(learning_rate=cfg["lr"])
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    ms_cb = MsperBatchLogger(warmup_batches=5, max_batches=50)

    history = model.fit(
        X_TR, Y_TR,
        validation_data=(X_VAL, Y_VAL),
        epochs=epochs,
        batch_size=cfg["batch_size"],
        callbacks=[ms_cb],
        verbose=0
    )

    val_acc = float(history.history["val_accuracy"][-1])
    ms_per_batch = float(ms_cb.ms_per_batch) if ms_cb.ms_per_batch is not None else float("inf")

    f1 = 1.0 - val_acc
    f2 = ms_per_batch

    return (f1, f2)

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

# @dataclass
# class Genome:
#     n_blocks: int  # broj blokova u CNN-u, obično između 1 i 5
#     filters: int  # broj filtera u konvolucijskim slojevima, obično između 16 i 256
#     kernel_size: int  # veličina kernela u konvolucijskim slojevima, obično 3 ili 5
#     learning_rate: float  # stopa učenja, obično između 1e-4 i 1e-2
#     batch_size: int  # veličina batcha, obično između 16 i 128
#     dropout_rate: float  # stopa dropout-a, obično između 0.0 i 0.5

@dataclass
class Individual:
    x: List[float]  # genom koji predstavlja arhitekturu CNN-a i hiperparametre
    f: Tuple[float, float] = (0.0, 0.0) # (accuracy, ms_per_batch)
    rank: int = 0  # rang u populaciji, niže je bolje
    crowding_distance: float = 0.0  # crowding distance, veće je bolje
    # def __init__(self, x: Genome):
    #     self.x = x
    #     self.f = (0.0, 0.0)
    #     self.rank = 0
    #     self.crowding_distance = 0.0