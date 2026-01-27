from tensorflow.python.ops import math_ops
import numpy as np
import tensorflow as tf


class softmax_weight(tf.keras.constraints.Constraint):
    def __init__(self, feature_uniques):
        fu = np.array(feature_uniques, dtype=int).ravel()
        idxs = np.cumsum(np.concatenate([[0], fu]))
        self.feature_idxs = [(int(idxs[i]), int(idxs[i + 1])) for i in range(len(idxs) - 1)]

    def __call__(self, w):
        w_new = [math_ops.log(tf.nn.softmax(w[i:j, :], axis=0)) for i, j in self.feature_idxs]
        return tf.concat(w_new, axis=0)

    def get_config(self):
        return {"feature_idxs": self.feature_idxs}


def elr_loss(KL_LOSS):
    def loss(y_true, y_pred):
        return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred) + KL_LOSS
    return loss


def get_lr(input_dim, output_dim, constraint=None, KL_LOSS=0):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(
        output_dim, input_dim=input_dim, activation="softmax", kernel_constraint=constraint
    ))
    model.compile(loss=elr_loss(KL_LOSS), optimizer="adam", metrics=["accuracy"])
    return model


def sample(*arrays, n=None, frac=None, random_state=None):
    rng = np.random
    if isinstance(random_state, int):
        rng = rng.RandomState(random_state)
    elif isinstance(random_state, np.random.RandomState):
        rng = random_state

    arr0 = np.asarray(arrays[0])
    original_size = len(arr0)

    if n is None and frac is None:
        raise ValueError("Specify one of n or frac.")
    if n is None:
        n = int(original_size * frac)

    idxs = rng.choice(original_size, n, replace=False)
    if len(arrays) > 1:
        return tuple(np.asarray(a)[idxs] for a in arrays)
    return arr0[idxs]
