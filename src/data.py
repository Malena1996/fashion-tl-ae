import numpy as np
import tensorflow as tf

def load_fashion_mnist(batch_size=256, seed=42):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Normalizar imágenes
    x_train = (x_train.astype("float32") / 255.0)[..., None]
    x_test  = (x_test.astype("float32") / 255.0)[..., None]

    # Crear validación (10%)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x_train))
    val_size = int(0.1 * len(x_train))
    val_idx, tr_idx = idx[:val_size], idx[val_size:]

    x_val, y_val = x_train[val_idx], y_train[val_idx]
    x_train, y_train = x_train[tr_idx], y_train[tr_idx]

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(20000, seed=seed).batch(batch_size)
    val_ds   = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
    test_ds  = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    return train_ds, val_ds, test_ds
