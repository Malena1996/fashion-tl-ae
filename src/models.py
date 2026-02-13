import tensorflow as tf
from tensorflow.keras import layers as L

def build_encoder(latent_dim=32):
    inp = tf.keras.Input(shape=(28, 28, 1))
    x = L.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inp)   # 14x14
    x = L.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)     # 7x7
    x = L.Flatten()(x)
    x = L.Dense(128, activation="relu")(x)
    z = L.Dense(latent_dim, name="z")(x)
    return tf.keras.Model(inp, z, name="encoder")

def build_decoder(latent_dim=32):
    inp = tf.keras.Input(shape=(latent_dim,))
    x = L.Dense(7 * 7 * 64, activation="relu")(inp)
    x = L.Reshape((7, 7, 64))(x)
    x = L.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)  # 14x14
    x = L.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)  # 28x28
    out = L.Conv2D(1, 3, padding="same", activation="sigmoid")(x)
    return tf.keras.Model(inp, out, name="decoder")

def build_autoencoder(latent_dim=32):
    encoder = build_encoder(latent_dim)
    decoder = build_decoder(latent_dim)
    inp = tf.keras.Input(shape=(28, 28, 1))
    z = encoder(inp)
    recon = decoder(z)
    ae = tf.keras.Model(inp, recon, name="autoencoder")
    return ae, encoder, decoder

def build_classifier_from_encoder(encoder, n_classes=10, dropout=0.2):
    inp = tf.keras.Input(shape=(28, 28, 1))
    z = encoder(inp)  # encoder congelado en Stage 2
    x = L.Dense(128, activation="relu")(z)
    x = L.Dropout(dropout)(x)
    x = L.Dense(64, activation="relu")(x)
    out = L.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="clf_from_encoder")
