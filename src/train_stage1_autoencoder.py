import tensorflow as tf
from src.data import load_fashion_mnist
from src.models import build_autoencoder
from src.utils import set_seed, load_config, ensure_dir, save_json

def main():
    cfg = load_config("configs/stage1.yaml")
    set_seed(cfg["seed"])
    ensure_dir(cfg["out_dir"])

    train_ds, val_ds, _ = load_fashion_mnist(batch_size=cfg["batch_size"], seed=cfg["seed"])

    ae, encoder, decoder = build_autoencoder(latent_dim=cfg["latent_dim"])
    ae.compile(
        optimizer=tf.keras.optimizers.Adam(cfg["lr"]),
        loss=tf.keras.losses.MeanSquaredError()
    )

    hist = ae.fit(
        train_ds.map(lambda x, y: (x, x)),
        validation_data=val_ds.map(lambda x, y: (x, x)),
        epochs=cfg["epochs"]
    )

    ae.save(f'{cfg["out_dir"]}/autoencoder.keras')
    encoder.save(f'{cfg["out_dir"]}/encoder.keras')
    decoder.save(f'{cfg["out_dir"]}/decoder.keras')
    save_json(f'{cfg["out_dir"]}/history.json', hist.history)

if __name__ == "__main__":
    main()
