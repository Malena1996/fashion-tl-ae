import tensorflow as tf
from src.data import load_fashion_mnist
from src.models import build_classifier_from_encoder
from src.utils import set_seed, load_config, ensure_dir, save_json

def main():
    cfg = load_config("configs/stage2.yaml")
    set_seed(cfg["seed"])
    ensure_dir(cfg["out_dir"])

    train_ds, val_ds, test_ds = load_fashion_mnist(batch_size=cfg["batch_size"], seed=cfg["seed"])

    encoder = tf.keras.models.load_model(cfg["encoder_path"])
    encoder.trainable = False  # congelado

    clf = build_classifier_from_encoder(encoder, dropout=cfg["dropout"])
    clf.compile(
        optimizer=tf.keras.optimizers.Adam(cfg["lr"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    hist = clf.fit(train_ds, validation_data=val_ds, epochs=cfg["epochs"])
    test_loss, test_acc = clf.evaluate(test_ds)

    clf.save(f'{cfg["out_dir"]}/final_classifier.keras')
    save_json(f'{cfg["out_dir"]}/history.json', hist.history)
    save_json(f'{cfg["out_dir"]}/test_metrics.json', {"test_loss": float(test_loss), "test_acc": float(test_acc)})

if __name__ == "__main__":
    main()
