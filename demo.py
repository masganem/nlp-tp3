"""Quick demo: train model on a small subset."""

from train import train_model


if __name__ == "__main__":
    print("Training model on subset...")
    train_model(num_epochs=8, batch_size=256, max_samples=2**18, lr=0.002)
