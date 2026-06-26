"""Part 9: train the small GPT model.

This is the longest-running chapter script. It loads the training text, trains a
fresh GPT model, saves a loss plot, and writes `trained_gpt_model.pth` so the
decoding chapter can optionally reuse the trained weights.
"""

import time

import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MaxNLocator

from common import (
    GPT_CONFIG_124M,
    create_fresh_model,
    create_train_val_loaders,
    get_tokenizer,
    load_verdict_text,
    train_model_simple,
)


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """Save the training/validation loss curve from the tracked losses."""
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # A second x-axis shows roughly how many tokens the model has seen.
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    plt.savefig("loss-plot.pdf")
    plt.close(fig)


def main():
    # Prepare the story dataset exactly once inside this process.
    text_data = load_verdict_text()
    train_loader, val_loader = create_train_val_loaders(text_data)
    tokenizer = get_tokenizer()
    # Use CUDA if available; otherwise this runs on CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = time.time()
    # Training must start from train mode because dropout is active during updates.
    model = create_fresh_model(seed=123, eval_mode=False)
    model.to(device)
    # AdamW is the optimizer used in the notebook for this small training run.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    # This mirrors the notebook settings. Reduce num_epochs for a faster smoke test.
    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # Save artifacts that later scripts can inspect without rerunning training.
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    torch.save(model.state_dict(), "trained_gpt_model.pth")
    print("Saved trained model to trained_gpt_model.pth")
    print("Saved loss plot to loss-plot.pdf")


if __name__ == "__main__":
    main()
