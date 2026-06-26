"""Part 10: decoding strategies that control randomness.

This chapter first generates text with the trained checkpoint if one exists, then
uses a tiny toy vocabulary to demonstrate argmax sampling, multinomial sampling,
and temperature scaling.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from common import (
    GPT_CONFIG_124M,
    GPTModel,
    generate_text_simple,
    get_tokenizer,
    text_to_token_ids,
    token_ids_to_text,
)


def load_model_for_decoding():
    """Load trained weights when available; otherwise fall back to a fresh model."""
    model = GPTModel(GPT_CONFIG_124M)
    checkpoint = Path("trained_gpt_model.pth")
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        print("Loaded trained_gpt_model.pth")
    else:
        torch.manual_seed(123)
        model = GPTModel(GPT_CONFIG_124M)
        print("No trained_gpt_model.pth found; using a freshly initialized model.")
    model.to("cpu")
    model.eval()
    return model


def softmax_with_temperature(logits, temperature):
    """Temperature < 1 sharpens probabilities; temperature > 1 flattens them."""
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)


def print_sampled_tokens(probas, inverse_vocab):
    """Sample repeatedly to make the probability distribution visible as counts."""
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item() for _ in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")


def main():
    # The model generation section uses the real GPT tokenizer and model.
    tokenizer = get_tokenizer()
    model = load_model_for_decoding()

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=25,
        context_size=GPT_CONFIG_124M["context_length"],
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    # The rest of the file uses a tiny vocabulary so sampling behavior is obvious.
    vocab = {
        "closer": 0,
        "every": 1,
        "effort": 2,
        "forward": 3,
        "inches": 4,
        "moves": 5,
        "pizza": 6,
        "toward": 7,
        "you": 8,
    }
    inverse_vocab = {v: k for k, v in vocab.items()}

    # These logits are artificial scores for the toy vocabulary.
    next_token_logits = torch.tensor(
        [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
    )

    # Argmax is deterministic: it always selects the largest probability.
    probas = torch.softmax(next_token_logits, dim=0)
    next_token_id = torch.argmax(probas).item()
    print("Argmax:", inverse_vocab[next_token_id])

    # Multinomial sampling is random but weighted by the softmax probabilities.
    torch.manual_seed(123)
    next_token_id = torch.multinomial(probas, num_samples=1).item()
    print("Multinomial sample:", inverse_vocab[next_token_id])
    print_sampled_tokens(probas, inverse_vocab)

    # Compare original, sharp, and flat probability distributions.
    temperatures = [1, 0.1, 5]
    scaled_probas = [softmax_with_temperature(next_token_logits, t) for t in temperatures]

    x = torch.arange(len(vocab))
    bar_width = 0.15
    fig, ax = plt.subplots(figsize=(5, 3))
    for i, temperature in enumerate(temperatures):
        ax.bar(
            x + i * bar_width,
            scaled_probas[i],
            bar_width,
            label=f"Temperature = {temperature}",
        )

    ax.set_ylabel("Probability")
    ax.set_xticks(x)
    ax.set_xticklabels(vocab.keys(), rotation=90)
    ax.legend()
    plt.tight_layout()
    plt.savefig("temperature-plot.pdf")
    plt.close(fig)
    print("Saved temperature plot to temperature-plot.pdf")


if __name__ == "__main__":
    main()
