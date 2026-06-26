"""Part 7: generate text from output token logits.

This script is intentionally self-contained: it creates a fresh random model,
encodes "Hello, I am", generates six more tokens, and decodes the final IDs.
Running it as a separate process avoids reusing a trained `model` or stale `out`
from the notebook.
"""

import torch

from common import GPT_CONFIG_124M_ORIG, create_fresh_model, generate_text_simple, get_tokenizer


def main():
    # Use the same GPT-2 tokenizer as the notebook examples.
    tokenizer = get_tokenizer()

    # Part 7 uses the original 1024-token context length. The fixed seed makes the
    # random initial weights reproduce the same output each time.
    model = create_fresh_model(seed=123, cfg=GPT_CONFIG_124M_ORIG)

    # Convert the prompt text to token IDs before feeding it into the model.
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)

    # PyTorch models expect a batch dimension, so [tokens] becomes [1, tokens].
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    # Greedy decoding appends one highest-probability token per loop iteration.
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M_ORIG["context_length"],
    )
    print("Output:", out)
    print("Output length:", len(out[0]))

    # Convert the generated token IDs back into readable text.
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)


if __name__ == "__main__":
    main()
