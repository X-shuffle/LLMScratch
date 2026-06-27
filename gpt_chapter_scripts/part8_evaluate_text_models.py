"""Part 8: evaluate generative text models.

This script demonstrates the notebook's evaluation flow in a clean process:
generate from an untrained model, inspect token probabilities, compute
cross-entropy/perplexity, then measure train/validation loss on The Verdict.
"""

import torch

from common import (
    GPT_CONFIG_124M,
    calc_loss_loader,
    create_fresh_model,
    create_train_val_loaders,
    generate_text_simple,
    get_tokenizer,
    load_verdict_text,
    text_to_token_ids,
    token_ids_to_text,
)


def main():
    # Start from a fresh random model so results are not affected by training code.
    tokenizer = get_tokenizer()
    model = create_fresh_model(seed=123)

    # First, show that a random model can generate tokens but not meaningful text.
    start_context = "Every effort moves you"
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"],
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    # Tiny hand-written batch: inputs are token IDs, targets are the next token IDs.
    inputs = torch.tensor(
        [[16833, 3626, 6100], [40, 1107, 588]]
    )
    targets = torch.tensor(
        [[3626, 6100, 345], [1107, 588, 11311]]
    )

    # Model output shape is (batch, tokens, vocab_size).
    with torch.no_grad():
        logits = model(inputs)

    # Softmax converts logits to probabilities over all vocabulary entries.
    probas = torch.softmax(logits, dim=-1)
    print("Probability tensor shape:", probas.shape)

    # Argmax gives the model's most likely token at each position.
    predicted_token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    print("Token IDs:\n", predicted_token_ids)
    print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
    print(f"Outputs batch 1: {token_ids_to_text(predicted_token_ids[0].flatten(), tokenizer)}")

    # Pull out the probability assigned to the correct target token at each position.
    target_probas_1 = probas[0, [0, 1, 2], targets[0]]
    target_probas_2 = probas[1, [0, 1, 2], targets[1]]
    print("Text 1:", target_probas_1)
    print("Text 2:", target_probas_2)

    # Cross-entropy is negative average log probability of the correct tokens.
    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
    avg_log_probas = torch.mean(log_probas)
    neg_avg_log_probas = avg_log_probas * -1
    print("Log probabilities:", log_probas)
    print("Average log probability:", avg_log_probas)
    print("Negative average log probability:", neg_avg_log_probas)

    # PyTorch cross_entropy expects shape (N, classes), so flatten batch and time.
    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()
    print("Logits shape:", logits.shape)
    print("Targets shape:", targets.shape)
    print("Flattened logits:", logits_flat.shape)
    print("Flattened targets:", targets_flat.shape)

    # Perplexity is exp(cross_entropy); lower is better.
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    perplexity = torch.exp(loss)
    print("Cross-entropy loss:", loss)
    print("Perplexity:", perplexity)

    # Load the small story used as training/validation data in the notebook.
    text_data = load_verdict_text()
    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    print("First 100 characters:", text_data[:99])
    print("Last 100 characters:", text_data[-99:])
    print("Characters:", total_characters)
    print("Tokens:", total_tokens)

    # Build non-overlapping train/validation batches with shifted targets.
    train_loader, val_loader = create_train_val_loaders(text_data)
    print("Train loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)
        break
    print("\nValidation loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)
        break
    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    train_tokens = sum(input_batch.numel() for input_batch, _ in train_loader)
    val_tokens = sum(input_batch.numel() for input_batch, _ in val_loader)
    print("Training tokens:", train_tokens)
    print("Validation tokens:", val_tokens)
    print("All tokens:", train_tokens + val_tokens)

    # Evaluate the untrained model before any optimization has happened.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device use:", device)
    model.to(device)
    torch.manual_seed(123)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)

    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)


if __name__ == "__main__":
    main()
