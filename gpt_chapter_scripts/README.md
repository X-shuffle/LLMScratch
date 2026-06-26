# GPT chapter scripts

These scripts were split from `LLM Pretraining.ipynb` starting at:

`## GPT ARCHITECTURE PART 7: GENERATING TEXT FROM OUTPUT TOKENS`

Run each chapter in a fresh Python process to avoid notebook state leaking between examples:

```bash
python part7_generate_text.py
python part8_evaluate_text_models.py
python part9_train_loop.py
python part10_decoding_strategies.py
```

`common.py` contains the shared model, tokenizer, dataloader, loss, and generation helpers.

`part9_train_loop.py` saves `trained_gpt_model.pth`. If that file exists, `part10_decoding_strategies.py` loads it; otherwise it uses a freshly initialized model.

Use the same Python environment as the notebook kernel, because these scripts need `torch`, `tiktoken`, and for plotting scripts, `matplotlib`.
