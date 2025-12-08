import time
import torch

from src.experiments.inference_advanced.speculative_decode import (
    load_big_and_small,
    greedy_generate,
    speculative_generate,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    big_model, small_model, tokenizer = load_big_and_small(device=device)

    prompt = "hello world"
    max_new_tokens = 30

    # 1) Normal greedy generate (büyük model)
    t0 = time.time()
    out_greedy = greedy_generate(big_model, tokenizer, prompt, max_new_tokens, device=device)
    t1 = time.time()
    print(f"[greedy] time: {t1 - t0:.3f}s")
    print(f"[greedy] output: {out_greedy[:120]}")

    # 2) Speculative generate (küçük + büyük model)
    t0 = time.time()
    out_spec = speculative_generate(
        big_model,
        small_model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        draft_steps=4,
        device=device,
    )
    t1 = time.time()
    print(f"[speculative] time: {t1 - t0:.3f}s")
    print(f"[speculative] output: {out_spec[:120]}")


if __name__ == "__main__":
    main()
