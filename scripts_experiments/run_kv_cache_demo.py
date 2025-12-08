import time
import torch

from src.experiments.inference_advanced.kv_cache_generate import (
    load_experiment_model,
    naive_generate,
    kv_cache_generate_dummy,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, tokenizer = load_experiment_model(device=device)

    prompt = "hello world"
    max_new_tokens = 30

    # 1) Naive generate (tam sequence)
    t0 = time.time()
    out_naive = naive_generate(model, tokenizer, prompt, max_new_tokens, device=device)
    t1 = time.time()
    print(f"[naive] time: {t1 - t0:.3f}s")
    print(f"[naive] output: {out_naive[:120]}")

    # 2) Dummy KV cache generate
    t0 = time.time()
    out_kv = kv_cache_generate_dummy(model, tokenizer, prompt, max_new_tokens, device=device)
    t1 = time.time()
    print(f"[kv-cache-dummy] time: {t1 - t0:.3f}s")
    print(f"[kv-cache-dummy] output: {out_kv[:120]}")


if __name__ == "__main__":
    main()
