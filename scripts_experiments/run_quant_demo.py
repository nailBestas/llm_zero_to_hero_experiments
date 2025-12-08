import torch

from src.experiments.inference_advanced.quant_inference import (
    load_fp32,
    load_fp16,
    greedy_generate,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    prompt = "hello world"
    max_new_tokens = 30

    # FP32 model
    model32, tokenizer = load_fp32(device=device)
    out32, t32 = greedy_generate(model32, tokenizer, prompt, max_new_tokens, device=device)
    print(f"[fp32] time: {t32:.3f}s")
    print(f"[fp32] output: {out32[:120]}")

    # FP16 model (sadece GPU'da anlamlı)
    if device == "cuda":
        model16, tokenizer16 = load_fp16(device=device)
        out16, t16 = greedy_generate(model16, tokenizer16, prompt, max_new_tokens, device=device)
        print(f"[fp16] time: {t16:.3f}s")
        print(f"[fp16] output: {out16[:120]}")
    else:
        print("[fp16] skipped: CUDA not available")


if __name__ == "__main__":
    main()
