import torch
import torch.onnx as onnx

from src.inference.generation import load_model_and_tokenizer


def export_onnx_model(output_path: str = "models/mini_lm/mini_lm.onnx", device: str = "cpu"):
    model, tokenizer = load_model_and_tokenizer(device=device)
    model.eval()

    # Örnek giriş (1, seq_len)
    example_text = "hello world"
    encoding = tokenizer.encode(example_text)
    ids = encoding.ids
    example_input = torch.tensor([ids], dtype=torch.long, device=device)

    # ONNX export
    onnx.export(
        model,
        example_input,
        output_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"},
        },
        opset_version=17,
    )
    print(f"Exported ONNX model to {output_path}")


if __name__ == "__main__":
    device = "cpu"  # export için CPU yeterli
    export_onnx_model(device=device)
