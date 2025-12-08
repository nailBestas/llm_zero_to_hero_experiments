import numpy as np
import torch
import onnxruntime as ort

from src.data_pipeline.tokenizer_utils import load_tokenizer


def main():
    onnx_path = "models/mini_lm/mini_lm.onnx"
    tokenizer_path = "data/tokenizer/tokenizer.json"

    tokenizer = load_tokenizer(tokenizer_path)

    # ONNX Runtime session
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    prompt = "hello world"
    encoding = tokenizer.encode(prompt)
    ids = encoding.ids
    input_ids = np.array([ids], dtype=np.int64)

    # Tek adım forward (logits)
    outputs = sess.run([output_name], {input_name: input_ids})
    logits = outputs[0]  # numpy array, shape (1, seq_len, vocab_size)

    print("ONNX logits shape:", logits.shape)


if __name__ == "__main__":
    main()
