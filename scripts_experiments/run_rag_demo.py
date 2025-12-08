import os
import torch

from src.experiments.agent_rag.rag_pipeline import build_index, answer_with_rag


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    corpus_path = "data/rag_corpus.txt"
    index_dir = "data/rag_index"

    if not os.path.exists(index_dir):
        print("Index yok, önce corpus'tan index inşa ediliyor...")
        build_index(corpus_path, index_dir)

    query = "Bu notlarda RAG ile ilgili ne anlatılıyor?"
    answer = answer_with_rag(query, index_dir=index_dir, max_new_tokens=80, device=device)
    print("\n=== RAG cevabı ===")
    print(answer)


if __name__ == "__main__":
    main()
