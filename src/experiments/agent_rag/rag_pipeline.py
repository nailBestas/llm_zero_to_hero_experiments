from typing import List
from src.experiments.agent_rag.embedding_store import EmbeddingStore
from src.inference.generation import load_model_and_tokenizer, generate



def build_index(corpus_path: str, index_dir: str) -> None:
    store = EmbeddingStore()
    store.build_from_corpus(corpus_path)
    store.save(index_dir)


def answer_with_rag(
    query: str,
    index_dir: str,
    max_new_tokens: int = 64,
    device: str = "cuda",
) -> str:
    store = EmbeddingStore()
    store.load(index_dir)
    retrieved: List[tuple[str, float]] = store.search(query, top_k=3)
    context = "\n\n".join([t for t, _ in retrieved])

    model, tokenizer = load_model_and_tokenizer(device=device)
    prompt = f"Soru: {query}\n\nİlgili notlar:\n{context}\n\nCevap:"
    out = generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens, device=device)

    return out
