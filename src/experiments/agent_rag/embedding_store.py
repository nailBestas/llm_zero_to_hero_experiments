import json
from pathlib import Path
from typing import List, Tuple

import torch
from sentence_transformers import SentenceTransformer


class EmbeddingStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.texts: List[str] = []
        self.embeddings: torch.Tensor | None = None

    def build_from_corpus(self, corpus_path: str) -> None:
        path = Path(corpus_path)
        lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.texts = lines
        embs = self.model.encode(lines, convert_to_tensor=True, device=self.device, show_progress_bar=True)
        self.embeddings = embs / embs.norm(dim=-1, keepdim=True)

    def save(self, out_dir: str) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.embeddings, out / "embeddings.pt")
        (out / "texts.json").write_text(json.dumps(self.texts, ensure_ascii=False), encoding="utf-8")

    def load(self, out_dir: str) -> None:
        out = Path(out_dir)
        self.embeddings = torch.load(out / "embeddings.pt", map_location=self.device)
        self.texts = json.loads((out / "texts.json").read_text(encoding="utf-8"))

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        assert self.embeddings is not None
        q_emb = self.model.encode([query], convert_to_tensor=True, device=self.device)
        q_emb = q_emb / q_emb.norm(dim=-1, keepdim=True)
        scores = (self.embeddings @ q_emb.T).squeeze(-1)
        vals, idx = torch.topk(scores, k=min(top_k, scores.size(0)))
        results = []
        for s, i in zip(vals.tolist(), idx.tolist()):
            results.append((self.texts[i], float(s)))
        return results
