from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
import torchvision.transforms as T
from torchvision import models


def load_labels() -> list[str]:
    # Şimdilik gerçek ImageNet label dosyası kullanmamak için
    # 1000 adet "class_0", "class_1", ... üretelim.
    return [f"class_{i}" for i in range(1000)]



def load_model(device: str = "cuda") -> torch.nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval().to(device)
    return model


def preprocess_image(img_bgr, device: str = "cuda") -> torch.Tensor:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    tensor = transform(img_rgb).unsqueeze(0).to(device)  # [1, 3, 224, 224]
    return tensor


def predict_image(model, img_path: Path, labels: list[str], device: str = "cuda", topk: int = 5):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"Resim okunamadı: {img_path}")
        return

    x = preprocess_image(img_bgr, device=device)

    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    topk_vals, topk_idxs = torch.topk(probs, k=topk)
    print(f"\n=== {img_path.name} için tahminler ===")
    for i in range(topk):
        idx = topk_idxs[i].item()
        p = topk_vals[i].item()
        label = labels[idx] if idx < len(labels) else f"class_{idx}"
        print(f"{i+1}. {label:30s}  {p*100:5.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images-dir",
        type=str,
        default="images",
        help="Sınıflandırılacak resimlerin olduğu klasör",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    device = args.device
    print(f"Device: {device}")

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Resim klasörü bulunamadı: {images_dir}")

    img_files = [p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not img_files:
        raise FileNotFoundError(f"{images_dir} içinde .jpg/.jpeg/.png dosya bulunamadı.")

    labels = load_labels()
    model = load_model(device=device)

    for img_path in img_files:
        predict_image(model, img_path, labels, device=device, topk=5)


if __name__ == "__main__":
    main()
