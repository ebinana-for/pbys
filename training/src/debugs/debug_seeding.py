"""Script de diagnostic pour isoler la source de non-reproductibilité."""
import argparse
import hashlib
import sys

import torch

from src.age_gender.dataset.get_dataloader import get_train_loader
from src.age_gender.train_multi_heads import load_model, set_seed
from src.utils.config import get_config


def h(tensor: torch.Tensor) -> str:
    """Hash déterministe du contenu d'un tenseur."""
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.md5(arr.tobytes()).hexdigest()[:10]


def main(config_file, seed):
    config = get_config()
    config.set_config(config_file)

    print(f"=== SEED = {seed} ===")
    set_seed(seed)

    # 1. Dataloader : hash des 3 premiers batches AVANT tout entrainement
    print("\n--- [1] Dataloader ---")
    train_loader = get_train_loader(config)
    it = iter(train_loader)
    for i in range(3):
        images, gender, age, *_ = next(it)
        print(f"batch {i}: images={h(images)} gender={h(gender)} age={h(age)}")

    # 2. Modele : hash des poids juste apres init (avant tout forward)
    print("\n--- [2] Model init ---")
    model = load_model(config)
    for name, param in list(model.named_parameters())[:3]:
        print(f"{name}: {h(param)}")

    # 3. Forward pass sur le 1er batch
    print("\n--- [3] Forward pass ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    images, gender, age, *_ = next(iter(train_loader))
    out = model(images.to(device))
    if isinstance(out, (list, tuple)):
        for i, o in enumerate(out):
            print(f"output head {i}: {h(o)}")
    else:
        print(f"output: {h(out)}")

    # 4. Backward pass : hash des gradients
    print("\n--- [4] Backward pass ---")
    loss = sum(o.sum() for o in out) if isinstance(out, (list, tuple)) else out.sum()
    loss.backward()
    for name, param in list(model.named_parameters())[:3]:
        if param.grad is not None:
            print(f"grad {name}: {h(param.grad)}")

    print("\n=== FIN DIAGNOSTIC ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.config_file, args.seed)