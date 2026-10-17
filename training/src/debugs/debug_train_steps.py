"""Diagnostic : reproduit la boucle d'entrainement et hash chaque step."""
import argparse
import hashlib

import torch

from src.age_gender.dataset.get_dataloader import get_train_loader
from src.age_gender.train_multi_heads import load_model, set_seed
from src.utils.config import get_config
from src.utils.losses import Loss


def h(tensor):
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.md5(arr.tobytes()).hexdigest()[:8]


def params_hash(model):
    m = hashlib.md5()
    for p in model.parameters():
        m.update(p.detach().cpu().contiguous().numpy().tobytes())
    return m.hexdigest()[:8]


def main(config_file, seed, steps, use_dp, workers):
    config = get_config()
    config.set_config(config_file)
    if workers is not None:
        config.input.set_param("num_workers", workers)  # sinon: config.input.num_workers = workers

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} n_gpu={torch.cuda.device_count()} "
          f"dataparallel={use_dp} workers={config.input.num_workers}")

    train_loader = get_train_loader(config)
    model = load_model(config).to(device)
    heads_list = model.heads_list
    loss = Loss(config, heads_list)
    if use_dp:
        model = torch.nn.DataParallel(model).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.train.lr, betas=(config.train.beta1, 0.999)
    )  # adapte si ton optimizer est different

    model.train()
    for step, (images, gender, age, *_) in enumerate(train_loader):
        if step >= steps:
            break
        gender, age = gender.to(device), age.to(device)
        preds = model(images.to(device))
        label_list = [gender if e["name"] == "gender" else age for e in heads_list]
        cost = loss.compute(preds, label_list, epoch=0)
        model.zero_grad()
        cost.backward()
        optimizer.step()
        print(f"step {step:03d} data={h(images)} loss={cost.item():.8f} "
              f"params={params_hash(model)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config_file", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--no_dp", action="store_true")
    p.add_argument("--workers", type=int, default=None)
    a = p.parse_args()
    main(a.config_file, a.seed, a.steps, not a.no_dp, a.workers)