#!/usr/bin/env python3
"""
Isole la BatchNorm du bloc 3 (features.3.conv.1.1) hors de tout le training.

On genere UNE FOIS un tenseur d'entree x et un gradient de sortie dy, puis on
refait N fois forward + backward avec exactement les memes valeurs, et on
compte combien de resultats DIFFERENTS on obtient pour grad_input,
grad_weight, grad_bias et la sortie.

  - 1 seul hash   -> deterministe
  - plusieurs     -> le kernel lui-meme est non deterministe sur ce GPU,
                     peu importe les seeds, les donnees ou le DataLoader.

Cas testes : nn.BatchNorm2d avec cuDNN on/off, memoire contiguous/channels_last,
et une BatchNorm "manuelle" (ops PyTorch de base) comme candidate de remplacement.

Usage:
    python3 debug_bn_backward.py
    python3 debug_bn_backward.py --shape 128 96 56 48 --repeats 50
"""
import argparse
import hashlib
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn


def h(t):
    return hashlib.md5(
        t.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()[:12]


class ManualBN2d(nn.Module):
    """BatchNorm en mode train ecrite avec des ops de base (mean, **, sqrt)."""

    def __init__(self, c, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        var = ((x - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        xhat = (x - mean) / torch.sqrt(var + self.eps)
        return xhat * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


def run_case(name, make_bn, x0, dy0, w, b, repeats, memfmt):
    seen = {"output": set(), "grad_input": set(),
            "grad_weight": set(), "grad_bias": set()}
    for _ in range(repeats):
        bn = make_bn(x0.shape[1]).cuda().train()
        with torch.no_grad():
            bn.weight.copy_(w)
            bn.bias.copy_(b)
        x = x0.clone().to(memory_format=memfmt).requires_grad_(True)
        dy = dy0.clone().to(memory_format=memfmt)
        out = bn(x)
        out.backward(dy)
        torch.cuda.synchronize()
        seen["output"].add(h(out))
        seen["grad_input"].add(h(x.grad))
        seen["grad_weight"].add(h(bn.weight.grad))
        seen["grad_bias"].add(h(bn.bias.grad))

    counts = {k: len(v) for k, v in seen.items()}
    verdict = "DETERMINISTE" if all(c == 1 for c in counts.values()) \
        else "NON DETERMINISTE"
    print(f"[{verdict:>16}] {name:<45} "
          f"distincts sur {repeats} essais: {counts}", flush=True)
    return verdict == "DETERMINISTE"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", type=int, nargs=4, default=[128, 96, 56, 48],
                   metavar=("N", "C", "H", "W"),
                   help="Forme d'entree de la BN. Verifie-la dans "
                        "step_13.json -> block3_trace.input_shape")
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("Pas de GPU : ce test n'a de sens que sur CUDA.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    print(f"torch={torch.__version__} cuda={torch.version.cuda} "
          f"cudnn={torch.backends.cudnn.version()} "
          f"gpu={torch.cuda.get_device_name(0)}")
    print(f"shape={tuple(args.shape)} repeats={args.repeats}\n", flush=True)

    n, c, hh, ww = args.shape
    x0 = torch.randn(n, c, hh, ww, device="cuda")
    dy0 = torch.randn(n, c, hh, ww, device="cuda")
    w = torch.rand(c, device="cuda") + 0.5
    b = torch.randn(c, device="cuda") * 0.1

    results = {}
    for cudnn_on in (False, True):
        for fmt_name, fmt in (("contiguous", torch.contiguous_format),
                              ("channels_last", torch.channels_last)):
            torch.backends.cudnn.enabled = cudnn_on
            name = f"nn.BatchNorm2d cudnn={'on ' if cudnn_on else 'off'} {fmt_name}"
            results[name] = run_case(name, nn.BatchNorm2d, x0, dy0, w, b,
                                     args.repeats, fmt)

    torch.backends.cudnn.enabled = False
    for fmt_name, fmt in (("contiguous", torch.contiguous_format),
                          ("channels_last", torch.channels_last)):
        name = f"ManualBN2d (ops de base)  {fmt_name}"
        results[name] = run_case(name, ManualBN2d, x0, dy0, w, b,
                                 args.repeats, fmt)

    print("\n=== LECTURE ===")
    bn_ok = [v for k, v in results.items() if k.startswith("nn.")]
    manual_ok = [v for k, v in results.items() if k.startswith("Manual")]
    if all(bn_ok):
        print("nn.BatchNorm2d est deterministe EN ISOLATION : la cause n'est "
              "pas le kernel seul, regarde l'etat memoire/ordre d'execution "
              "dans le vrai training (envoie-moi ce resultat).")
    elif all(manual_ok):
        print("nn.BatchNorm2d est non deterministe dans au moins une config, "
              "alors que ManualBN2d est deterministe : remplacer les BN du "
              "modele par ManualBN2d est une piste de correction.")
    else:
        print("Meme ManualBN2d varie : la non-determinisme vient d'une "
              "reduction PyTorch plus generale sur ce GPU/cette version.")


if __name__ == "__main__":
    main()