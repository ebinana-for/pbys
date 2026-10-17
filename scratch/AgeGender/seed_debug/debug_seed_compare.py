#!/usr/bin/env python3
"""Diagnostic de divergence entre deux runs.

Modes :
    --probe-only : initialisation et trois premiers batches
    --one-step   : une mise à jour instrumentée
    --scan-steps : plusieurs mises à jour instrumentées

Sans mode explicite : lance deux runs de chaque mode et les compare.
Avec --compare RUN_A RUN_B : compare les scans existants sans relancer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

# Configuration avant l'import de torch et l'initialisation CUDA.
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

# Test d'isolation : désactive cuDNN, sans désactiver CUDA.
torch.backends.cudnn.enabled = False

ROOT = Path(__file__).resolve().parent
TRAINING_DIR = ROOT / "source_code" / "training"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def apply_deterministic_settings(strict: bool = False):
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if strict:
        torch.use_deterministic_algorithms(True, warn_only=False)

    if torch.backends.cudnn.enabled:
        raise RuntimeError("cuDNN est encore actif")

    print(
        f"[debug] cudnn_enabled={torch.backends.cudnn.enabled} "
        f"deterministic={torch.are_deterministic_algorithms_enabled()} "
        f"warn_only={torch.is_deterministic_algorithms_warn_only_enabled()}",
        flush=True,
    )


def maybe_wrap_model(model, disable_dataparallel: bool = False):
    if disable_dataparallel:
        print("[debug] DataParallel désactivé", flush=True)
        return model
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(
            f"[debug] DataParallel : {torch.cuda.device_count()} GPU",
            flush=True,
        )
        return torch.nn.DataParallel(model)
    return model


def tensor_hash(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().contiguous().numpy()
    return hashlib.md5(np.asarray(x).tobytes()).hexdigest()[:16]


def recursive_hash(x):
    h = hashlib.md5()
    if isinstance(x, torch.Tensor):
        h.update(x.detach().cpu().contiguous().numpy().tobytes())
    elif isinstance(x, np.ndarray):
        h.update(x.tobytes())
    elif isinstance(x, (list, tuple)):
        for item in x:
            h.update(recursive_hash(item).encode())
    elif isinstance(x, dict):
        for k in sorted(x.keys(), key=lambda key: (type(key).__name__, str(key))):
            h.update(str(k).encode())
            h.update(recursive_hash(x[k]).encode())
    else:
        h.update(str(x).encode())
    return h.hexdigest()[:16]


def model_hash(model):
    h = hashlib.md5()
    for p in model.parameters():
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def dump_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def read_json(path: Path):
    with path.open() as f:
        return json.load(f)


def load_config(config_path):
    sys.path.insert(0, str(TRAINING_DIR))
    from src.utils.config import get_config

    cfg = get_config()
    cfg.set_config(str(config_path))
    seed = getattr(cfg.general, "seed", 42) if hasattr(cfg, "general") else 42
    return cfg, seed


def probe_run(
    config_path, out_dir,
    strict_deterministic=False, disable_dataparallel=False,
):
    cfg, seed = load_config(config_path)
    from src.age_gender.dataset.get_dataloader import get_train_loader
    from src.age_gender.train_multi_heads import load_model

    set_seed(seed)
    apply_deterministic_settings(strict_deterministic)
    train_loader = get_train_loader(cfg)

    batches = []
    for i, batch in enumerate(train_loader):
        if i >= 3:
            break
        images, gender, age, *rest = batch
        batches.append({
            "batch_idx": i,
            "images_hash": tensor_hash(images),
            "gender_hash": tensor_hash(gender),
            "age_hash": tensor_hash(age),
            "image_shape": list(images.shape),
            "label_shape": list(gender.shape),
        })

    model = maybe_wrap_model(load_model(cfg), disable_dataparallel)
    payload = {
        "seed": seed,
        "strict_deterministic": strict_deterministic,
        "disable_dataparallel": disable_dataparallel,
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "batches": batches,
        "model_hash": model_hash(model),
        "state_hash": recursive_hash(model.state_dict()),
        "num_params": sum(p.numel() for p in model.parameters()),
    }
    dump_json(Path(out_dir) / "probe.json", payload)
    print(f"[probe] écrit : {out_dir}/probe.json", flush=True)

def prepare_training(config_path, strict_deterministic, disable_dataparallel):
    cfg, seed = load_config(config_path)
    from src.age_gender.dataset.get_dataloader import get_train_loader
    from src.age_gender.train_multi_heads import load_model
    from src.age_gender.train_multi_heads import set_seed as training_set_seed
    from src.utils.losses import Loss

    # Conserve la fonction de seed utilisée par ton training.
    training_set_seed(seed)
    apply_deterministic_settings(strict_deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = get_train_loader(cfg)
    base_model = load_model(cfg)
    heads_list = base_model.heads_list
    model = maybe_wrap_model(base_model, disable_dataparallel).to(device)
    loss = Loss(cfg, heads_list)

    # Conserve Adam et les paramètres de ton script de diagnostic.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.lr,
        betas=(cfg.train.beta1, 0.999),
    )
    return seed, device, train_loader, model, heads_list, loss, optimizer

def trace_block3(model):
    base = (
        model.module
        if isinstance(model, torch.nn.DataParallel)
        else model
    )
    bn = base.features[3].conv[1][1]

    step = getattr(trace_block3, "current_step", None)
    capture_dir = getattr(trace_block3, "capture_dir", None)

    if step is None or capture_dir is None:
        raise RuntimeError(
            "Le scan doit fournir current_step et capture_dir."
        )

    capture_dir = Path(capture_dir)
    capture_dir.mkdir(parents=True, exist_ok=True)
    capture_path = capture_dir / f"bn_step_{step}.pt"

    trace = {
        "module": "features.3.conv.1.1",
        "forward": {},
        "backward": {},
    }
    handles = []
    capture = {}

    def save_if_complete():
        required = {
            "x", "state", "output", "grad_output", "grad_input"
        }
        if required.issubset(capture):
            torch.save(capture, capture_path)

    def pre_hook(module, inputs):
        x = inputs[0]

        # Une seule copie pour le hash et la sauvegarde.
        snapshot = x.detach().cpu().clone()
        trace["forward"]["input"] = tensor_hash(snapshot)

        capture.update({
            "step": step,
            "x": snapshot,
            "x_stride": list(x.stride()),
            "x_storage_offset": x.storage_offset(),
            "state": {
                k: v.detach().cpu().clone()
                for k, v in module.state_dict().items()
            },
            "num_features": module.num_features,
            "eps": module.eps,
            "momentum": module.momentum,
            "affine": module.affine,
            "track_running_stats": module.track_running_stats,
            "training": module.training,
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "torch_version": str(torch.__version__),
        })

        if x.requires_grad:
            def input_grad_hook(grad):
                # Même copie pour le hash JSON et le fichier .pt.
                snapshot = grad.detach().cpu().clone()
                capture["grad_input"] = snapshot
                trace["backward"]["input"] = tensor_hash(snapshot)

                save_if_complete()
                return None

            handles.append(x.register_hook(input_grad_hook))

    def forward_hook(module, inputs, output):
        # Capture avant l'activation inplace suivante.
        snapshot = output.detach().cpu().clone()
        capture["output"] = snapshot
        trace["forward"]["output"] = tensor_hash(snapshot)

        if output.requires_grad:
            def output_grad_hook(grad):
                # Même copie pour le hash JSON et le fichier .pt.
                snapshot = grad.detach().cpu().clone()
                capture["grad_output"] = snapshot
                capture["grad_output_stride"] = list(grad.stride())
                trace["backward"]["output"] = tensor_hash(snapshot)

                save_if_complete()
                return None

            handles.append(output.register_hook(output_grad_hook))

    handles.append(bn.register_forward_pre_hook(pre_hook))
    handles.append(bn.register_forward_hook(forward_hook))

    return trace, handles

def scan_steps_run(
    config_path, out_dir, max_steps=100,
    strict_deterministic=False, disable_dataparallel=False,
):
    if max_steps < 1:
        raise ValueError("--max-steps doit être supérieur à zéro")

    out_path = Path(out_dir)
    capture_dir = out_path / "bn_cases"

    if out_path.exists() and any(out_path.glob("step_*.json")):
        raise FileExistsError(
            f"{out_dir} contient déjà un scan. "
            "Utilise un nouveau dossier."
        )

    if capture_dir.exists() and any(capture_dir.iterdir()):
        raise FileExistsError(
            f"{capture_dir} contient déjà des captures. "
            "Utilise un nouveau dossier."
        )

    out_path.mkdir(parents=True, exist_ok=True)
    capture_dir.mkdir(parents=True, exist_ok=True)

    context = prepare_training(
        config_path, strict_deterministic, disable_dataparallel
    )
    seed, device, loader, model, heads, loss, optimizer = context

    trace_block3.capture_dir = capture_dir
    completed = 0

    try:
        for step_idx, batch in enumerate(loader):
            if step_idx >= max_steps:
                break

            trace_block3.current_step = step_idx

            payload = instrumented_step(
                batch, step_idx, seed, device,
                model, heads, loss, optimizer,
                strict_deterministic, disable_dataparallel,
            )

            capture_path = capture_dir / f"bn_step_{step_idx}.pt"
            if not capture_path.is_file():
                raise RuntimeError(
                    f"Capture BatchNorm incomplète au step {step_idx}."
                )

            json_path = out_path / f"step_{step_idx}.json"
            dump_json(json_path, payload)
            completed += 1

            print(
                f"[scan] step={step_idx} "
                f"JSON={json_path} capture={capture_path}",
                flush=True,
            )

    finally:
        for attribute in ("current_step", "capture_dir"):
            if hasattr(trace_block3, attribute):
                delattr(trace_block3, attribute)

    print(
        f"[scan] terminé : {completed} steps capturés dans {out_dir}",
        flush=True,
    )

def replay_bn(path, repeats=100):
    if repeats < 1:
        raise ValueError("repeats doit être supérieur à zéro")

    case = torch.load(path, map_location="cpu", weights_only=True)

    required = {"x", "state", "output", "grad_output", "grad_input"}
    missing = required - set(case)
    if missing:
        raise ValueError(
            f"Ancienne capture ou capture incomplète : {sorted(missing)}. "
            "Il faut une capture produite par trace_block3() corrigée."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("Ce replay nécessite CUDA")

    torch.backends.cudnn.enabled = case["cudnn_enabled"]
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)

    device = torch.device("cuda")
    bn = torch.nn.BatchNorm2d(
        case["num_features"],
        eps=case["eps"],
        momentum=case["momentum"],
        affine=case["affine"],
        track_running_stats=case["track_running_stats"],
    ).to(device=device, dtype=case["x"].dtype)
    bn.train(case["training"])

    initial_state = {
        k: v.to(device) for k, v in case["state"].items()
    }

    def restore_tensor(values, strides):
        """Préserve les strides non chevauchants, sinon signale le fallback."""
        shape = tuple(values.shape)
        strides = tuple(strides)

        # Vérification conservative avant une copie dans empty_strided.
        dimensions = sorted(
            (stride, size)
            for size, stride in zip(shape, strides)
            if size > 1
        )
        span = 1
        non_overlapping = True
        for stride, size in dimensions:
            if stride < span:
                non_overlapping = False
                break
            span += (size - 1) * stride

        if non_overlapping:
            result = torch.empty_strided(
                shape, strides, dtype=values.dtype, device=device
            )
            result.copy_(values)
            return result

        print(
            "[limite] Strides chevauchants : copie contiguë utilisée.",
            flush=True,
        )
        return values.to(device).contiguous()

    grad_output = restore_tensor(
        case["grad_output"], case["grad_output_stride"]
    )
    original_output = case["output"]
    original_grad_input = case["grad_input"]

    print(
        f"[replay] cas step={case['step']} "
        f"cuDNN={torch.backends.cudnn.enabled}"
    )
    print(
        f"[layout] x strides={case['x_stride']} "
        f"offset original={case['x_storage_offset']}"
    )
    print(
        "[limite] Le replay recrée le forward et ses intermédiaires ; "
        "il ne restaure pas le graphe original."
    )

    reference = None
    unstable = False
    all_outputs_match = True
    all_input_grads_match = True

    for iteration in range(repeats):
        bn.load_state_dict(initial_state)
        bn.zero_grad(set_to_none=True)

        x = restore_tensor(case["x"], case["x_stride"])
        x.requires_grad_(True)

        y = bn(x)
        y.backward(grad_output)

        output_cpu = y.detach().cpu()
        input_grad_cpu = x.grad.detach().cpu()

        output_matches = torch.equal(output_cpu, original_output)
        input_grad_matches = torch.equal(
            input_grad_cpu, original_grad_input
        )
        all_outputs_match &= output_matches
        all_input_grads_match &= input_grad_matches

        result = {
            "output": tensor_hash(y),
            "grad_input": tensor_hash(x.grad),
            "grad_weight": (
                tensor_hash(bn.weight.grad) if bn.affine else None
            ),
            "grad_bias": (
                tensor_hash(bn.bias.grad) if bn.affine else None
            ),
        }

        if iteration == 0:
            print("=== COMPARAISON AU TRAINING ORIGINAL ===")
            print("Sortie identique :", output_matches)
            print("Gradient d'entrée identique :", input_grad_matches)
            print(
                "Écart absolu maximal du gradient d'entrée :",
                (input_grad_cpu - original_grad_input).abs().max().item(),
            )
            print(
                "Éléments du gradient d'entrée différents :",
                (input_grad_cpu != original_grad_input).sum().item(),
            )
            reference = result

        elif result != reference:
            unstable = True
            print(f"REPLAY INSTABLE à la répétition {iteration}")
            print("Référence :", reference)
            print("Résultat  :", result)
            break

    print("=== BILAN ===")
    if unstable:
        print("Le cas isolé produit des résultats différents entre répétitions.")
    else:
        print(f"Replay stable sur {repeats} répétitions.")

    print("Toutes les sorties correspondent au training :", all_outputs_match)
    print(
        "Tous les gradients d'entrée correspondent au training :",
        all_input_grads_match,
    )
    
def instrumented_step(
    batch, step_idx, seed, device, model, heads_list, loss, optimizer,
    strict_deterministic, disable_dataparallel,
):
    images, gender, age, *rest = batch
    gender = gender.to(device)
    age = age.to(device)
    images_device = images.to(device)
    model.zero_grad()

    # Captures avant le forward.
    before_hash = model_hash(model)
    state_before = recursive_hash(model.state_dict())
    optimizer_before = recursive_hash(optimizer.state_dict())

    rng_cpu_before = tensor_hash(torch.get_rng_state())
    rng_cuda_before = (
        tensor_hash(torch.cuda.get_rng_state(device))
        if device.type == "cuda" else None
    )
    rng_python_before = recursive_hash(random.getstate())
    rng_numpy_before = recursive_hash(np.random.get_state())

    block3_trace, block3_handles = trace_block3(model)
    logits_grads = {}
    logits_hook_handles = []

    def capture_logits_grad(index):
        def hook(grad):
            logits_grads[str(index)] = tensor_hash(grad)
            return None
        return hook

    # Nettoyage des hooks même si le forward ou le backward échoue.
    try:
        preds = model(images_device)

        for index, prediction in enumerate(preds):
            if prediction.requires_grad:
                logits_hook_handles.append(
                    prediction.register_hook(
                        capture_logits_grad(index)
                    )
                )

        logits_hash = recursive_hash(preds)
        state_after_forward = recursive_hash(model.state_dict())

        label_list = []
        for elt in heads_list:
            if elt["name"] == "gender":
                label_list.append(gender)
            elif elt["name"] == "age":
                label_list.append(age)

        cost = loss.compute(preds, label_list, epoch=0)
        cost.backward()

    finally:
        for handle in logits_hook_handles:
            handle.remove()
        for handle in block3_handles:
            handle.remove()

    # Gradients individuels avant la mise à jour.
    parameter_grads = {
        name: tensor_hash(p.grad) if p.grad is not None else None
        for name, p in model.named_parameters()
    }

    grads_hash = recursive_hash({
        name: p.grad
        for name, p in model.named_parameters()
        if p.grad is not None
    })

    grad_norm_squared = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm_squared += float(
                p.grad.detach().pow(2).sum().item()
            )

    optimizer.step()

    return {
        "step": step_idx,
        "seed": seed,
        "device": str(device),
        "strict_deterministic": strict_deterministic,
        "disable_dataparallel": disable_dataparallel,
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "deterministic_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "loss": float(cost.detach().cpu().item()),
        "loss_exact_hash": tensor_hash(cost),
        "ohem_kept_samples": getattr(loss, "last_kept_samples", "Non trouvé"),
        "before_model_hash": before_hash,
        "after_model_hash": model_hash(model),
        "state_before": state_before,
        "state_after_forward": state_after_forward,
        "state_after": recursive_hash(model.state_dict()),
        "optimizer_before": optimizer_before,
        "optimizer_after": recursive_hash(optimizer.state_dict()),
        "rng_cpu_before": rng_cpu_before,
        "rng_cuda_before": rng_cuda_before,
        "rng_python_before": rng_python_before,
        "rng_numpy_before": rng_numpy_before,
        "input_image_hash": tensor_hash(images),
        "gender_hash": tensor_hash(gender),
        "age_hash": tensor_hash(age),
        "logits_hash": logits_hash,
        "logits_grads": logits_grads,
        "parameter_grads": parameter_grads,
        "grads_hash": grads_hash,
        "grad_norm": float(grad_norm_squared ** 0.5),
        "block3_trace": block3_trace,
    }

def step_run(
    config_path, out_dir,
    strict_deterministic=False, disable_dataparallel=False,
):
    out_path = Path(out_dir)
    capture_dir = out_path / "bn_cases"
    capture_dir.mkdir(parents=True, exist_ok=True)

    context = prepare_training(
        config_path, strict_deterministic, disable_dataparallel
    )
    seed, device, loader, model, heads, loss, optimizer = context

    trace_block3.current_step = 0
    trace_block3.capture_dir = capture_dir

    try:
        payload = instrumented_step(
            next(iter(loader)), 0, seed, device,
            model, heads, loss, optimizer,
            strict_deterministic, disable_dataparallel,
        )
    finally:
        for attribute in ("current_step", "capture_dir"):
            if hasattr(trace_block3, attribute):
                delattr(trace_block3, attribute)

    dump_json(out_path / "one_step.json", payload)
    print(f"[one_step] écrit : {out_path / 'one_step.json'}", flush=True)

def scan_steps_run(
    config_path, out_dir, max_steps=100,
    strict_deterministic=False, disable_dataparallel=False,
):
    if max_steps < 1:
        raise ValueError("--max-steps doit être supérieur à zéro")

    out_path = Path(out_dir)
    capture_dir = out_path / "bn_cases"

    if out_path.exists() and any(out_path.glob("step_*.json")):
        raise FileExistsError(
            f"{out_dir} contient déjà un scan. "
            "Utilise un nouveau dossier."
        )

    if capture_dir.exists() and any(capture_dir.iterdir()):
        raise FileExistsError(
            f"{capture_dir} contient déjà des captures. "
            "Utilise un nouveau dossier."
        )

    out_path.mkdir(parents=True, exist_ok=True)
    capture_dir.mkdir(parents=True, exist_ok=True)

    context = prepare_training(
        config_path, strict_deterministic, disable_dataparallel
    )
    seed, device, loader, model, heads, loss, optimizer = context

    trace_block3.capture_dir = capture_dir
    completed = 0

    try:
        for step_idx, batch in enumerate(loader):
            if step_idx >= max_steps:
                break

            trace_block3.current_step = step_idx

            payload = instrumented_step(
                batch, step_idx, seed, device,
                model, heads, loss, optimizer,
                strict_deterministic, disable_dataparallel,
            )

            capture_path = capture_dir / f"bn_step_{step_idx}.pt"
            if not capture_path.is_file():
                raise RuntimeError(
                    f"Capture BatchNorm incomplète au step {step_idx}."
                )

            json_path = out_path / f"step_{step_idx}.json"
            dump_json(json_path, payload)
            completed += 1

            print(
                f"[scan] step={step_idx} "
                f"JSON={json_path} capture={capture_path}",
                flush=True,
            )

    finally:
        for attribute in ("current_step", "capture_dir"):
            if hasattr(trace_block3, attribute):
                delattr(trace_block3, attribute)

    print(
        f"[scan] terminé : {completed} steps capturés dans {out_dir}",
        flush=True,
    )

def show_difference(a, b):
    differences = []
    missing = object()

    def walk(left, right, path=""):
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                child_path = f"{path}.{key}" if path else str(key)
                walk(
                    left.get(key, missing),
                    right.get(key, missing),
                    child_path,
                )
        elif left != right:
            differences.append((path, left, right))

    walk(a, b)

    def display(value):
        if value is missing:
            return "<absent>"
        return json.dumps(value, ensure_ascii=False)

    print(f"{len(differences)} champs différents")

    # Affiche la BatchNorm ciblée en priorité.
    differences.sort(
        key=lambda item: (
            not item[0].startswith("block3_trace"),
            item[0],
        )
    )

    for path, left, right in differences[:20]:
        print(f"{path}:")
        print(f"  A = {display(left)}")
        print(f"  B = {display(right)}")

    if len(differences) > 20:
        print(
            f"... {len(differences) - 20} autres différences "
            "conservées dans les JSON."
        )

def compare_probes(run_a, run_b):
    a = read_json(Path(run_a) / "probe.json")
    b = read_json(Path(run_b) / "probe.json")
    print("\n=== COMPARAISON INITIALISATION / BATCHES ===")
    if a != b:
        show_difference(a, b)
    else:
        print("Initialisation et batches inspectés identiques.")


def compare_one_step(run_a, run_b):
    a = read_json(Path(run_a) / "one_step.json")
    b = read_json(Path(run_b) / "one_step.json")
    print("\n=== COMPARAISON D'UNE MISE À JOUR ===")
    if a != b:
        show_difference(a, b)
    else:
        print("Tous les champs inspectés sont identiques.")


def compare_scan_steps(run_a, run_b):
    def files_by_step(folder):
        return {
            int(path.stem.split("_")[-1]): path
            for path in Path(folder).glob("step_*.json")
        }

    a_files = files_by_step(run_a)
    b_files = files_by_step(run_b)
    print("\n=== COMPARAISON DU SCAN ===")

    if not a_files or not b_files:
        raise FileNotFoundError("Un des dossiers ne contient aucun step_*.json")

    for step in sorted(set(a_files) | set(b_files)):
        if step not in a_files or step not in b_files:
            print(f"Step {step} absent dans un des deux runs.")
            return

        a = read_json(a_files[step])
        b = read_json(b_files[step])
        if a != b:
            print(f"PREMIÈRE DIFFÉRENCE AU STEP {step}")
            show_difference(a, b)
            return

    print(f"Aucune différence dans les {len(a_files)} steps comparés.")


def run_in_subprocess(
    config_path, out_dir, mode, max_steps,
    strict_deterministic, disable_dataparallel,
):
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(TRAINING_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    )
    _, seed = load_config(config_path)
    env["PYTHONHASHSEED"] = str(seed)

    args = [
        sys.executable,
        str(Path(__file__).resolve()),
        f"--{mode}",
        "--config", config_path,
        "--out", out_dir,
        "--max-steps", str(max_steps),
    ]
    if strict_deterministic:
        args.append("--strict-deterministic")
    if disable_dataparallel:
        args.append("--disable-dataparallel")

    subprocess.run(args, env=env, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--out", default="/tmp/seed_debug")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--probe-only", action="store_true")
    modes.add_argument("--one-step", action="store_true")
    modes.add_argument("--scan-steps", action="store_true")
    modes.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"))
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--strict-deterministic", action="store_true")
    parser.add_argument("--disable-dataparallel", action="store_true")
    args = parser.parse_args()

    if args.compare:
        compare_scan_steps(*args.compare)
        return

    if not args.config:
        parser.error("--config est requis pour lancer un run")

    if args.probe_only:
        probe_run(
            args.config, args.out,
            args.strict_deterministic, args.disable_dataparallel,
        )
        return
    if args.one_step:
        step_run(
            args.config, args.out,
            args.strict_deterministic, args.disable_dataparallel,
        )
        return
    if args.scan_steps:
        scan_steps_run(
            args.config, args.out, args.max_steps,
            args.strict_deterministic, args.disable_dataparallel,
        )
        return

    run_a = Path(args.out) / "runA"
    run_b = Path(args.out) / "runB"
    # Ne supprime pas silencieusement des résultats existants.
    if run_a.exists() or run_b.exists():
        parser.error("runA ou runB existe déjà : choisis un nouveau --out")

    for mode, compare in (
        ("probe-only", compare_probes),
        ("one-step", compare_one_step),
        ("scan-steps", compare_scan_steps),
    ):
        for folder in (run_a, run_b):
            run_in_subprocess(
                args.config, str(folder), mode, args.max_steps,
                args.strict_deterministic, args.disable_dataparallel,
            )
        compare(str(run_a), str(run_b))


if __name__ == "__main__":
    main()