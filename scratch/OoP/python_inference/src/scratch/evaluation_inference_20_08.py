"""
Générateur de CSV d'inférence via l'évaluateur PyTorch - 20/08/2026
Utilise PostureDataset pour évaluer le dataset et exporte les résultats 
au format strict C++ avec les vérités terrains (GT) associées.
Supporte l'inférence via PyTorch (.pt) et ONNX (.onnx).
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../training')))

from tqdm import tqdm
import torch
import pickle
import numpy as np
import pandas as pd
from fire import Fire
from torch.utils.data import DataLoader

from src.utils.config import get_config
from src.posture.train_posture import load_model
from src.posture.dataset.posture_dataset import PostureDataset
from src import PROJECT_DIR


def run(model_path, config, eval_side, output_dir, use_cpp_letterboxing):
    """
    Run the evaluation of the posture model and export results to CSV with Ground Truth.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Injection du mode dans l'environnement pour communiquer avec PostureDataset
    # (Méthode plus sûre que de modifier la config si celle-ci est verrouillée en lecture seule)
    os.environ["USE_CPP_LETTERBOXING"] = "1" if use_cpp_letterboxing else "0"
    mode_str = "RATIO C++ (Letterboxing)" if use_cpp_letterboxing else "DISTORSION (Étirement par défaut)"
    print(f"📐 Mode Redimensionnement (Dataset) : {mode_str}")

    # GPU settings
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if output_dir is None:
        root_dir = os.path.dirname(model_path)
        output_dir = os.path.join(root_dir, "csv_exports_with_gt")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    is_onnx = model_path.endswith(".onnx")
    
    # --- CHARGEMENT DU MODÈLE (.ONNX OU .PT) ---
    if is_onnx:
        import onnxruntime as ort
        print(f"⚙️ Chargement du modèle ONNX : {model_path}")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        model = ort.InferenceSession(model_path, providers=providers)
    else:
        print(f"⚙️ Chargement du modèle PyTorch : {model_path}")
        # On ajoute weights_only=False pour éviter l'erreur de sécurité de PyTorch 2.6
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            trained_state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            trained_state_dict = checkpoint["state_dict"]
        else:
            trained_state_dict = checkpoint
        
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            trained_state_dict, "module."
        )
        model = load_model(config)
        model.load_state_dict(trained_state_dict)
        model.to(device)
        model.eval()

    # Create dataloader
    eval_dataset = PostureDataset(config, task="TEST", side=eval_side)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    # Result
    sequence_df = eval_dataset.img_df.copy()
    sequence_df["seq_name"] = sequence_df["file"].apply(
        lambda x: os.path.basename(os.path.dirname(x))
    )
    num_positive_labels = len(config.input.positive_labels)
    
    print("🚀 Démarrage de l'inférence sur le dataset...")
    pred_dict, label_dict = infer(
        eval_loader,
        sequence_df,
        model,
        device,
        cache=False,
        num_positive_labels=num_positive_labels,
        is_onnx=is_onnx
    )

    class_names = ["correctly_seated"] + [
        eval_dataset.label_map[key] for key in range(num_positive_labels)
    ]
    
    # --- Remplir la colonne 0 (correctly_seated) ---
    for seq in pred_dict.keys():
        pos_preds = pred_dict[seq][:, 1:]
        valid_mask = ~np.isnan(pos_preds).any(axis=1)
        pred_dict[seq][valid_mask, 0] = 1.0 - np.clip(np.nanmax(pos_preds[valid_mask], axis=1), 0, 1)

    # =====================================================================
    # EXPORT DES PRÉDICTIONS + GT EN CSV (Format strict C++)
    # =====================================================================
    cpp_class_order = [
        "closetoairbag", "legsondashboard", "torsodissimetry",           
        "reachingback", "lateralbodycontact", "oneshoulderincontact",
        "pelvisrotation", "pelvisdrift"
    ]
    
    print(f"\n💾 Exportation des {len(pred_dict.keys())} séquences au format CSV (Pred + GT)...")
    
    for seq in pred_dict.keys():
        seq_preds = pred_dict[seq]
        seq_labels = label_dict[seq]
        df_export = pd.DataFrame()
        
        df_export["index"] = np.arange(seq_preds.shape[0])
        
        for ii, class_name_from_model in enumerate(class_names):
            clean_name = class_name_from_model.lower().replace("_", "").replace(" ", "")
            final_name = "closetoairbag" if clean_name == "tooclosetoairbag" else clean_name
                
            df_export[final_name] = seq_preds[:, ii]
            df_export[f"{final_name}_gt"] = seq_labels[:, ii]

        for cpp_col in cpp_class_order:
            if cpp_col not in df_export.columns:
                df_export[cpp_col] = -1.0
                df_export[f"{cpp_col}_gt"] = -1
                
        final_columns = ["index"]
        for cpp_col in cpp_class_order:
            final_columns.append(cpp_col)
            final_columns.append(f"{cpp_col}_gt")
            
        df_export = df_export[final_columns]
        
        csv_out_path = os.path.join(output_dir, f"{seq}_eval_with_gt.csv")
        df_export.to_csv(csv_out_path, index=False)
        print(f"  -> Sauvegardé : {csv_out_path}")
        
    print("\n✅ Inférence et export terminés avec succès !")


def infer(eval_loader, sequence_df, model, device, cache, num_positive_labels, is_onnx):
    """
    Run inference on the evaluation dataset and compute predictions and labels.
    """
    compute = True
    if cache:
        cache_dir = PROJECT_DIR / "cache"
        if not os.path.isdir(str(cache_dir)):
            os.makedirs(str(cache_dir))
        cache_file = cache_dir / "posture_eval_tmp.pkl"
        if os.path.isfile(str(cache_file)):
            with open(str(cache_file), "rb") as handle:
                data = pickle.load(handle)
                pred_dict, label_dict = data
                compute = False

    if compute:
        sequence_list = np.unique(sequence_df["seq_name"].values).tolist()
        sequence_len = [
            sequence_df[sequence_df["seq_name"] == name]["frame_index"].max() + 1
            for name in sequence_list
        ]
        out_dim = num_positive_labels + 1
        pred_dict = {
            key: np.empty((nrows, out_dim), dtype=np.float64) * np.nan
            for key, nrows in zip(sequence_list, sequence_len)
        }
        label_dict = {
            key: np.zeros((nrows, out_dim), dtype=np.int64)
            for key, nrows in zip(sequence_list, sequence_len)
        }
        dataloader_pbar = tqdm(
            eval_loader, total=len(eval_loader), colour="green", leave=None
        )
        with torch.no_grad():
            cnt = 0
            for (
                image_tensors,
                posture_labels,
                posture_kpts,
                seq_name,
                frame_idx,
            ) in dataloader_pbar:
                assert seq_name[0] in pred_dict, "Sequence {} not is result_dict".format(seq_name[0])
                labels = np.squeeze(posture_labels.numpy())
                
                # --- INFÉRENCE HYBRIDE (ONNX ou PyTorch) ---
                if is_onnx:
                    input_names = [inp.name for inp in model.get_inputs()]
                    ort_inputs = {input_names[0]: image_tensors.cpu().numpy().astype(np.float32)}
                    
                    if len(input_names) > 1:
                        ort_inputs[input_names[1]] = posture_kpts.cpu().numpy().astype(np.float32)
                        
                    ort_outs = model.run(None, ort_inputs)
                    pred_raw = np.squeeze(ort_outs[0])
                    pos_probs = 1.0 / (1.0 + np.exp(-pred_raw))
                    
                    if pos_probs.ndim == 0:
                        pos_probs = np.array([pos_probs])
                    pred_dict[seq_name[0]][frame_idx.item(), 1:] = pos_probs
                else:
                    if model.n_inputs == 2:
                        pred = model(image_tensors.to(device), posture_kpts.to(device))
                    else:
                        pred = model(image_tensors.to(device))
                    
                    pos_probs = np.squeeze(torch.sigmoid(pred).cpu().numpy())
                    if pos_probs.ndim == 0:
                        pos_probs = np.array([pos_probs])
                    pred_dict[seq_name[0]][frame_idx.item(), 1:] = pos_probs
                
                label_dict[seq_name[0]][frame_idx.item(), :] = labels
                cnt += 1

        if cache:
            data = [pred_dict, label_dict]
            with open(str(cache_file), "wb") as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return pred_dict, label_dict


def main(model_path, output_dir=None, config_path=None, eval_side=None, use_cpp_letterboxing=False):
    """
    Évalue un modèle de posture sur le dataset de test et exporte les résultats avec vérités terrains.

    Args:
        model_path (str): Chemin vers le modèle (.onnx ou .pt).
        output_dir (str, optional): Dossier d'export pour les CSV générés.
        config_path (str, optional): Chemin vers le fichier config.yaml.
        eval_side (str, optional): Filtre anatomique ('driver', 'passenger' ou None pour les deux).
        use_cpp_letterboxing (bool): Modifie la géométrie du tenseur. Si True, applique le respect du 
                                     ratio C++ (bandes noires). Si False (par défaut), applique 
                                     l'étirement/distorsion standard de l'entraînement.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    if config_path is None:
        inference_folder = os.path.dirname(model_path)
        config_path = os.path.join(inference_folder, "../config.yaml")
    assert os.path.isfile(config_path), f"Config file not found: {config_path}"

    config = get_config()
    config.set_config(config_path)

    possible_eval_sides = ["driver", "passenger", None]
    if eval_side not in possible_eval_sides:
        raise ValueError(
            f"incorrect eval_side: {eval_side}. Must be one of: {possible_eval_sides}"
        )
    run(model_path, config, eval_side, output_dir, use_cpp_letterboxing)


if __name__ == "__main__":
    Fire(main)