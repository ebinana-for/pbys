"""Convert PyTorch posture classification model to ONNX format."""
import torch
import argparse
from src.utils.config import get_config
from src.posture.train_posture import load_model


def make_parser():
    parser = argparse.ArgumentParser("ONNX convert parser")
    parser.add_argument("--config_file", required=True, help="Path to input config")
    parser.add_argument("--model_path", required=True, help="Path to input model")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch_size")
    parser.add_argument(
        "--dynamic", action="store_true", required=False, help="Dynamic model flag"
    )
    parser.add_argument(
        "--fuse", action="store_true", required=False, help="fuse the wo heads"
    )
    return parser


def convertONNX(config, model_path, batch_size, dynamic, fuse):
    """
    Convert the model to ONNX format.
    Args:
        config: Configuration object with model parameters.
        model_path: Path to the trained model file.
        batch_size: Batch size for the dummy input.
        dynamic: Flag to indicate if the model should be dynamic.
        fuse: not used in this context, but kept for compatibility.
    Returns:
        None
    """
    # Get config
    config = get_config()
    config.set_config(args.config_file)

    # Model configuration
    model = load_model(config)
    model.to(torch.device("cpu"))
    model.eval()

    checkpoint_data = torch.load(model_path, map_location=torch.device("cpu"))
    
    # Extraction des poids si la sauvegarde est un dictionnaire complet
    if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
        state_dict = checkpoint_data["model_state_dict"]
    else:
        state_dict = checkpoint_data
        
    # Nettoyage adaptatif : retire 'module.' uniquement s'il est présent
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
    model.load_state_dict(clean_state_dict, strict=True)
    
    dynamic_axes_dict = {}
    if dynamic:
        dynamic_axes_dict = {
            "images": {0: "bs"},
        }

    dummy_input = torch.rand(
        batch_size,
        config.model.input_channels,
        config.input.input_height,
        config.input.input_width,
    )

    if "tr" in config.model.name or "mlp" in config.model.name:  # 2-entries model
        dummy_input2 = torch.rand(batch_size, 1, 24)
        model(dummy_input, dummy_input2)
        onnx_input = (dummy_input, dummy_input2)
        if dynamic:
            dynamic_axes_dict["kpts"] = {0: "bs"}
        input_names = ["images", "kpts"]
    else:
        model(dummy_input)
        onnx_input = dummy_input
        input_names = ["images"]

    torch.onnx.export(
        model,
        onnx_input,
        model_path.replace(".pth", ".onnx"),
        export_params=True,
        input_names=input_names,
        output_names=["output"],
        opset_version=12,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes_dict,
    )
    print("Model has been converted to ONNX")


if __name__ == "__main__":
    # Get args
    args = make_parser().parse_args()

    # Get config
    config = get_config()
    config.set_config(args.config_file)

    print(
        f"Converting with batch_size {args.batch_size}, dynamic {args.dynamic}, \
        fuse {args.fuse}"
    )
    convertONNX(config, args.model_path, args.batch_size, args.dynamic, args.fuse)
