import os
import sys
import argparse
import glob
import random
import shutil

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)

sys.path.append(parent)
from common_utils import *
from model_configs import *


parser = argparse.ArgumentParser()
parser.add_argument(
    "model_name",
    type=str,
    help="Which model to compile: age_gender, positions, positions_data_aug, positions_without_keypts, or positions_golden_reference",
    choices=[
        "age_gender",
        "positions",
        "positions_data_aug",
        "positions_without_keypts",
        "positions_golden_reference",
    ],
)
parser.add_argument(
    "--no-offload", dest="offload", action="store_false", help="do not offload to TIDL"
)
parser.add_argument(
    "--num_bits",
    dest="num_bits",
    type=int,
    default=8,
    choices=[8, 16, 32],
    help="number of bits used for quantization (use 32 for float-mode TIDL subgraphs)",
)
parser.add_argument(
    "--num_subgraphs",
    dest="num_subgraphs_max",
    default=16,
    type=int,
    help="maximum number of TIDL subgraphs for offload (actual number of subgraphs may be less that this)",
)
parser.add_argument(
    "--pc-inference",
    dest="device",
    action="store_false",
    help="compile for inference on PC",
)
parser.add_argument(
    "--num_iters",
    dest="calib_iters",
    default=10,
    type=int,
    help="number of calibration iterations",
)
parser.add_argument(
    "--calib_set_path",
    dest="calib_set_path",
    type=str,
    default=None,
    help="Path to the calibration set. If not provided, a default path will be used.",
)
parser.add_argument(
    "-z", "--run_model_zoo", action="store_true", help="Run model zoo models"
)
args = parser.parse_args()

# model_id = "cl-dlr-onnx_mobilenetv2"
# if args.run_model_zoo:
#     model_id = "cl-3090_tvmdlr_imagenet1k_torchvision_mobilenet_v2_tv_onnx"
# download_model(models_configs, model_id)

# model specifics
model_id = f"model_{args.model_name}"
model_path = models_configs[model_id]["session"]["model_path"]
model_path_opt = "./"+model_id+"_opt.onnx"
model_input_name = "imagesNet_IN"
model_input_height = models_configs[model_id]["extra_info"]["input_size"]
model_input_shape = (1, 1, model_input_height, 192)
model_input_dtype = "float32"
model_layout = "NCHW"
model_output_directory = artifacts_folder + model_id

# TIDL compiler specifics
# We are compiling the model for J7 device using
# a compiler distributed with SDK 7.0
DEVICE = os.environ["SOC"]
SDK_VERSION = (10, 0)

# convert the model to relay IR format
import onnx
from tvm import relay

print(model_path)
from onnx_model_opt import (
    tidlOnnxModelOptimize as onnxOpt,
)

onnxOpt(
    model_path,
    model_path_opt,
    scaleList=models_configs[model_id]["session"]["input_scale"],
    meanList=models_configs[model_id]["session"]["input_mean"],
)

onnx_model = onnx.load(model_path_opt)
mod, params = relay.frontend.from_onnx(
    onnx_model, shape={model_input_name: model_input_shape}
)

if args.device:
    build_target = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu"
    cross_cc_args = {
        "cc": os.path.join(
            os.environ["ARM64_GCC_PATH"], "bin", "aarch64-none-linux-gnu-gcc"
        )
    }
    model_output_directory = model_output_directory + "_device"
else:
    build_target = "llvm"
    cross_cc_args = {}
model_output_directory = model_output_directory + "/artifacts"


def get_default_calibration_set_path(model_id):
    if model_id == "model_age_gender":
        folder = "../data/images/agegender-calibration"
    elif model_id in ("model_positions", "model_positions_data_aug"):
        folder = "../data/images/position_calibration_twoclass"
    elif model_id == "model_positions_without_keypts":
        folder = "../data/images/position_without_keypts_calibration"
    elif model_id == "model_positions_golden_reference":
        folder = "../data/images/position_without_keypts_calibration"
    else:
        print("Unknown model_id for default calibration set path")
        sys.exit(1)
    print(f"No calibration set given. Using default path. {folder}")
    return folder


# image preprocessing for calibration
def preprocess_for_onnx_mobilenetv2(image_path):
    import cv2
    import numpy as np

    # read the image using openCV
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # convert to RGB
    img = img[:, :, ::-1]
    img = np.expand_dims(img[:, :, 0], axis=2)

    img = cv2.resize(img, (192, model_input_height), interpolation=cv2.INTER_CUBIC)

    img = np.expand_dims(img, axis=2)

    img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

    config = models_configs[model_id]

    gen_param_yaml(model_output_directory, config, model_input_height, 192)
    return img


# create the directory if not present
# clear the directory
os.makedirs(model_output_directory, exist_ok=True)


for root, dirs, files in os.walk(model_output_directory, topdown=False):
    [os.remove(os.path.join(root, f)) for f in files]
    [os.rmdir(os.path.join(root, d)) for d in dirs]

if args.offload:
    print("offload")
    from tvm.relay.backend.contrib import tidl

    assert args.num_bits in [8, 16, 32]
    assert args.num_subgraphs_max <= 16

    # Use advanced calibration for 8-bit quantization
    # Use simple calibration for 16-bit quantization and float-mode
    advanced_options = {
        8: {
            "calibration_iterations": args.calib_iters,
            "calibration_frames": 100,
            # below options are set to default values, include here for reference
            "quantization_scale_type": 4,
            "high_resolution_optimization": 0,
            "pre_batchnorm_fold": 1,
            # below options are only overwritable at accuracy level 9, otherwise ignored
            "activation_clipping": 1,
            "weight_clipping": 1,
            "bias_calibration": 1,
            "channel_wise_quantization": 0,
        },
        16: {
            "calibration_iterations": 1,
        },
        32: {
            "calibration_iterations": 1,
        },
    }
    folder = args.calib_set_path if args.calib_set_path else get_default_calibration_set_path(model_id)
    calib_files = glob.glob(f"{folder}/*.png")
    if len(calib_files) == 0:
        print("No calibration files found. Aborting")
        sys.exit(1)
    else:
        print(f"Found {len(calib_files)} calibration files in {folder}")
    random.shuffle(calib_files)
    calib_images = list(map(preprocess_for_onnx_mobilenetv2, calib_files))
    calib_input_list = [{model_input_name: img} for img in calib_images]
    print("device", DEVICE)
    # Create the TIDL compiler with appropriate parameters
    compiler = tidl.TIDLCompiler(
        DEVICE,
        SDK_VERSION,
        tidl_tools_path=os.environ["TIDL_TOOLS_PATH"],
        artifacts_folder=model_output_directory,
        tensor_bits=args.num_bits,
        debug_level=0,
        max_num_subgraphs=args.num_subgraphs_max,
        c7x_codegen=0,
        accuracy_level=(1 if args.num_bits == 8 else 0),
        advanced_options=advanced_options[args.num_bits],
    )

    # partition the graph into TIDL operations and TVM operations
    mod, status = compiler.enable(mod, params, calib_input_list)

    # build the relay module into deployables
    with tidl.build_config(tidl_compiler=compiler):
        graph, lib, params = relay.build_module.build(
            mod, target=build_target, params=params
        )

    # remove nodes / params not needed for inference
    tidl.remove_tidl_params(params)
else:
    import tvm

    # build the relay module into deployables
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = relay.build(mod, target=build_target, params=params)

# save the deployables
path_lib = os.path.join(model_output_directory, "deploy_lib.so")
path_graph = os.path.join(model_output_directory, "deploy_graph.json")
path_params = os.path.join(model_output_directory, "deploy_params.params")

lib.export_library(path_lib, **cross_cc_args)
with open(path_graph, "w") as fo:
    fo.write(graph)
with open(path_params, "wb") as fo:
    fo.write(relay.save_param_dict(params))

directory_path = os.path.join(model_output_directory, "tempDir")
print("Removing directory:", directory_path)
if os.path.isdir(directory_path):
    shutil.rmtree(directory_path)