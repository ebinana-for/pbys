# Forvia-OMS Usage guide

# Install

Please refer to [Input data description](./README_input_data_description.md) to extract and prepare input datasets and pre-trained models.

Then, refer to [Installation documentation](./README.md) for the detailed explanation of how to use setup the environment for the library.

# Training

## Age/Gender prediction model

### Datasets

The available datasets are:

- UTK-Face (public dataset)
- IMDB (public dataset)
- Lagenda (public dataset)
- Forvia-NIR (non-public dataset): set of faces extracted from Forvia video records.

Each dataset if composed of a train/test split, with associated images and label files in csv format. FOr the public datasets, the label files have been filtered and checked for discarding erroneous labels. Thus the following files should be used:

- For UTK-Face: utk_face_train_sanity_check1.csv / utk_face_test_sanity_check1.csv
- For IMDB: imdb_train_new_sanity_check1.csv / imdb_test_new_sanity_check1.csv
- For Lagenda: lagenda_annotation_train_sanity_check1.csv / lagenda_annotation_test_sanity_check1.csv
- For Forvia-NIR: forvia_face_train.csv / forvia_face_test.csv

### Training configuration

The parameters (for datasets, model and training) are gathered in the file ```config/train_config_age_gender.yaml```.
You should change the following entries:

- Paths to the datasets: images and labels (`/data/...`). They should be updated according to their location on your server.
- The path where model will be saved (train_dir).
- The batch size according the available GPU memory.
- Aditionally, the loss function and other model hyperparameters.

To launch the training procedure, do:

```
cd scripts
bash train_age_gender.sh -d 0 -c ../config/train_config_age_gender.yaml
```

Option `-d 0` lets you control GPU id, on which the model will be trained. By default, a subfolder with th current date will be created in the specified output directory. Pytorch models will be saved at each epoch, as well as a a dump of the configuration file (`<train_dir>/config.yaml`) and a tensorboard file. You can vizualize the latterwith the following command:

```
tensorboard --logdir=<path_to_the_saved_models> --port=1240
```

Note: ports 1240 to 1242 are binded by default in ```docker/run_docker.sh```

## Posture prediction model

### Datasets

The available datasets are:
- recordings_forvia_30_01
- recordings_forvia_31_01
- recordings_forvia_23_05
- recordings_forvia_28_05
- recordings_pbys_02_04
These datasets are non-public. They consist in video records made by Forvia and Probayes. Frames have been annotated with postures and body keypoints, then they have been extracted and put into a train and test sets.

Each dataset folder is expected to be organized as follows:
```
recordings_forvia_23_05
├── train/
|   ├── ...
└── test/
    ├── images/
    ├── keypoints/
    ├── labels/
        ├── andreas_23_cid
        │   ├── default.txt
        │   └── synsets.txt
        ├── benjamin_31_M_ohc
        │   ├── default.txt
        │   └── synsets.txt
        ├── benjamin_40_M_ohc
            ├── default.txt
            └── synsets.txt
...
```
`./labels/` - is a required anotations folder, and subfolders have the same names as the .csv files from the C++ inference code.


### Training configuration

The parameters (for datasets, model and training) are gathered in the file config/```train_config_posture_multi_class.yaml```.
You should change:

- Paths to the datasets: images and labels (`/data/...`). They should be updated according to their location on your server.
- The path where model will be saved (`train_dir`).
- The batch size according the available GPU memory.
- Aditionally, the number of classes and associated posture labels, and other model hyperparameters. For example, if you want to remove one class, move the corresponding label name from positive_labels to negative_labels section. Be carefull to not change label names as they must match those in the annotation files.

To run the train, do:
```
cd scripts
bash train_posture.sh -d 0 -c ../config/train_config_posture_multi_class.yaml
```

Option `-d 0` lets you control GPU id, on which the model will be trained. By default, a subfolder with th current date will be created in the specified output directory. Pytorch models will be saved at each epoch, as well as a a dump of the configuration file (`<train_dir>/config.yaml`) and a tensorboard file. You can vizualize the latterwith the following command:

```
tensorboard --logdir=<path_to_the_saved_models> --port=1240
```

> **_NOTE:_**  Ports 1240 to 1242 are binded by default in ```docker/run_docker.sh```


# ONNX model conversion

Trained models are saved in pytorch (.pth) format.
Once you have chosen the best performing model among the trained ones, you need to convert it before using it on the target board.
They need first to be converted to onnx format.

> **_NOTE:_**  a nvidia GPU is required for this conversion.

## Age/Gender

Run:

```
cd scripts
bash convert_age_gender.sh -d 0 -b 1 -c ../config/train_config_age_gender.yaml -m <path_to_pth_model> -f
```

The .onnx model will be saved in the same folder as the .pth model.
To convert the onnx to the final format for the board, please refer to the forvia-oms-libray documentation. Note: model config file `config.yaml` file is saved by the train script next to the checkpoint.


## Posture

Run:
```
bash convert_posture.sh -d 0 -b 1 -c ../config/train_config_posture_multi_class.yaml -m <path_to_pth_model>
```

The .onnx model will be saved in the same folder as the .pth model. To convert the onnx to the final format for the board, please refer to the forvia-oms-libray documentation. Note: model config file `config.yaml` file is saved by the train script next to the checkpoint.

# Inference on PC and python-evaluation

This section describes how to run a trained model (pth or onnx) and computes metrics.
> **_NOTE 1:_**   this is an evaluation on PC only.

> **_NOTE 2:_**   a Nvidia GPU is required.

## Age/Gender

### Public test datasets

The onnx model can be run on the public test datasets prepared during the project.
The output of this evaluation are a series of histograms and other plots for each testing dataset.

1. Open ```config/test_config_age_gender.yaml``` and update the following fields:

    - test_dir: where to save evaluation results.
    - checkpoint: the path to the onnx model want you to evaluate.
    - img_path, annot_path: the paths toward dataset images and labels.

2. Run:

```
python src/age_gender/age_gender_eval.py --config_file config/test_config_age_gender.yaml
```

### Non-public test datasets

The following instructions evaluate models for age and gender predictions on non-public datasets, i.e. data acquired using Forvia's imaging system. The script requires videos (XX.mp4) and related tracking data (XX.csv) to be stored in an unique folder <folder_path>. The files must be named using the following structure: Surname_Age_Gender.mp4/csv, with Age an integer, Gender M for Male and F for Female.

1. Open ```config/test_config_age_gender.yaml``` and update the following fields:

    - test_dir: where to save evaluation results.
    - checkpoint: the path to the onnx model you want to evaluate.
    - non_public_video_path: the path toward the folder that contains videos and tracking files.

2. Run:

```
python src/age_gender/age_gender_eval_non_public.py -c config/test_config_age_gender.yaml
```


## Posture

Posture model evaluation is performed on non-public datasets only (records made by Forvia and Probayes), using a pytorch model (.pth)

```
python src/posture/posture_model_evaluation.py <path_to_pth_model> <output_dir> <path_to_yaml_config>

```
To evaluate only driver-side records, run:

```
python src/posture/posture_model_evaluation.py <path_to_pth_model> <output_dir> <path_to_yaml_config> driver
```

To evaluate only passenger-side records, run:

```
python src/posture/posture_model_evaluation.py <path_to_pth_model> <output_dir> <path_to_yaml_config> passenger
```

> **_NOTE:_** it is important to take the same config file as the one used for training (the one dumped into the training output directory). However, you may change the test dataset paths for evaluating different ones.


# Inference on target board and c++ evaluation


## Age/Gender

### Preparation of public dataset for running with C++ code

To evaluate the embedded model with the C++ code (and in particular, an int8 model) on public age/gender datasets, one needs to first package these datasets into a proper format:

 - images are assembled into the video
 - fake face keypoints are calculated so that the bbox of the face is reused by the c++ inference code "as if" they were detected keypoints.

Run the following script to prepare the datasets:

```
python src/age_gender/build_video_for_age_eval_tidl.py <path_to_config.yaml> <output_dir_path> debug=True
```

Where config file is the evaluation configuration file, for example `config/config_video_age_evaluation_tidl.yaml`. (Make sure that the paths to the test datasets are correct on your machine / setup).

> Known issues (1):
Warning: if the keypoint -> bbox calculation logic in the C++ code changes, the modifications in this script are required.

> Known issues (2):
Due to rounding errors, a small fraction of images are discarded by the C++ code as the bounding box turns out to be out of bounds of the video frame.

> Known issues (3):
Due to some Dataloader error, UTKFace video creation fails at frame 55608. To avoid this, launch the script with the `debug=True` option, this would limit the number of treated frames to 55607, the script would pass and last ~5% of UTKFace images would be lost.

### Inference

C++ inference code takes .mp4 videos + .csv with keypoints (from smarteye) as input. For each input videos, it generates a csv prediction file. Please refer to the forvia-oms-library documentation for the prediction procdure.


## Posture

Posture model evaluation is performed on non-public datasets only (.mp4 records made by Forvia and Probayes, and associated smarteye output csv files).
These records must belong to the datasets used for model evaluation, i.e. they must be manually labelled beforehand (see the posture datasets section).

### Dataset preparation
C++ evaluation scripts take a folder with keypoints and video files as input. Python script `src/posture/gather_assets_cpp_evaluation.py` automates assembling of the test dataset, based on the train-model configuration file. It copies all information needed for evaluation: videos, labels and keypoints to one place to facilitate further evaluation.

```
python src/posture/gather_assets_cpp_evaluation.py <model_config.yaml> <output_directory>
```

> The script works under assumption that the dataset is organized similarly to the one delivered at the end of the project.

### Running the inference

C++ inference code takes .mp4 videos + .csv with keypoints (from smarteye) as input. For each input videos, it generates a csv prediction file. Please refer to the forvia-oms-library documentation for the prediction procdure.

### Metrics

#### Confusion matrix and TP-FP sequence evaluation barplots:

```
python src/posture/posture_model_evaluation_tidl.py <inference_results_path> --config_path <model_config_path> --eval_side <["driver", "passenger", None]>
```

where folder `<inference_results_path>` should contain output of the C++ inference script: `<video>_out_positions.csv`, one per video; <model_config_path> - path to `config.yaml` file with the description of the datasets. Make sure that the paths in the config file are accessible from your environment (docker, sshfs, etc.); `--eval_side` option allows filtering of the passenger / driver positions only.

*Best practice*: the evaluation folder should be as follow

    eval_folder
    |- andreas_23_cid_out_positions.csv      # C++ prediction output, where andreas_23_cid must be the same as the original video record filename
    |   .
    |   .
    |- benjamin_31_M_ohc_out_positions.csv      # C++ prediction output, where benjamin_31_M_ohc must be the same as the original video record filename
    |- model
        |- config.yaml # the test datasets defined in this config file must contain the records we want to evaluate. Otherwise, the records


A subfolder named metrics will be created in the <output_dir> folder. It will contain evaluation plots.

#### Sliding window sequence evaluation:

The following script performs sequence-wise metric calculation, and plots score evolution in Positive segments and FPR-Recall-threshold curves:

```
python src/posture/sequence_evaluation.py <inference_results_path>
```

where folder `<inference_results_path>` should contain output of the C++ inference script: `<video>_out_positions.csv`, one per video. Additional CLI paths are not required when the script is launched on the output of the `gather_assets_cpp_evaluation.py` script.

## Int-8 quantization (calibration) datasets

Thumbnails with input images for both models can be created with the script, which runs the DataLoader, prepares the input, then filters it to prepare a reasonable dataset distribution and saves the images:

```
python src/create_calibration_img.py <path_to_model_config.yaml> <output_dir_path> agegender
python src/create_calibration_img.py <path_to_model_config.yaml> <output_dir_path> posture
```
