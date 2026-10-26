# Forvia-OMS-Library

## PC x86 Installation (Docker)
### Prerequisites:
The installation was tested on a computer running Ubuntu 22.04 with a monitor display.
PC x86 installation relies on docker. A valid docker install is required.

To build and install the docker, run:

```
cd docker
bash build_docker.sh
bash run_docker.sh
bash connect_to_docker.sh
```


## PC x86 Installation (Manual)
```
source install_deps.sh
source install_dlti_host_x86_target_am62a.sh
```

## Preparation of models

It is safer to regenerate the models, since compiled models on a PC may not work on another PC

### Using already compiled models :
```
tar -xzf model-artifacts.tar.gz
```

### Generating models from onnx:
```
cd python_scripts
python3 onnx_compilation.py <model_type> --pc-inference
```
Where <model_type> is one of the following: age_gender, positions, positions_without_keypts
Remove the `--pc-inference` option to generate models for the AM62A target.

The generated artifacts wil be in the model-artifacts folder.

Note: to change the path of the onnx files, open model_configs.py and change the model_path field for the model. Path to the calibration dataset and number of iterations can be changed via the CLI arguments of the script.

## Software compilation

To compile the shared libraries:
```
mkdir build
cd build
cmake ..
make -j
```
If you want to build the doxygen API documentation, replace ``cmake ..`` by ``cmake .. -DBUILD_DOC=ON``.

### Tests
To run the unit tests, run:
```
cd build\
ctest
```


### Documentation

If you added the argument to the cmake command line, the doxygen API documentation is available in the html_doc folder.


## Samples

### Age & gender
This sample estimate the age and gender of a person and display the results on the image

samples_age_gender_with_csv_and_video  [videofile] [csvfile]

```
cd bin
./Release/samples_age_gender_with_csv_and_video ../data/videos/thibault_vincent_28_M_178_68.mp4 ../data/videos/thibault_vincent_28_M_178_68.csv
```
### Positions

This sample classify the positions of a person and display the results on the image

samples_positions_with_csv_and_video [videofile] [csvfile]

```
cd bin
./Release/samples_positions_with_csv_and_video ../data/videos/thibault_vincent_28_M_178_68.mp4 ../data/videos/thibault_vincent_28_M_178_68.csv
```

### Age & gender evaluation
This sample estimate the age and gender of a person for all couples of video/csv of a directory and write the results to a csv file

samples_age_gender_evaluation [dir]

```
cd bin
./Release/samples_age_gender_evaluation ../data/videos/
```
### Positions evaluation

This sample classify the positions of a person from a video and write the results to a csv file


samples_positions_evaluation [dir]

```
cd bin
./Release/samples_positions_evaluation ../data/videos/
```

## AM62A Installation

Copy the git repository to the device.
Copy the artifacts generated for the device to the model-artifacts folder or use the precompiled models:

```
tar -xzf model-artifacts.tar.gz
```

### Compilation
```
export SOC=am62a
```

To compile the shared libraries and samples:
```
mkdir build
cd build
cmake ..
make -j
```


## Samples

### Age & gender evaluation
This sample estimates the age and gender of a person from a video and write the results to the console

samples_age_gender_with_csv_and_video_device [videofile] [csvfile]

```
cd bin
./Release/samples_age_gender_with_csv_and_video_device ../data/videos/thibault_vincent_28_M_178_68.mp4 ../data/videos/thibault_vincent_28_M_178_68.csv
```
### Positions evaluation

This sample classifies the positions of a person from a video and write the results to the console

```
cd bin
./Release/samples_positions_with_csv_and_video_device ../data/videos/thibault_vincent_28_M_178_68.mp4 ../data/videos/thibault_vincent_28_M_178_68.csv
```

