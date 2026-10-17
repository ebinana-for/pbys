### Probayes-Forvia, july 2025. 
# Input data content description 

All the input data used for this project is supplied on an external medium, in the ```forvia``` directory.

The content is as follows.

## private_datasets

This directory contains the data used to train the position classification model:
- raw records
- annotations
- videos in mp4 format
- body keypoints calculated using a third-party model
- extracted images, divided into 2 datasets, train and test

It also contains a database of faces extracted from the recordings, in the form of NIR images. This database is used to train the age/gender model, in addition to the public datasets.

## public_datasets

This directory contains 3 public datasets of faces, used for training the age/gender model:
- imdb-clean
- Lagenda
- UTK-Face

It is important to use these delivered datasets (do not re-download them from the internet) as they have been checked and cleaned.

## trained_models

This directory contains the following models in ONNX format:

- age_gender: age and gender estimation model

- position/no_keypoints/july_2025: position classification model trained on the train base and evaluated on the test base. Body keypoints are not plotted on the input images.

- position/no_keypoints/july_2025_all: position classification model trained on the train+test basis. Model not evaluated, but theoretically more robust because trained on more data. Body keypoints are not plotted on the input images.

- position/with_keypoints/july_2025: position classification model trained on the train base and evaluated on the test base. Body keypoints are plotted on the input images.

- position/with_keypoints/july_2025_all: position classification model trained on the train+test basis. Model not evaluated, but theoretically more robust because trained on more data. Body keypoints are plotted on the input images.

## evals
Empty directory, to be used for storing model evaluation results.


# Data extraction

The private_datasets and public_datasets directories contain tar archives. These must be extracted before they can be used. Each archive must have the same name after extraction, and be located in the same place in the tree.

To extract an archive, run the following commands:

```
cd forvia/private_datasets
tar -xf recordings_forvia_23_05.tar
tar -xf recordings_forvia_26_06.tar
tar -xf recordings_forvia_28_05.tar
tar -xf recordings_forvia_30_01.tar
tar -xf recordings_forvia_31_01.tar
tar -xf recordings_pbys_02_04.tar
cd forvia/public_datasets
tar -xf imdb-clean.tar
tar -xf Lagenda.tar
tar -xf UTK-Face.tar
```

# Data usage
When mounting docker volumes (see forvia-oms/README.md), the contents of this directory must be visible in the data volume, i.e:

```
root@8e462e792169:/workspace# ls /data/
private_datasets  public_datasets  trained_models evals

```

