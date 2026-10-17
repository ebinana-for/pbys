# Prerequisites
1. Nvidia Containet Toolkit.
Training requires Nvidia GPU; [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) has to be installed on the machine.
2. Git-lfs. The model and some data files are stored using git-lfs. It should be instaled on the machine, and file update has to be done: `git lfs update`.

# Environment
## Setup in docker
We suggest running the code in the docker container; below we describe environment setup for docker

Build the docker image.
```
cd docker
bash docker/build_docker.sh
```
Launch the container. You might want to change some parameters in the `run_docker.sh` script:
 - ports if the default 1240-1242 ports are already occupied on the machine
 - change `DATADIR` to point to the root folder with the data and models. It will be mounted to the container. (see Sec. [Mounting the data](#Mounting-the-data) for more details.)
 - mound additional folders to the container: `-v <dir_to_mount>:<mountpoint_in_container> \`
```
bash docker/run_docker.sh
bash docker/connect_to_docker.sh
```

#### Mounting the data
Environmental variable `DATADIR` in the `run_rocker.sh` file points to the default location of the dataset. This folder will be mounted to the `/data/` folder in the container, and will correspond to all model and script configuration files of the repositories.

For example, to reuse the datasets which were delivered for the project, one needs to copy them to the machine to the `<data_root>/` and specify `DATADIR=<data_root>` in the `run_docker.sh` file.

Once inside the container, you can check that the data is properly mounted by running the `ls /data/` command:

```
root@8e462e792169:/workspace# ls /data/
private_datasets  public_datasets  trained_models evals
```

#### Running docker as user

> Running docker container as a root may be dangerous and leads to permission problems. It is possible to run all the scripts as a non-root user.

`bash docker/user_connect_to_docker.sh`
`HOME=/workspace`

> The second line changes your `$HOME` directory, it is necessary to avoid permission problems with pip, cache folders, etc. This allows installing pip packages as user without switching to root. But this also may lead to the old, incompatible and corrupt packages to skeak into the new clean container. If this happens, remove the old python package repository:
`rm -r /workspace/.local/lib/python3.XX/`


## Alternative: Setup in virtual environment
It is possible to run the code without docker, but you will have to make sure that nvidia CUDA Toolkit is properly installed. Virtual environment is strongly recommended to avoid  package conflicts.

```
python -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -e .
```

# Utilisation
Please refer to [Utilisation documentation](./README_utilisation.md) for the detailed explanation of how to use the repository.
