#!/bin/bash
# 1. Mise à jour et installation des paquets système
sudo apt-get update
sudo apt-get install -y cmake doxygen pre-commit libboost-all-dev graphviz libopencv-dev zip libprotobuf-dev protobuf-compiler libprotoc-dev swig curl vim git wget gdb nano pkg-config libgtk-3-dev libyaml-cpp-dev git-lfs

# 2. Installation de la dépendance Python
pip install identify==2.6.0
