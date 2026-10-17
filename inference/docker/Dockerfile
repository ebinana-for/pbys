FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt install -y doxygen zip swig curl vim git wget gdb nano pre-commit graphviz cmake protobuf-compiler
RUN apt install -y libboost-all-dev  libopencv-dev  libprotobuf-dev libprotoc-dev pkg-config libgtk-3-dev libyaml-cpp-dev

WORKDIR /external
# Supported SOC name strings am62, am62a, am68a, am68pa, am69a, am67a
ENV SOC=am62a
ENV TIDL_TOOLS_TYPE=CPU
RUN git clone https://github.com/TexasInstruments/edgeai-tidl-tools.git
RUN cd edgeai-tidl-tools && git checkout 10_00_08_00 && bash setup.sh

ENV TIDL_TOOLS_PATH=/external/edgeai-tidl-tools/tidl_tools
ENV LD_LIBRARY_PATH=$TIDL_TOOLS_PATH:$TIDL_TOOLS_PATH/osrt_deps/opencv_4.2.0_x86_u22/opencv/
ENV ARM64_GCC_PATH=/external/edgeai-tidl-tools/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/bin/
