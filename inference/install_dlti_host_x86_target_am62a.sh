
tar -xzvf model-artifacts.tar.gz
mkdir external
pushd external
git clone https://github.com/TexasInstruments/edgeai-tidl-tools.git
pushd edgeai-tidl-tools
git checkout 10_00_08_00
# Supported SOC name strings am62, am62a, am68a, am68pa, am69a, am67a
export SOC=am62a
export TIDL_TOOLS_TYPE=CPU
source ./setup.sh
export TIDL_TOOLS_PATH=$(pwd)/tidl_tools
export LD_LIBRARY_PATH=$TIDL_TOOLS_PATH:$TIDL_TOOLS_PATH/osrt_deps/opencv_4.2.0_x86_u22/opencv/
export ARM64_GCC_PATH=$(pwd)/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/bin/
popd
popd