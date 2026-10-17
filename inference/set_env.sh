# Supported SOC name strings am62, am62a, am68a, am68pa, am69a, am67a
export SOC=am62a
export TIDL_TOOLS_TYPE=CPU
cd ./external/edgeai-tidl-tools/
export TIDL_TOOLS_PATH=$(pwd)/tidl_tools
export LD_LIBRARY_PATH=$TIDL_TOOLS_PATH
export ARM64_GCC_PATH=$(pwd)/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu
#export ARM64_GCC_PATH=$(pwd)/ti-processor-sdk-linux-edgeai-am62a-evm-10_00_00_08/linux-devkit/sysroots
cd ../..