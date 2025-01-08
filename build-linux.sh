#!/bin/bash
set -e

GCC_COMPILER=/home/infty/rk3588/rknn_prj/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu

#just show prj info
BUILD_PRJ_NAME="yolov8-c"
BUILD_PRJ_PATH="./yolo/yolo-c"

#cfg for prj

#DIR
BUILD_DIR="./build"
INSTALL_DIR="../install"

if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

if [[ -d "${INSTALL_DIR}" ]]; then
  rm -rf ${INSTALL_DIR}
fi

CMAKE_PATH="../yolo/yolo-c/cpp"
#SOC
TARGET_SOC="rk3588"
#Linux
TARGET_ARCH="aarch64"
# Debug / Release
BUILD_TYPE="Release"
# Build with Address Sanitizer for memory check, BUILD_TYPE need set to Debug
ENABLE_ASAN="OFF"
DISABLE_RGA="OFF"



if [[ -z ${GCC_COMPILER} ]];then
    if [[ ${TARGET_SOC} = "rv1106"  || ${TARGET_SOC} = "rv1103" ]];then
        echo "Please set GCC_COMPILER for $TARGET_SOC"
        echo "such as export GCC_COMPILER=~/opt/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf"
        exit
    elif [[ ${TARGET_SOC} = "rv1109" || ${TARGET_SOC} = "rv1126" ]];then
        GCC_COMPILER=arm-linux-gnueabihf
    else
        GCC_COMPILER=aarch64-linux-gnu
    fi
fi
echo "$GCC_COMPILER"
export CC=${GCC_COMPILER}-gcc
export CXX=${GCC_COMPILER}-g++

if command -v ${CC} >/dev/null 2>&1; then
    :
else
    echo "${CC} is not available"
    echo "Please set GCC_COMPILER for $TARGET_SOC"
    echo "such as export GCC_COMPILER=~/opt/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf"
    exit
fi


echo "==================================="
echo "BUILD_PRJ_NAME=${BUILD_PRJ_NAME}"
echo "BUILD_PRJ_PATH=${BUILD_PRJ_PATH}"
echo "TARGET_SOC=${TARGET_SOC}"
echo "TARGET_ARCH=${TARGET_ARCH}"
echo "BUILD_TYPE=${BUILD_TYPE}"
echo "ENABLE_ASAN=${ENABLE_ASAN}"
echo "DISABLE_RGA=${DISABLE_RGA}"
echo "INSTALL_DIR=${INSTALL_DIR}"
echo "BUILD_DIR=${BUILD_DIR}"
echo "CC=${CC}"
echo "CXX=${CXX}"
echo "==================================="


cd ${BUILD_DIR}
cmake ${CMAKE_PATH} \
    -DTARGET_SOC=${TARGET_SOC} \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=${TARGET_ARCH} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DENABLE_ASAN=${ENABLE_ASAN} \
    -DDISABLE_RGA=${DISABLE_RGA} \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}
make -j4
make install

# # Check if there is a rknn model in the install directory
# suffix=".rknn"
# shopt -s nullglob
# if [ -d "$INSTALL_DIR" ]; then
#     files=("$INSTALL_DIR/model/"/*"$suffix")
#     shopt -u nullglob

#     if [ ${#files[@]} -le 0 ]; then
#         echo -e "\e[91mThe RKNN model can not be found in \"$INSTALL_DIR/model\", please check!\e[0m"
#     fi
# else
#     echo -e "\e[91mInstall directory \"$INSTALL_DIR\" does not exist, please check!\e[0m"
# fi
