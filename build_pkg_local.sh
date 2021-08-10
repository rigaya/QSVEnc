#!/bin/sh

TARGET_EXE=qsvencc
TARGET_OS=$1
PKG_TYPE=$2
OUTPUT_DIR=`pwd`/../output
NPROC=$(grep 'processor' /proc/cpuinfo | wc -l)

mkdir ${OUTPUT_DIR}

rm -rf AviSynthPlus vapoursynth
git clone https://github.com/AviSynth/AviSynthPlus.git AviSynthPlus
git clone https://github.com/vapoursynth/vapoursynth.git vapoursynth

docker build -t build_${TARGET_EXE}_${TARGET_OS} -f docker/docker_${TARGET_OS} .

RUN_NAME=build_pkg_${TARGET_EXE}_${TARGET_OS}
docker run -dit --rm -v ${OUTPUT_DIR}:/output -u "$(id -u):$(id -g)" --name ${RUN_NAME} build_${TARGET_EXE}_${TARGET_OS}

docker exec ${RUN_NAME} ./configure --extra-cxxflags="-I./AviSynthPlus/avs_core/include -I./vapoursynth/include"
docker exec ${RUN_NAME} make -j${NPROC}
docker exec ${RUN_NAME} ./${TARGET_EXE} --version
docker exec ${RUN_NAME} ./check_options.py
docker exec ${RUN_NAME} ./build_${PKG_TYPE}.sh
docker exec ${RUN_NAME} sh -c "cp -v ./*.${PKG_TYPE} /output/"

rm -rf AviSynthPlus vapoursynth
