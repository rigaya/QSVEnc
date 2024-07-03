#!/bin/sh

export VPL_INSTALL_DIR=`pwd`/buildVPL
if [ ! -e _build ]; then
    mkdir _build
fi
if [ ! -e ${VPL_INSTALL_DIR} ]; then
    mkdir $VPL_INSTALL_DIR
fi
cd _build
cmake \
    -DCMAKE_INSTALL_PREFIX=${VPL_INSTALL_DIR} \
    -DBUILD_DEV=OFF \
    -DBUILD_TOOLS=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_PYTHON_BINDING=OFF \
    -DCMAKE_C_FLAGS_RELEASE="$@" \
    -DCMAKE_CXX_FLAGS_RELEASE="$@" \
    ../libvpl

cmake --build . --config Release
cmake --build . --config Release --target install
