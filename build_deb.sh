#!/bin/sh

PACKAGE_NAME=qsvencc
PACKAGE_BIN=qsvencc
PACKAGE_MAINTAINER=rigaya
PACKAGE_DESCRIPTION=
PACKAGE_ROOT=.debpkg
PACKAGE_VERSION=`./scripts/get-version.sh`
PACKAGE_ARCH=`uname -m`
PACKAGE_ARCH=`echo ${PACKAGE_ARCH} | sed -e 's/x86_64/amd64/g'`

PACKAGE_DEPENDS="libc6(>=2.31), libva-drm2, libva-x11-2, intel-media-va-driver-non-free | intel-media-va-driver | i965-va-driver | va-driver"
PACKAGE_RECOMMENDS="intel-opencl-icd, libmfx1, libmfxgen1 | libmfx-gen1.2, libigfxcmrt7"

if [ -e /etc/lsb-release ]; then
    PACKAGE_OS_ID=`cat /etc/lsb-release | grep DISTRIB_ID | cut -f 2 --delim="="`
    PACKAGE_OS_VER=`cat /etc/lsb-release | grep DISTRIB_RELEASE | cut -f 2 --delim="="`
    PACKAGE_OS_CODENAME=`cat /etc/lsb-release | grep DISTRIB_CODENAME | cut -f 2 --delim="="`
    PACKAGE_OS="_${PACKAGE_OS_ID}${PACKAGE_OS_VER}"
    case "${PACKAGE_OS_CODENAME}" in
        focal|jammy|noble) ;;
        *)
            echo "${PACKAGE_OS_ID}${PACKAGE_OS_VER} ${PACKAGE_OS_CODENAME} not supported in this script!"
            exit 1
            ;;
    esac
fi

if [ ! -e ${PACKAGE_BIN} ]; then
    echo "${PACKAGE_BIN} does not exist!"
    exit 1
fi

mkdir -p ${PACKAGE_ROOT}/DEBIAN
build_pkg/replace.py \
    -i build_pkg/template/DEBIAN/control \
    -o ${PACKAGE_ROOT}/DEBIAN/control \
    --pkg-name ${PACKAGE_NAME} \
    --pkg-bin ${PACKAGE_BIN} \
    --pkg-version ${PACKAGE_VERSION} \
    --pkg-arch ${PACKAGE_ARCH} \
    --pkg-maintainer ${PACKAGE_MAINTAINER} \
    --pkg-depends "${PACKAGE_DEPENDS}" \
    --pkg-desc ${PACKAGE_DESCRIPTION}
# build_pkg は他のエンコーダと共有のサブモジュールなので、Recommends は生成後の control に追記する。
sed -i "/^Depends:/a Recommends: ${PACKAGE_RECOMMENDS}" ${PACKAGE_ROOT}/DEBIAN/control

mkdir -p ${PACKAGE_ROOT}/usr/bin
cp ${PACKAGE_BIN} ${PACKAGE_ROOT}/usr/bin
chmod +x ${PACKAGE_ROOT}/usr/bin/${PACKAGE_BIN}

DEB_FILE="${PACKAGE_NAME}_${PACKAGE_VERSION}_${PACKAGE_ARCH}.deb"
dpkg-deb -b "${PACKAGE_ROOT}" "${DEB_FILE}"
