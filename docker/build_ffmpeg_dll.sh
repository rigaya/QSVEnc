#!/bin/bash
#MSYS2用ffmpeg dllビルドスクリプト
#Visual Studioへの環境変数を通して起動する
#pacman -S base-devel mingw-w64-i686-toolchain mingw-w64-x86_64-toolchain autotools autogen
#pacman -S p7zip git nasm yasm python unzip
# cmake関連
#pacman -S mingw32/mingw-w64-i686-cmake mingw64/mingw-w64-x86_64-cmake
# 通常の pacman -S cmakeで導入しないこと
#普通にpacman -S mesonとやるとうまくdav1dがビルドできないので注意
#pacman -S mingw32/mingw-w64-i686-meson mingw64/mingw-w64-x86_64-meson
# harfbuzzに必要
#pacman -S gtk-doc mingw64/mingw-w64-x86_64-ragel mingw32/mingw-w64-i686-ragel
#fontconfigに必要
#pacman -S gperf mingw32/mingw-w64-i686-python-lxml mingw64/mingw-w64-x86_64-python-lxml
#pacman -S mingw-w64-i686-python mingw-w64-i686-python-six
#pacman -S mingw-w64-x86_64-python mingw-w64-x86_64-python-six
#libdoviに必要
# curl -o rustup-init.exe -sSL https://win.rustup.rs/
# ./rustup-init.exe -y --default-host=x86_64-pc-windows-gnu
# rustup install stable --profile minimal
# rustup default stable
# rustup target add x86_64-pc-windows-gnu
# rustup target add x86_64-pc-windows-msvc
# rustup target add i686-pc-windows-gnu
# rustup target add i686-pc-windows-msvc
# デフォルトをgnuのほうにしておかないとlinkエラーが出る
# rustup default stable-x86_64-pc-windows-gnu
# cargo install cargo-c
# Vulkan
# pacman -S mingw-w64-i686-uasm mingw-w64-x86_64-uasm
NJOBS=$NUMBER_OF_PROCESSORS
PATCHES_DIR=`pwd`/patches
YUVFILE=`pwd`/test.yuv
YUVFILE_10=`pwd`/test_10.yuv
TEST_YUV_8_URL="https://github.com/rigaya/ffmpeg_dlls_for_hwenc/releases/download/20250825/test_8.7z"
TEST_YUV_10_URL="https://github.com/rigaya/ffmpeg_dlls_for_hwenc/releases/download/20250825/test_10.7z"

BUILD_ALL="FALSE"
SSE4_2="FALSE"
UPDATE_FFMPEG="FALSE"
ENABLE_SWSCALE="FALSE"
FOR_FFMPEG4="FALSE"
FOR_AUDENC="FALSE"
ADD_TLVMMT="FALSE"
BUILD_EXE="FALSE"
ENABLE_GPL="FALSE"
ENABLE_LTO="FALSE"

set -e

# [ bitrate, swscale, exe ]
TARGET_BUILD=""

while getopts ":-:at:ur" key; do
  if [[ $key == - ]]; then
    key=${OPTARG}
    keyarg=${!OPTIND}
  else
    keyarg=$OPTARG
  fi

  case $key in
    mmttlv )
      ADD_TLVMMT="TRUE"
      ;;
    enable-gpl )
      ENABLE_GPL="TRUE"
      ;;
    enable-swscale )
      ENABLE_SWSCALE="TRUE"
      ;;
    lto )
      ENABLE_LTO="TRUE"
      ;;
    a | all )
      BUILD_ALL="TRUE"
      ;;
    u | update-ffmpeg )
      UPDATE_FFMPEG="TRUE"
      ;;
    r )
      FOR_FFMPEG4="TRUE"
      ;;
    t | target )
      TARGET_BUILD=$keyarg
      ;;
    ? | *)
      echo "help message"
      exit 0
      ;;
  esac

  while [[ -n ${!OPTIND} && ${!OPTIND} != -* ]]; do
      args+=(${!OPTIND})
      OPTIND=$((OPTIND + 1))
  done
done

BUILD_DIR=`pwd`/build_ffmpeg
if [ "$FOR_FFMPEG4" = "TRUE" ]; then
    BUILD_DIR=${BUILD_DIR}4
fi

if [ $ENABLE_LTO = "TRUE" ]; then
    BUILD_DIR=${BUILD_DIR}_lto
fi


if [ "$TARGET_BUILD" = "audenc" ]; then
    FOR_AUDENC="TRUE"
    BUILD_EXE="TRUE"
elif [ "$TARGET_BUILD" = "exe" ]; then
    BUILD_EXE="TRUE"
    ENABLE_SWSCALE="TRUE"
fi

if [ $BUILD_EXE != "TRUE" ]; then
    BUILD_DIR=${BUILD_DIR}_dll
fi

echo TARGET_BUILD=$TARGET_BUILD
echo FOR_FFMPEG4=$FOR_FFMPEG4
echo BUILD_DIR=$BUILD_DIR

mkdir -p $BUILD_DIR
mkdir -p $BUILD_DIR/src
cd $BUILD_DIR/src

if [ "$ENABLE_GPL" != "FALSE" ]; then
  if [ "$BUILD_EXE" = "FALSE" ]; then
    echo "--enable-gpl can be only used when --target exe is set."
    exit 1
  fi
fi

# [ "x86", "x64" ]
if [ "${MSYSTEM:-}" = "MINGW32" ]; then
    TARGET_ARCH="x86"
    VC_ARCH="win32"
    FFMPEG_ARCH="i686"
    MINGWDIR="mingw32"
    CMAKE_GENERATOR="MSYS Makefiles"
    CARGOC_TARGET="i686-pc-windows-gnu"
elif [ "${MSYSTEM:-}" = "MINGW64" ]; then
    TARGET_ARCH="x64"
    VC_ARCH="x64"
    FFMPEG_ARCH="x86_64"
    MINGWDIR="mingw64"
    CMAKE_GENERATOR="MSYS Makefiles"
    CARGOC_TARGET="x86_64-pc-windows-gnu"
else
    if [ `arch` = "x86_64" ]; then
        TARGET_ARCH="x64"
    else
        TARGET_ARCH="x86"
    fi
    VC_ARCH=
    FFMPEG_ARCH=`arch`
    MINGWDIR=
    CMAKE_GENERATOR="Unix Makefiles"
    CARGOC_TARGET="x86_64-unknown-linux-gnu"
fi

if [ "$MINGWDIR" = "" ]; then
    FFMPEG_TARGET_OS="linux"
else
    FFMPEG_TARGET_OS="mingw32"
fi

PYTHON_BIN="python"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    PYTHON_BIN="python3"
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "python/python3 not found."
    exit 1
fi

INSTALL_DIR=$BUILD_DIR/$TARGET_ARCH/build
PKG_CONFIG_PATH_FFMPEG=${INSTALL_DIR}/lib/pkgconfig


LIBSTDCXX_A=
LIBSTDCXX_DIR=
if [ "$MINGWDIR" = "" ]; then
    LIBSTDCXX_A=`gcc -print-file-name=libstdc++.a`
    LIBSTDCXX_DIR=`dirname "$LIBSTDCXX_A"`
fi
LIBSTDCXX_STATIC_FLAGS="-Wl,-Bstatic -lstdc++ -Wl,-Bdynamic"
if [ "$MINGWDIR" = "" ] && [ -n "$LIBSTDCXX_A" ] && [ -f "$LIBSTDCXX_A" ]; then
    # ビルド中/ビルド後で同じ指定に統一するため、libstdc++.a を -l: 形式で指定する。
    LIBSTDCXX_STATIC_FLAGS="-L${LIBSTDCXX_DIR} -l:libstdc++.a"
fi

FFMPEG_DISABLE_ASM=""
#BUILD_CCFLAGS="-mtune=alderlake -msse2 -fexcess-precision=fast -mfpmath=sse -ffast-math -fomit-frame-pointer -ffunction-sections -fno-ident -D_FORTIFY_SOURCE=0 -I${INSTALL_DIR}/include"
BUILD_CCFLAGS="-mtune=alderlake -msse2 -mfpmath=sse -fomit-frame-pointer -fno-ident -D_FORTIFY_SOURCE=0 -I${INSTALL_DIR}/include"
BUILD_LDFLAGS="-Wl,--strip-all -L${INSTALL_DIR}/lib"
if [ $TARGET_ARCH = "x86" ]; then
    BUILD_CCFLAGS="${BUILD_CCFLAGS} -m32 -mstackrealign"
    #  libavcodec/h264_cabac.c: In function 'ff_h264_decode_mb_cabac': libavcodec/x86/cabac.h:192:5: error: 'asm' operand has impossible 対策
    FFMPEG_DISABLE_ASM="--disable-inline-asm"
fi
if [ "$MINGWDIR" != "" ]; then
    BUILD_LDFLAGS="${BUILD_LDFLAGS} -static -static-libgcc -static-libstdc++"
else
    BUILD_LDFLAGS="${BUILD_LDFLAGS} ${LIBSTDCXX_STATIC_FLAGS}"
fi

if [ $ENABLE_LTO = "TRUE" ]; then
    BUILD_CCFLAGS="-flto -ffat-lto-objects ${BUILD_CCFLAGS}"
    BUILD_LDFLAGS="-flto=auto ${BUILD_LDFLAGS}"
else
    BUILD_CCFLAGS="-ffunction-sections ${BUILD_CCFLAGS}"
    BUILD_LDFLAGS="-Wl,--gc-sections ${BUILD_LDFLAGS}"
fi

PROFILE_GEN_CC="-fprofile-generate -fprofile-partial-training"
PROFILE_GEN_LD="-fprofile-generate -fprofile-partial-training"
PROFILE_USE_CC="-fprofile-use"
PROFILE_USE_LD="-fprofile-use"
PROFILE_SVTAV1="-fprofile-correction"

if [ "$FOR_FFMPEG4" = "TRUE" ]; then
    FFMPEG_DIR_NAME="ffmpeg4_dll"
else
    FFMPEG_DIR_NAME="ffmpeg_dll"
fi
FFMPEG_SSE="-msse2"
if [ $SSE4_2 = "TRUE" ]; then
    FFMPEG_DIR_NAME="${FFMPEG_DIR_NAME}_sse42"
    FFMPEG_SSE="-msse4.2 -mpopcnt"
fi

# static link用のフラグ (これらがないとundefined referenceが出る)
BUILD_CCFLAGS="${BUILD_CCFLAGS} -DLIBXML_STATIC -DFRIBIDI_LIB_STATIC"

# lameのstaticビルドに必要
BUILD_CCFLAGS="${BUILD_CCFLAGS} -DNCURSES_STATIC"

# small build用のフラグと通常用のフラグ
BUILD_CCFLAGS_SMALL="-Os -fno-unroll-loops ${BUILD_CCFLAGS}"
BUILD_CCFLAGS="-O3 ${BUILD_CCFLAGS}"

if [ $ENABLE_SWSCALE = "TRUE" ]; then
    FFMPEG_DIR_NAME="${FFMPEG_DIR_NAME}_swscale"
fi
if [ $ADD_TLVMMT = "TRUE" ]; then
    FFMPEG_DIR_NAME="${FFMPEG_DIR_NAME}_tlvmmt"
fi
if [ $FOR_AUDENC = "TRUE" ]; then
    FFMPEG_DIR_NAME="${FFMPEG_DIR_NAME}_audenc"
fi
if [ $BUILD_EXE = "TRUE" ]; then
    FFMPEG_DIR_NAME="${FFMPEG_DIR_NAME}_exe"
fi
if [ $BUILD_ALL != "FALSE" ]; then
    UPDATE_FFMPEG="TRUE"
fi

echo TARGET_ARCH=$TARGET_ARCH
echo BUILD_ALL=$BUILD_ALL
echo SSE4_2=$SSE4_2
echo UPDATE_FFMPEG=$UPDATE_FFMPEG
echo FOR_AUDENC=$FOR_AUDENC
echo ENABLE_SWSCALE=$ENABLE_SWSCALE
echo FFMPEG_DIR_NAME=$FFMPEG_DIR_NAME
echo BUILD_EXE=$BUILD_EXE
echo ENABLE_LTO=$ENABLE_LTO

# ============================================================
# ライブラリごとのビルドフラグ設定
# ビルド設定に基づいて、必要なライブラリのみビルドする
# ============================================================

# ライブラリのビルドが必要か判定するヘルパー関数
should_build() {
    local flag_name="BUILD_LIB_$1"
    [ "${!flag_name}" = "TRUE" ]
}

# Normalize .pc files to force static libstdc++ linkage in a way that works
# with FFmpeg configure checks driven by `cc`.
normalize_static_libstdcxx_pc_dir() {
    local pc_dir="$1"
    [ -d "$pc_dir" ] || return 0
    local replacement="$LIBSTDCXX_STATIC_FLAGS"
    local replacement_escaped=
    replacement_escaped=$(printf '%s' "$replacement" | sed -e 's/[\/&]/\\&/g')
    shopt -s nullglob
    local pc=
    for pc in "$pc_dir"/*.pc; do
        sed -E -i \
            -e "s|-Wl,-Bstatic[[:space:]]+-lstdc\\+\\+[[:space:]]+-Wl,-Bdynamic|${replacement_escaped}|g" \
            -e "s|-L[^[:space:]]+[[:space:]]+-l:libstdc\\+\\+\\.a|${replacement_escaped}|g" \
            -e "s|/usr/lib/gcc/[^[:space:]]+/[0-9.]+/libstdc\\+\\+\\.a|${replacement_escaped}|g" \
            -e "s|-static-libstdc\\+\\+|${replacement_escaped}|g" \
            -e 's/-Wl,-Bstatic[[:space:]]+-Wl,-Bstatic/-Wl,-Bstatic/g' \
            -e 's/-Wl,-Bdynamic[[:space:]]+-Wl,-Bdynamic/-Wl,-Bdynamic/g' \
            "$pc"
    done
    shopt -u nullglob
}

start_build() {
    echo "=== Building $1 ======================================="
}

ensure_test_yuv_files() {
    local work_dir
    work_dir="$(dirname "${YUVFILE}")"

    if ! command -v 7z >/dev/null 2>&1; then
        echo "7z is required to extract test_8.7z/test_10.7z."
        exit 1
    fi

    if [ ! -f "${YUVFILE}" ]; then
        wget -O "${work_dir}/test_8.7z" "${TEST_YUV_8_URL}"
        7z x -y "${work_dir}/test_8.7z" -o"${work_dir}"
    fi
    if [ ! -f "${YUVFILE}" ]; then
        echo "test.yuv not found after extracting test_8.7z."
        exit 1
    fi

    if [ ! -f "${YUVFILE_10}" ]; then
        wget -O "${work_dir}/test_10.7z" "${TEST_YUV_10_URL}"
        7z x -y "${work_dir}/test_10.7z" -o"${work_dir}"
    fi
    if [ ! -f "${YUVFILE_10}" ]; then
        echo "test_10.yuv not found after extracting test_10.7z."
        exit 1
    fi
}

# --- 全ライブラリのビルドフラグを初期化 (FALSE = ビルドしない) ---
BUILD_LIB_ZLIB="FALSE"
BUILD_LIB_BZIP2="FALSE"
BUILD_LIB_LIBPNG="FALSE"
BUILD_LIB_EXPAT="FALSE"
BUILD_LIB_FREETYPE="FALSE"
BUILD_LIB_LIBICONV="FALSE"
BUILD_LIB_FONTCONFIG="FALSE"
BUILD_LIB_FRIBIDI="FALSE"
BUILD_LIB_HARFBUZZ="FALSE"
BUILD_LIB_LIBUNIBREAK="FALSE"
BUILD_LIB_LIBASS="FALSE"
BUILD_LIB_LIBASS_DLL="FALSE"
BUILD_LIB_OPUS="FALSE"
BUILD_LIB_LIBOGG="FALSE"
BUILD_LIB_LIBVORBIS="FALSE"
BUILD_LIB_SPEEX="FALSE"
BUILD_LIB_LAME="FALSE"
BUILD_LIB_LIBSNDFILE="FALSE"
BUILD_LIB_TWOLAME="FALSE"
BUILD_LIB_SOXR="FALSE"
BUILD_LIB_LIBXML2="FALSE"
BUILD_LIB_LIBBLURAY="FALSE"
BUILD_LIB_ARIBB24="FALSE"
BUILD_LIB_LIBARIBCAPTION="FALSE"
BUILD_LIB_DAV1D="FALSE"
BUILD_LIB_LIBVPL="FALSE"
BUILD_LIB_LIBVPX="FALSE"
BUILD_LIB_NV_CODEC_HEADERS="FALSE"
BUILD_LIB_LIBXXHASH="FALSE"
BUILD_LIB_DOVI_TOOL="FALSE"
BUILD_LIB_GLSLANG="FALSE"
BUILD_LIB_LIBJPEG_TURBO="FALSE"
BUILD_LIB_LCMS2="FALSE"
BUILD_LIB_SHADERC="FALSE"
BUILD_LIB_SPIRV_CROSS="FALSE"
BUILD_LIB_VULKAN_LOADER="FALSE"
BUILD_LIB_LIBPLACEBO="FALSE"
BUILD_LIB_LIBPLACEBO_DLL="FALSE"
BUILD_LIB_ZIMG="FALSE"
BUILD_LIB_VVENC="FALSE"
BUILD_LIB_SVT_AV1="FALSE"
BUILD_LIB_XVIDCORE="FALSE"
BUILD_LIB_X264="FALSE"
BUILD_LIB_X265="FALSE"

# --- 音声系ライブラリ (全モードで必要) ---
BUILD_LIB_OPUS="TRUE"
BUILD_LIB_LIBOGG="TRUE"
BUILD_LIB_LIBVORBIS="TRUE"
BUILD_LIB_SPEEX="TRUE"
BUILD_LIB_LAME="TRUE"
BUILD_LIB_LIBSNDFILE="TRUE"
BUILD_LIB_TWOLAME="TRUE"
BUILD_LIB_SOXR="TRUE"

# --- 映像系ライブラリ (audenc以外で必要) ---
if [ "$FOR_AUDENC" != "TRUE" ]; then
    # 基本ライブラリ (字幕・フォント描画の依存チェーン)
    # freetype <- zlib, bzip2, libpng
    # fontconfig <- freetype, libiconv, expat, libpng
    BUILD_LIB_ZLIB="TRUE"
    BUILD_LIB_BZIP2="TRUE"
    BUILD_LIB_LIBPNG="TRUE"
    BUILD_LIB_EXPAT="TRUE"
    BUILD_LIB_FREETYPE="TRUE"
    BUILD_LIB_LIBICONV="TRUE"
    BUILD_LIB_FONTCONFIG="TRUE"

    # 字幕関連
    # libass <- freetype, fribidi, fontconfig, (harfbuzz, libunibreak: x64のみ)
    BUILD_LIB_FRIBIDI="TRUE"
    BUILD_LIB_LIBASS="TRUE"

    # デコーダー/デマクサー関連
    BUILD_LIB_LIBXML2="TRUE"
    BUILD_LIB_ARIBB24="TRUE"
    BUILD_LIB_LIBARIBCAPTION="TRUE"
    BUILD_LIB_DAV1D="TRUE"

    # HWアクセラレーション関連
    BUILD_LIB_LIBVPL="TRUE"
    BUILD_LIB_NV_CODEC_HEADERS="TRUE"

    # 映像コーデック
    BUILD_LIB_LIBVPX="TRUE"

    # GPU/Vulkan関連 (libplacebo依存チェーン)
    # libplacebo <- libjpeg, lcms2, shaderc, SPIRV-Cross, dovi_tool, libxxhash, Vulkan-Loader
    BUILD_LIB_LIBXXHASH="TRUE"
    BUILD_LIB_DOVI_TOOL="TRUE"
    BUILD_LIB_LIBJPEG_TURBO="TRUE"
    BUILD_LIB_LCMS2="TRUE"
    BUILD_LIB_SHADERC="TRUE"
    BUILD_LIB_SPIRV_CROSS="TRUE"
    BUILD_LIB_VULKAN_LOADER="TRUE"
    BUILD_LIB_LIBPLACEBO="TRUE"

    # 画像処理
    BUILD_LIB_ZIMG="TRUE"

    #bluray
    BUILD_LIB_LIBBLURAY="TRUE"

    # x64専用ライブラリ
    if [ "$TARGET_ARCH" = "x64" ]; then
        BUILD_LIB_HARFBUZZ="TRUE"
        BUILD_LIB_LIBUNIBREAK="TRUE"
        # エンコーダー (x64のみ)
        BUILD_LIB_VVENC="TRUE"
        BUILD_LIB_SVT_AV1="TRUE"
    fi

    # exe/dll固有のライブラリ
    if [ "$BUILD_EXE" = "TRUE" ]; then
        # exe: glslangが必要, DLL版ライブラリは不要
        BUILD_LIB_GLSLANG="TRUE"
    else
        # dll: DLL版ライブラリが必要, libblurayも使用
        if [ "$MINGWDIR" != "" ]; then
            BUILD_LIB_LIBASS_DLL="TRUE"
            BUILD_LIB_LIBPLACEBO_DLL="TRUE"
        fi
    fi

    # GPLライブラリ (--enable-gpl指定時のみ)
    if [ "$ENABLE_GPL" = "TRUE" ]; then
        BUILD_LIB_XVIDCORE="TRUE"
        BUILD_LIB_X264="TRUE"
        BUILD_LIB_X265="TRUE"
    fi
fi

# Linux静的リンク用途では、DLL専用ターゲットのみ無効化する
if [ "$MINGWDIR" = "" ]; then
    # static libplacebo連鎖は有効のまま維持する
    BUILD_LIB_LIBPLACEBO_DLL="FALSE"
    #BUILD_LIB_LIBARIBCAPTION="FALSE"
    #BUILD_LIB_SOXR="FALSE"
    #BUILD_LIB_TWOLAME="FALSE"
    #BUILD_LIB_VVENC="FALSE"
    #BUILD_LIB_SVT_AV1="FALSE"
fi

# --- ビルドフラグの表示 ---
echo "--- Library build flags (TRUE only) ---"
for flag in BUILD_LIB_ZLIB BUILD_LIB_BZIP2 BUILD_LIB_LIBPNG BUILD_LIB_EXPAT BUILD_LIB_FREETYPE BUILD_LIB_LIBICONV BUILD_LIB_FONTCONFIG BUILD_LIB_FRIBIDI BUILD_LIB_HARFBUZZ BUILD_LIB_LIBUNIBREAK BUILD_LIB_LIBASS BUILD_LIB_LIBASS_DLL BUILD_LIB_OPUS BUILD_LIB_LIBOGG BUILD_LIB_LIBVORBIS BUILD_LIB_SPEEX BUILD_LIB_LAME BUILD_LIB_LIBSNDFILE BUILD_LIB_TWOLAME BUILD_LIB_SOXR BUILD_LIB_LIBXML2 BUILD_LIB_LIBBLURAY BUILD_LIB_ARIBB24 BUILD_LIB_LIBARIBCAPTION BUILD_LIB_DAV1D BUILD_LIB_LIBVPL BUILD_LIB_LIBVPX BUILD_LIB_NV_CODEC_HEADERS BUILD_LIB_LIBXXHASH BUILD_LIB_DOVI_TOOL BUILD_LIB_GLSLANG BUILD_LIB_LIBJPEG_TURBO BUILD_LIB_LCMS2 BUILD_LIB_SHADERC BUILD_LIB_SPIRV_CROSS BUILD_LIB_VULKAN_LOADER BUILD_LIB_LIBPLACEBO BUILD_LIB_LIBPLACEBO_DLL BUILD_LIB_ZIMG BUILD_LIB_VVENC BUILD_LIB_SVT_AV1 BUILD_LIB_XVIDCORE BUILD_LIB_X264 BUILD_LIB_X265; do
    if [ "${!flag}" = "TRUE" ]; then
        echo "  $flag"
    fi
done

#--- ソースのダウンロード ---------------------------------------
if [ "$FOR_FFMPEG4" = "TRUE" ]; then
    if [ ! -d "ffmpeg" ]; then
        wget https://ffmpeg.org/releases/ffmpeg-4.4.3.tar.xz
        tar xf ffmpeg-4.4.3.tar.xz
        mv ffmpeg-4.4.3 ffmpeg
    fi
else
    if [ ! -d "ffmpeg" ]; then
        UPDATE_FFMPEG="TRUE"
    fi
    if [ $UPDATE_FFMPEG != "FALSE" ]; then
        #if [ ! -d "ffmpeg" ] || [ ! -d "ffmpeg/.git" ]; then
        #    if [ -d "ffmpeg" ]; then
        #        rm -rf ffmpeg
        #    fi
        #    git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg
        #else
        #    cd ffmpeg
        #    make uninstall && make distclean &> /dev/null
        #    cd ..
        #fi
        #cd ffmpeg
        #git fetch
        #git reset --hard
        #git checkout -b build 9d15fe77e33b757c75a4186fa049857462737713
        #cd ..
        wget https://ffmpeg.org/releases/ffmpeg-8.0.tar.xz
        tar xf ffmpeg-8.0.tar.xz
        mv ffmpeg-8.0 ffmpeg
        #wget https://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2
        #tar xf ffmpeg-snapshot.tar.bz2
    fi
fi

if should_build ZLIB && [ ! -d "zlib-1.3.2" ]; then
    wget https://zlib.net/zlib-1.3.2.tar.xz
    tar xf zlib-1.3.2.tar.xz
fi

if should_build LIBPNG && [ ! -d "libpng-1.6.50" ]; then
    wget https://download.sourceforge.net/libpng/libpng-1.6.50.tar.xz
    tar xf libpng-1.6.50.tar.xz
fi

if should_build BZIP2 && [ ! -d "bzip2-1.0.8" ]; then
    wget https://sourceware.org/pub/bzip2/bzip2-1.0.8.tar.gz
    tar xf bzip2-1.0.8.tar.gz
fi

if should_build EXPAT && [ ! -d "expat-2.7.1" ]; then
    wget https://github.com/libexpat/libexpat/releases/download/R_2_7_1/expat-2.7.1.tar.xz
    tar xf expat-2.7.1.tar.xz
fi

# freetype-2.12.1はダメ
if should_build FREETYPE && [ ! -d "freetype-2.11.0" ]; then
    wget https://download.savannah.gnu.org/releases/freetype/freetype-2.11.0.tar.xz
    tar xf freetype-2.11.0.tar.xz
fi

if should_build LIBICONV && [ ! -d "libiconv-1.16" ]; then
    wget https://ftp.gnu.org/pub/gnu/libiconv/libiconv-1.16.tar.gz
    tar xf libiconv-1.16.tar.gz
fi

#2.12.6でないといろいろ面倒 -> 2.12.1もだめ, 2.13.0もだめ
if should_build FONTCONFIG && [ ! -d "fontconfig-2.12.6" ]; then
    wget https://www.freedesktop.org/software/fontconfig/release/fontconfig-2.12.6.tar.gz
    tar xf fontconfig-2.12.6.tar.gz
fi

if should_build FRIBIDI && [ ! -d "fribidi-1.0.16" ]; then
    wget https://github.com/fribidi/fribidi/releases/download/v1.0.16/fribidi-1.0.16.tar.xz
    tar xf fribidi-1.0.16.tar.xz
fi

#if [ ! -d "graphite2-1.3.14" ]; then
#    wget https://github.com/silnrsi/graphite/releases/download/1.3.14/graphite2-1.3.14.tgz
#    tar xf graphite2-1.3.14.tgz
#fi

if should_build HARFBUZZ && [ ! -d "harfbuzz-11.4.4" ]; then
    wget https://github.com/harfbuzz/harfbuzz/releases/download/11.4.4/harfbuzz-11.4.4.tar.xz
    tar xf harfbuzz-11.4.4.tar.xz
fi

if should_build LIBUNIBREAK && [ ! -d "libunibreak-6.1" ]; then
    wget https://github.com/adah1972/libunibreak/releases/download/libunibreak_6_1/libunibreak-6.1.tar.gz
    tar xf libunibreak-6.1.tar.gz
fi

LIBASS_VERSION="0.17.4"
if [ $TARGET_ARCH = "x86" ]; then
    LIBASS_VERSION="0.14.0"
fi  
if should_build LIBASS && [ ! -d "libass-${LIBASS_VERSION}" ]; then
    wget https://github.com/libass/libass/releases/download/${LIBASS_VERSION}/libass-${LIBASS_VERSION}.tar.xz
    tar xf libass-${LIBASS_VERSION}.tar.xz
fi

if should_build LIBOGG && [ ! -d "libogg-1.3.6" ]; then
    wget https://downloads.xiph.org/releases/ogg/libogg-1.3.6.tar.xz
    tar xf libogg-1.3.6.tar.xz
fi

if should_build LIBVORBIS && [ ! -d "libvorbis-1.3.7" ]; then
    wget https://downloads.xiph.org/releases/vorbis/libvorbis-1.3.7.tar.xz
    tar xf libvorbis-1.3.7.tar.xz
fi

if should_build OPUS && [ ! -d "opus-1.6.1" ]; then
    wget https://downloads.xiph.org/releases/opus/opus-1.6.1.tar.gz
    tar xf opus-1.6.1.tar.gz
fi

if should_build SPEEX && [ ! -d "speex-1.2.1" ]; then
    wget http://downloads.xiph.org/releases/speex/speex-1.2.1.tar.gz
    tar xf speex-1.2.1.tar.gz
fi

if should_build LAME && [ ! -d "lame-3.100" ]; then
    wget https://sf-west-interserver-2.dl.sourceforge.net/project/lame/lame/3.100/lame-3.100.tar.gz
    tar xf lame-3.100.tar.gz
fi

if should_build TWOLAME && [ ! -d "twolame-0.4.0" ]; then
    wget https://sf-west-interserver-2.dl.sourceforge.net/project/twolame/twolame/0.4.0/twolame-0.4.0.tar.gz
    tar xf twolame-0.4.0.tar.gz
fi

if should_build LIBSNDFILE && [ ! -d "libsndfile-1.2.2" ]; then
    wget https://github.com/libsndfile/libsndfile/releases/download/1.2.2/libsndfile-1.2.2.tar.xz
    tar xf libsndfile-1.2.2.tar.xz
fi

if should_build SOXR && [ ! -d "soxr-0.1.3-Source" ]; then
    wget https://sf-west-interserver-2.dl.sourceforge.net/project/soxr/soxr-0.1.3-Source.tar.xz
    tar xf soxr-0.1.3-Source.tar.xz
fi

if should_build LIBXML2 && [ ! -d "libxml2-2.14.5" ]; then
    wget -O libxml2-2.14.5.tar.gz https://github.com/GNOME/libxml2/archive/refs/tags/v2.14.5.tar.gz
    tar xf libxml2-2.14.5.tar.gz
fi

#if [ ! -d "apache-ant-1.10.6-src.tar.xz" ]; then
#    wget https://archive.apache.org/dist/ant/source/apache-ant-1.10.6-src.tar.xz
#    tar xf apache-ant-1.10.6-src.tar.xz
#fi

if should_build LIBBLURAY && [ ! -d "libbluray-1.3.4" ]; then
    wget https://download.videolan.org/pub/videolan/libbluray/1.3.4/libbluray-1.3.4.tar.bz2
    tar xf libbluray-1.3.4.tar.bz2
fi

if should_build ARIBB24 && [ ! -d "aribb24-master" ]; then
    wget https://github.com/nkoriyama/aribb24/archive/master.zip
    mv master.zip aribb24-master.zip
    unzip aribb24-master.zip
fi

if should_build LIBARIBCAPTION && [ ! -d "libaribcaption-1.1.1" ]; then
    wget -O libaribcaption-1.1.1.tar.gz https://github.com/xqq/libaribcaption/archive/refs/tags/v1.1.1.tar.gz
    tar xf libaribcaption-1.1.1.tar.gz
fi

if should_build LIBVPL && [ ! -d "libvpl-2.16.0" ]; then
    wget -O libvpl-2.16.0.tar.gz https://github.com/intel/libvpl/archive/refs/tags/v2.16.0.tar.gz
    tar xf libvpl-2.16.0.tar.gz
fi

if should_build NV_CODEC_HEADERS && [ ! -d "nv-codec-headers-12.2.72.0" ]; then
    wget https://github.com/FFmpeg/nv-codec-headers/releases/download/n12.2.72.0/nv-codec-headers-12.2.72.0.tar.gz
    tar xf nv-codec-headers-12.2.72.0.tar.gz
fi

if should_build LIBVPX && [ ! -d "libvpx-1.16.0" ]; then
    wget -O libvpx-1.16.0.tar.gz https://github.com/webmproject/libvpx/archive/refs/tags/v1.16.0.tar.gz
    tar xf libvpx-1.16.0.tar.gz
fi

# if [ ! -d "gperf-3.0.4" ]; then
    # wget http://ftp.gnu.org/gnu/gperf/gperf-3.0.4.tar.gz
    # tar xf gperf-3.0.4.tar.gz
# fi

# if [ ! -d "gmp-6.1.0" ]; then
    # wget https://gmplib.org/download/gmp/gmp-6.1.0.tar.xz --no-check-certificate
    # tar xf gmp-6.1.0.tar.xz
# fi

# if [ ! -d "nettle-2.7.1" ]; then
    # wget ftp://ftp.gnu.org/gnu/nettle/nettle-2.7.1.tar.gz
    # tar xf nettle-2.7.1.tar.gz
# fi

# if [ ! -d "gnutls-3.3.19" ]; then
    # wget ftp://ftp.gnutls.org/gcrypt/gnutls/v3.4/gnutls-3.3.19.tar.xz
    # tar xf gnutls-3.3.19.tar.xz
# fi

if should_build DAV1D && [ ! -d "dav1d-1.5.3" ]; then
    wget https://code.videolan.org/videolan/dav1d/-/archive/1.5.3/dav1d-1.5.3.tar.gz
    tar xf dav1d-1.5.3.tar.gz
fi

if should_build LIBXXHASH && [ ! -d "libxxhash-0.8.3" ]; then
    wget -O libxxhash-0.8.3.tar.gz https://github.com/Cyan4973/xxHash/archive/refs/tags/v0.8.3.tar.gz
    tar xf libxxhash-0.8.3.tar.gz
    mv xxHash-0.8.3 libxxhash-0.8.3
fi

if should_build GLSLANG && [ ! -d "glslang-15.4.0" ]; then
    wget -O glslang-15.4.0.tar.gz https://github.com/KhronosGroup/glslang/archive/refs/tags/15.4.0.tar.gz
    tar xf glslang-15.4.0.tar.gz
fi

if should_build SHADERC; then
    if [ ! -d "shaderc" ]; then
        git clone https://github.com/google/shaderc shaderc
        cd shaderc && git checkout tags/v2024.1 && "${PYTHON_BIN}" ./utils/git-sync-deps && cd ..
    elif [ ! -d "shaderc/third_party/spirv-tools" ] || [ ! -d "shaderc/third_party/spirv-headers" ]; then
        cd shaderc && "${PYTHON_BIN}" ./utils/git-sync-deps && cd ..
    fi
fi

if should_build SPIRV_CROSS && [ ! -d "SPIRV-Cross" ]; then
    git clone https://github.com/KhronosGroup/SPIRV-Cross.git
fi

if should_build DOVI_TOOL && [ ! -d "dovi_tool-2.3.1" ]; then
    wget -O dovi_tool-2.3.1.tar.gz https://github.com/quietvoid/dovi_tool/archive/refs/tags/2.3.1.tar.gz
    tar xf dovi_tool-2.3.1.tar.gz
fi

if should_build LIBJPEG_TURBO && [ ! -d "libjpeg-turbo-3.1.1" ]; then
    wget https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.1.1/libjpeg-turbo-3.1.1.tar.gz
    tar xf libjpeg-turbo-3.1.1.tar.gz
fi

if should_build LCMS2 && [ ! -d "lcms2-2.17" ]; then
    wget https://github.com/mm2/Little-CMS/releases/download/lcms2.17/lcms2-2.17.tar.gz
    tar xf lcms2-2.17.tar.gz
fi

if should_build VULKAN_LOADER && [ ! -d "Vulkan-Loader-1.3.295" ]; then
    wget -O Vulkan-Loader-v1.3.295.tar.gz https://github.com/KhronosGroup/Vulkan-Loader/archive/refs/tags/v1.3.295.tar.gz
    tar xf Vulkan-Loader-v1.3.295.tar.gz
fi

if should_build ZIMG && [ ! -d "zimg-3.0.6" ]; then
    wget -O zimg-3.0.6.tar.gz https://github.com/sekrit-twc/zimg/archive/refs/tags/release-3.0.6.tar.gz
    tar xf zimg-3.0.6.tar.gz
    mv zimg-release-3.0.6 zimg-3.0.6
fi

# 依存関係は以下の通り
# [ libjpeg -> lcms2 ], shaderc, SPIRV-Cross, dovi_tool, libxxhash, vulkan-loader -> libplacebo
# shadercがあればglslangは不要
if should_build LIBPLACEBO && [ ! -d "libplacebo" ]; then
    git clone --recursive https://code.videolan.org/videolan/libplacebo
    cd libplacebo && git checkout tags/v7.351.0 && cd ..
fi

if should_build VVENC && [ ! -d "vvenc-1.13.1" ]; then
    wget -O vvenc-v1.13.1.tar.gz https://github.com/fraunhoferhhi/vvenc/archive/refs/tags/v1.13.1.tar.gz
    tar xf vvenc-v1.13.1.tar.gz
fi

if should_build SVT_AV1 && [ ! -d "svt-av1" ]; then
    wget https://gitlab.com/AOMediaCodec/SVT-AV1/-/archive/v3.1.0/SVT-AV1-v3.1.0.tar.gz
    tar xf SVT-AV1-v3.1.0.tar.gz
    mv SVT-AV1-v3.1.0 svt-av1
fi

if should_build XVIDCORE && [ ! -d "xvidcore" ]; then
    wget https://downloads.xvid.com/downloads/xvidcore-1.3.7.tar.gz
    tar xf xvidcore-1.3.7.tar.gz
fi
if should_build X264 && [ ! -d "x264" ]; then
    git clone https://code.videolan.org/videolan/x264.git
fi
if should_build X265 && [ ! -d "x265" ]; then
    git clone https://bitbucket.org/multicoreware/x265_git.git x265
fi

# --- 出力先を準備 --------------------------------------
if [ $BUILD_ALL != "FALSE" ]; then
    rm -rf $BUILD_DIR/$TARGET_ARCH
fi

if [ ! -d $BUILD_DIR/$TARGET_ARCH ]; then
    mkdir $BUILD_DIR/$TARGET_ARCH
fi
cd $BUILD_DIR/$TARGET_ARCH
# --- 出力先の古いデータを削除 ----------------------
if [ $UPDATE_FFMPEG != "FALSE" ] && [ -d ffmpeg_test ]; then
    rm -rf ffmpeg_test
fi
if [ ! -d ffmpeg_test ]; then
    cp -r ../src/ffmpeg ffmpeg_test
fi

if [ -d $FFMPEG_DIR_NAME ]; then
    rm -rf $FFMPEG_DIR_NAME
fi
cp -r ../src/ffmpeg $FFMPEG_DIR_NAME

if [ $ADD_TLVMMT = "TRUE" ]; then
    cd $FFMPEG_DIR_NAME
    echo "Patch ffmpeg_tlvmmt.diff..."
    patch -p1 < $PATCHES_DIR/ffmpeg_tlvmmt.diff
    echo "Patch ffmpeg_tlvmmt_asset_group_desc.diff..."
    patch -p1 < $PATCHES_DIR/ffmpeg_tlvmmt_asset_group_desc.diff
    read -p "Check patch and hit enter: "
fi
  
  #$BUILD_DIR/src/soxr* $BUILD_DIR/src/nettle* $BUILD_DIR/src/gnutls*


# --- ビルド開始 対象のフォルダがなければビルドを行う -----------
# if [ ! -d "zlib" ]; then
    # cd $BUILD_DIR/$TARGET_ARCH
    # find ../src/ -type d -name "zlib-*" | xargs -i cp -r {} ./zlib
    # cd $BUILD_DIR/$TARGET_ARCH/zlib
    # CFLAGS="${BUILD_CCFLAGS_SMALL}" \
    # CPPFLAGS="${BUILD_CCFLAGS_SMALL}" \
    # CXXFLAGS="${BUILD_CCFLAGS_SMALL}" \
    # make -f win32/Makefile.gcc
    # rm -f $INSTALL_DIR/lib/libz.a
    # rm -f $INSTALL_DIR/include/zlib.h $INSTALL_DIR/include/zconf.h
    # cp libz.a $INSTALL_DIR/lib/
    # cp zlib.h zconf.h $INSTALL_DIR/include/
# fi

cd $BUILD_DIR/$TARGET_ARCH
if should_build ZLIB && [ ! -d "zlib" ]; then
    find ../src/ -type d -name "zlib-*" | xargs -i cp -r {} ./zlib
    start_build "zlib"
    cd ./zlib
    CFLAGS="${BUILD_CCFLAGS_SMALL}" \
    CPPFLAGS="${BUILD_CCFLAGS_SMALL}" \
    CXXFLAGS="${BUILD_CCFLAGS_SMALL}" \
    ./configure --static --prefix=$INSTALL_DIR
    make -j$NJOBS && make install
fi

cd $BUILD_DIR/$TARGET_ARCH
if should_build BZIP2 && [ ! -d "bzip2" ]; then
    find ../src/ -type d -name "bzip2-*" | xargs -i cp -r {} ./bzip2
    start_build "bzip2"
    cd ./bzip2
    if [ "$MINGWDIR" != "" ]; then
        patch -p1 < $PATCHES_DIR/bzip2-makefile.diff
    fi
    CFLAGS="${BUILD_CCFLAGS_SMALL}" \
    CPPFLAGS="${BUILD_CCFLAGS_SMALL}" \
    CXXFLAGS="${BUILD_CCFLAGS_SMALL}" \
    make -j$NJOBS && make PREFIX=$INSTALL_DIR install
fi

cd $BUILD_DIR/$TARGET_ARCH
if should_build LIBPNG && [ ! -d "libpng" ]; then
    find ../src/ -type d -name "libpng-*" | xargs -i cp -r {} ./libpng
    start_build "libpng"
    cd ./libpng
    PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig \
    CFLAGS="${BUILD_CCFLAGS_SMALL}" \
    CPPFLAGS="${BUILD_CCFLAGS_SMALL}" \
    CXXFLAGS="${BUILD_CCFLAGS_SMALL}" \
    LDFLAGS="${BUILD_LDFLAGS}" \
    ./configure \
    --prefix=$INSTALL_DIR \
    --enable-static \
    --disable-shared
    make -j$NJOBS && make install
fi

# cd $BUILD_DIR/$TARGET_ARCH
# if [ ! -d "gperf" ]; then
    # find ../src/ -type d -name "gperf-*" | xargs -i cp -r {} ./gperf
    # start_build "gperf"
    #libiconvにgperf.exeが必要
    #3.0.4必須 (3.1だと、fontconfigでエラーが出る場合がある)
    # cd ./gperf
    # CFLAGS="${BUILD_CCFLAGS}" \
    # CPPFLAGS="${BUILD_CCFLAGS}" \
    # CXXFLAGS="${BUILD_CCFLAGS}" \
    # ./configure \
    # --prefix=$INSTALL_DIR \
    # --enable-static \
    # --disable-shared
    # make -j$NJOBS
    # texがないとのエラーが出るが無視する
    # make install
# fi

cd $BUILD_DIR/$TARGET_ARCH
if should_build EXPAT && [ ! -d "expat" ]; then
    find ../src/ -type d -name "expat-*" | xargs -i cp -r {} ./expat
    start_build "expat"
    cd ./expat
    CFLAGS="${BUILD_CCFLAGS_SMALL}" \
    CPPFLAGS="${BUILD_CCFLAGS_SMALL}" \
    CXXFLAGS="${BUILD_CCFLAGS_SMALL}" \
    ./configure \
    --prefix=$INSTALL_DIR \
    --enable-static \
    --disable-shared \
    --without-docbook \
    --without-xmlwf \
    --without-examples \
    --without-tests \
    --without-getrandom 
    make -j$NJOBS && make install
fi

cd $BUILD_DIR/$TARGET_ARCH
if should_build FREETYPE && [ ! -d "freetype" ]; then
    find ../src/ -type d -name "freetype-*" | xargs -i cp -r {} ./freetype
    start_build "freetype"
    #msys側のzlib(zlib.h, zconf.h, libz.a, libz.pcを消さないとうまくいかない)
    #あるいはconfigure後に、build/unix/unix-cc.mk内の
    #CFLAGSから-IC:/.../MSYS/includeとLDFLAGSの-LC:/.../MSYS/libを消す
    cd ./freetype
    ZLIB_CFLAGS=" -I${INSTALL_DIR}/include" \
    ZLIB_LIBS="-L${INSTALL_DIR}/lib -lz" \
    BZIP2_CFLAGS=" -I${INSTALL_DIR}/include" \
    BZIP2_LIBS="-L${INSTALL_DIR}/lib -lbz2" \
    LIBPNG_CFLAGS=" -I${INSTALL_DIR}/include" \
    LIBPNG_LIBS="-L${INSTALL_DIR}/lib -lpng -lz" \
    CFLAGS="${BUILD_CCFLAGS_SMALL}" \
    CPPFLAGS="${BUILD_CCFLAGS_SMALL}" \
    CXXFLAGS="${BUILD_CCFLAGS_SMALL}" \
    ./configure \
    --prefix=$INSTALL_DIR \
    --enable-static \
    --disable-shared \
    --with-png=yes \
    --with-zlib=yes \
    --with-bzip2=yes \
    --with-harfbuzz=no \
    --with-brotli=no
    make -j$NJOBS && make install
    if [ "$MINGWDIR" != "" ]; then
        sed -i -e "s/ -lfreetype$/ -lfreetype -liconv -lpng -lbz2 -lz/g" $INSTALL_DIR/lib/pkgconfig/freetype2.pc
    else
        sed -i -e "s/ -lfreetype$/ -lfreetype -lpng -lbz2 -lz/g" $INSTALL_DIR/lib/pkgconfig/freetype2.pc
    fi
fi

cd $BUILD_DIR/$TARGET_ARCH
if should_build LIBICONV && [ ! -d "libiconv" ]; then
    find ../src/ -type d -name "libiconv-*" | xargs -i cp -r {} ./libiconv
    start_build "libiconv"
    cd ./libiconv
    if [ "$MINGWDIR" != "" ]; then
        gzip -dc $PATCHES_DIR/libiconv-1.16-ja-1.patch.gz | patch -p1
    fi
    PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig \
    CFLAGS="${BUILD_CCFLAGS_SMALL} -std=gnu17" \
    CPPFLAGS="${BUILD_CCFLAGS_SMALL}" \
    ./configure \
    --prefix=$INSTALL_DIR \
    --enable-static \
    --disable-shared
    make -j$NJOBS && make install
fi

cd $BUILD_DIR/$TARGET_ARCH
if should_build FONTCONFIG && [ ! -d "fontconfig" ]; then
    find ../src/ -type d -name "fontconfig-*" | xargs -i cp -r {} ./fontconfig
    start_build "fontconfig"
    cd ./fontconfig
    autoreconf -fvi
    PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig \
    FREETYPE_CFLAGS=-I$INSTALL_DIR/include/freetype2 \
    FREETYPE_LIBS="-L$INSTALL_DIR/lib -lfreetype" \
    EXPAT_CFLAGS="-I$INSTALL_DIR/include" \
    EXPAT_LIBS="-L$INSTALL_DIR/lib -lexpat" \
    CFLAGS="${BUILD_CCFLAGS_SMALL}" \
    CPPFLAGS="${BUILD_CCFLAGS_SMALL}" \
    CXXFLAGS="${BUILD_CCFLAGS_SMALL}" \
    LDFLAGS="${BUILD_LDFLAGS}" \
    ./configure \
    --prefix=$INSTALL_DIR \
    --disable-shared \
    --enable-static \
    --enable-iconv --with-libiconv-includes=$INSTALL_DIR/include \
    --disable-docs \
    --disable-libxml2
    make -j$NJOBS && make install
    #pkgconfig情報を書き換える
    sed -i -e "s/ -lfontconfig$/ -lfontconfig -lexpat -lpng -lz/g" $INSTALL_DIR/lib/pkgconfig/fontconfig.pc
    if [ "$MINGWDIR" != "" ]; then
        sed -i -e "s/ -lfreetype$/ -lfreetype -liconv -lpng -lz/g" $INSTALL_DIR/lib/pkgconfig/fontconfig.pc
    else
        sed -i -e "s/ -lfreetype$/ -lfreetype -lpng -lz/g" $INSTALL_DIR/lib/pkgconfig/fontconfig.pc
    fi
    sed -i -e "s/^Requires:[ \f\n\r\t]\+freetype2/Requires: freetype2 libpng/g" $INSTALL_DIR/lib/pkgconfig/fontconfig.pc
fi

cd $BUILD_DIR/$TARGET_ARCH
if should_build FRIBIDI && [ ! -d "fribidi" ]; then
    find ../src/ -type d -name "fribidi-*" | xargs -i cp -r {} ./fribidi
    start_build "fribidi"
    cd ./fribidi
    autoreconf -fvi
    PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig \
    CFLAGS="${BUILD_CCFLAGS_SMALL}" \
    CPPFLAGS="${BUILD_CCFLAGS_SMALL}" \
    ./configure \
    --prefix=$INSTALL_DIR \
    --enable-static \
    --enable-shared=no
    make -j$NJOBS && make install
fi

# cd $BUILD_DIR/$TARGET_ARCH
# if [ ! -d "graphite2" ]; then
#     find ../src/ -type d -name "graphite2-*" | xargs -i cp -r {} ./graphite2
#     cd ./graphite2
#     sed -i '/cmptest/d' tests/CMakeLists.txt
#     sed -i '/cmake_policy(SET CMP0012 NEW)/d' CMakeLists.txt
#     sed -i 's/PythonInterp/Python3/' CMakeLists.txt
#     find . -name CMakeLists.txt | xargs sed -i 's/VERSION 2.8.0 FATAL_ERROR/VERSION 4.0.0/'
#     sed -i '/Font.h/i #include <cstdint>' tests/featuremap/featuremaptest.cpp
#     mkdir build && cd build
#     cmake -G "${CMAKE_GENERATOR}" -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -DCMAKE_BUILD_TYPE=Release -DENABLE_SHARED=OFF -DENABLE_STATIC=ON ..
#     make -j$NJOBS && make install
#     read -p "Check install and hit enter: "
# fi

# x86では、libass.dllのビルド(リンク)に失敗するため、x64でのみビルドする
if should_build HARFBUZZ; then
    cd $BUILD_DIR/$TARGET_ARCH
    if [ ! -d "harfbuzz" ]; then
        find ../src/ -type d -name "harfbuzz-*" | xargs -i cp -r {} ./harfbuzz
        start_build "harfbuzz"
        cd ./harfbuzz
        CC=gcc \
        CXX=g++ \
        PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig \
        CFLAGS="${BUILD_CCFLAGS_SMALL} -I${INSTALL_DIR}/include" \
        CPPFLAGS="${BUILD_CCFLAGS_SMALL} -I${INSTALL_DIR}/include" \
        LDFLAGS="${BUILD_LDFLAGS} -L${INSTALL_DIR}/lib" \
        meson build --buildtype release
        meson configure build/ --prefix=$INSTALL_DIR --libdir=lib -Dbuildtype=release -Ddefault_library=static -Dglib=disabled -Dcairo=disabled -Dfreetype=enabled -Ddocs=disabled -Dtests=disabled -Dc_args="${BUILD_CCFLAGS_SMALL}" -Dc_link_args="${BUILD_LDFLAGS}"
        ninja -C build install
    fi
fi
if should_build LIBUNIBREAK; then
    cd $BUILD_DIR/$TARGET_ARCH
    if [ ! -d "libunibreak" ]; then
        find ../src/ -type d -name "libunibreak-*" | xargs -i cp -r {} ./libunibreak
        start_build "libunibreak"
        cd ./libunibreak
        CFLAGS="${BUILD_CCFLAGS_SMALL}" \
        CPPFLAGS="${BUILD_CCFLAGS_SMALL}" \
        LDFLAGS="${BUILD_LDFLAGS}" \
        ./configure \
        --prefix=$INSTALL_DIR \
        --enable-static \
        --enable-shared=no
        make -j$NJOBS && make install
    fi
fi

cd $BUILD_DIR/$TARGET_ARCH
if should_build LIBASS && [ ! -d "libass" ]; then
    find ../src/ -type d -name "libass-${LIBASS_VERSION}" | xargs -i cp -r {} ./libass
    start_build "libass"
    cd ./libass
    autoreconf -fvi
    PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig \
    CFLAGS="${BUILD_CCFLAGS_SMALL} -I${INSTALL_DIR}/include" \
    CPPFLAGS="${BUILD_CCFLAGS_SMALL} -I${INSTALL_DIR}/include" \
    LDFLAGS="${BUILD_LDFLAGS} -L${INSTALL_DIR}/lib" \
    ./configure \
    --prefix=$INSTALL_DIR \
    --enable-static \
    --enable-shared=no
    make -j$NJOBS && make install
fi

if should_build LIBASS_DLL && [ ! -d "libass_dll" ]; then
    find ../src/ -type d -name "libass-${LIBASS_VERSION}" | xargs -i cp -r {} ./libass_dll
    start_build "libass_dll"
    cd $BUILD_DIR/$TARGET_ARCH/libass_dll
    autoreconf -fvi
    PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig \
    CC="gcc -static-libgcc -static-libstdc++" \
    CFLAGS="${BUILD_CCFLAGS_SMALL}" \
    CPPFLAGS="${BUILD_CCFLAGS_SMALL}" \
    LDFLAGS="-L${INSTALL_DIR}/lib -static-libgcc -static-libstdc++ -Wl,-Bstatic -Wl,-lm,-liconv,-lfreetype,-lfribidi,-lfontconfig,-lexpat,-lfreetype,-lpng,-lbz2,-lz" \
    ./configure \
    --prefix=$INSTALL_DIR \
    --enable-static=no \
    --enable-shared=yes
    #実行したコマンドを出力するように
    sed -i -e 's/AM_DEFAULT_VERBOSITY = 0/AM_DEFAULT_VERBOSITY = 1/g' libass/Makefile
    make -j$NJOBS
    cd libass
    # ../libtool --tag=CC   --mode=link gcc -std=gnu99 \
    # -D_GNU_SOURCE \
    # ${BUILD_CCFLAGS_SMALL} \
    # -I${INSTALL_DIR}/include/freetype2 \
    # -I${INSTALL_DIR}/include/fribidi \
    # -I${INSTALL_DIR}/include \
    # -I${INSTALL_DIR}/include/freetype2 \
    # -no-undefined -version-info 8:0:3 -export-symbols ./libass.sym  -o libass.la -rpath ${INSTALL_DIR}/lib \
    # `find ./ -name "*.lo" | tr '\n' ' '` \
    # -L${INSTALL_DIR}/lib \
    # -static -static-libgcc -static-libstdc++ \
    # -Wl,-lm,-liconv,-lfreetype,-lfribidi,-lfontconfig,-lexpat,-lfreetype,-lpng,-lbz2,-lz \
    # -Wl,--output-def,libass.def -Wl,-s -Wl,-gc-sections
    # sed -i -e "s/ @[^ ]*//" libass.def

    LIBASS_DEF_FILENAME=`find ./.libs/libass-*.dll`.def
    LIBASS_DEF_FILENAME=${LIBASS_DEF_FILENAME/.dll.def/.def}
    cp -f `find ./.libs/libass-*.dll`.def ${LIBASS_DEF_FILENAME}
    cp -f ${LIBASS_DEF_FILENAME} .
    LIBASS_DEF_FILENAME=`basename $LIBASS_DEF_FILENAME`
    sed -i -e "s/ @[^ ]*//" ${LIBASS_DEF_FILENAME}
    LIBASS_LIB_FILENAME=$(basename $LIBASS_DEF_FILENAME .def).lib
    lib.exe -machine:$TARGET_ARCH -def:$LIBASS_DEF_FILENAME -out:$LIBASS_LIB_FILENAME
    cp `find ./.libs/libass-*.dll` .
fi

cd $BUILD_DIR/$TARGET_ARCH
if should_build OPUS && [ ! -d "opus" ]; then
    find ../src/ -type d -name "opus-*" | xargs -i cp -r {} ./opus
    start_build "opus"
    cd ./opus
    autoreconf -fvi
    CFLAGS="${BUILD_CCFLAGS} -fno-tree-vectorize -fno-fast-math" \
    CPPFLAGS="${BUILD_CCFLAGS} -fno-tree-vectorize -fno-fast-math" \
    CXXFLAGS="${BUILD_CCFLAGS} -fno-tree-vectorize -fno-fast-math" \
    ./configure \
    --prefix=$INSTALL_DIR \
    --enable-static=yes \
    --enable-shared=no \
    --disable-doc \
    --disable-extra-programs
    make -j$NJOBS && make install
fi

cd $BUILD_DIR/$TARGET_ARCH
if should_build LIBOGG && [ ! -d "libogg" ]; then
    find ../src/ -type d -name "libogg-*" | xargs -i cp -r {} ./libogg
    start_build "libogg"
    cd ./libogg
    autoreconf -fvi
    CFLAGS="${BUILD_CCFLAGS}" \
    CPPFLAGS="${BUILD_CCFLAGS}" \
    LDFLAGS="${BUILD_LDFLAGS}" \
    PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig \
    ./configure --prefix=$INSTALL_DIR \
        --disable-shared
    make -j$NJOBS && make install-strip
fi

cd $BUILD_DIR/$TARGET_ARCH
if should_build LIBVORBIS && [ ! -d "libvorbis" ]; then
    find ../src/ -type d -name "libvorbis-*" | xargs -i cp -r {} ./libvorbis
    start_build "libvorbis"
    cd ./libvorbis
    CFLAGS="${BUILD_CCFLAGS}" \
    CPPFLAGS="${BUILD_CCFLAGS}" \
    LDFLAGS="${BUILD_LDFLAGS}" \
    PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig \
    ./configure --prefix=$INSTALL_DIR \
        --disable-shared
    make -j$NJOBS && make install-strip
    sed -i -e "s/^Requires.private/Requires/g" $INSTALL_DIR/lib/pkgconfig/vorbis.pc
    sed -i -e "s/^Requires.private/Requires/g" $INSTALL_DIR/lib/pkgconfig/vorbisfile.pc
    sed -i -e "s/^Requires.private/Requires/g" $INSTALL_DIR/lib/pkgconfig/vorbisenc.pc
fi

cd $BUILD_DIR/$TARGET_ARCH
if should_build SPEEX && [ ! -d "speex" ]; then
    find ../src/ -type d -name "speex-*" | xargs -i cp -r {} ./speex
    start_build "speex"
    cd ./speex
    autoreconf -fvi
    CFLAGS="${BUILD_CCFLAGS}" \
    CPPFLAGS="${BUILD_CCFLAGS}" \
    LDFLAGS="${BUILD_LDFLAGS}" \
    PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig \
    ./configure --prefix=$INSTALL_DIR \
        --disable-shared
    make -j$NJOBS
    make install-strip
fi

cd $BUILD_DIR/$TARGET_ARCH
if should_build LAME && [ ! -d "lame" ]; then
    find ../src/ -type d -name "lame-*" | xargs -i cp -r {} ./lame
    start_build "lame"
    cd ./lame
    if [ "$MINGWDIR" != "" ]; then
        patch -p1 < $PATCHES_DIR/lame-3.100-parse_c.diff
    fi
    CFLAGS="${BUILD_CCFLAGS}" \
    CPPFLAGS="${BUILD_CCFLAGS}" \
    LDFLAGS="${BUILD_LDFLAGS}" \
     ./configure \
     --prefix=$INSTALL_DIR \
     --disable-shared \
     --enable-static \
     --disable-decoder
    make install -j$NJOBS
fi

cd $BUILD_DIR/$TARGET_ARCH
if should_build LIBSNDFILE && [ ! -d "libsndfile" ]; then
    find ../src/ -type d -name "libsndfile-*" | xargs -i cp -r {} ./libsndfile
    start_build "libsndfile"
    cd ./libsndfile
    CFLAGS="${BUILD_CCFLAGS}" \
    CPPFLAGS="${BUILD_CCFLAGS}" \
    LDFLAGS="${BUILD_LDFLAGS}" \
    PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig \
     ./configure \
     --prefix=$INSTALL_DIR \
     --disable-shared \
     --enable-static
    make install -j$NJOBS
fi

cd $BUILD_DIR/$TARGET_ARCH
if should_build TWOLAME && [ ! -d "twolame" ]; then
    find ../src/ -type d -name "twolame-*" | xargs -i cp -r {} ./twolame
    start_build "twolame"
    cd ./twolame
    if [ "$MINGWDIR" != "" ]; then
        patch -p1 < $PATCHES_DIR/twolame-0.4.0-mingw.diff
    fi
    PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig \
    CFLAGS="${BUILD_CCFLAGS}" \
    CPPFLAGS="${BUILD_CCFLAGS}" \
    LDFLAGS="${BUILD_LDFLAGS}" \
     ./configure \
     --prefix=$INSTALL_DIR \
     --disable-shared \
     --enable-static
    make install -j$NJOBS
fi

cd $BUILD_DIR/$TARGET_ARCH
if should_build SOXR && [ ! -d "soxr" ]; then
    find ../src/ -type d -name "soxr-*" | xargs -i cp -r {} ./soxr
    start_build "soxr"
    cd ./soxr
    which cmake
    cmake --version
    cmake -G "${CMAKE_GENERATOR}" \
    -D BUILD_SHARED_LIBS:BOOL=FALSE \
    -D CMAKE_C_FLAGS_RELEASE:STRING="${BUILD_CCFLAGS}" \
    -D CMAKE_EXE_LINKER_FLAGS_RELEASE:STRING="${BUILD_LDFLAGS}" \
    -D WITH_OPENMP:BOOL=NO \
    -D BUILD_TESTS:BOOL=NO \
    -D CMAKE_INSTALL_PREFIX=$INSTALL_DIR \
    -D CMAKE_POLICY_VERSION_MINIMUM=3.5 \
    .
    make install -j$NJOBS
    # staticリンク時にlibmが必要
    sed -i -e '/^Libs:/ s/$/ -lm/' ${INSTALL_DIR}/lib/pkgconfig/soxr.pc
fi

cd $BUILD_DIR/$TARGET_ARCH
if should_build LIBXML2 && [ ! -d "libxml2" ]; then
    find ../src/ -type d -name "libxml2-*" | xargs -i cp -r {} ./libxml2
    start_build "libxml2"
    cd ./libxml2
    if [ -e configure ]; then
        echo "configure already exists, skip autoreconf"
    else
        ./autogen.sh --without-python
    fi
    CFLAGS="${BUILD_CCFLAGS_SMALL}" \
    CPPFLAGS="${BUILD_CCFLAGS_SMALL}" \
    LDFLAGS="${BUILD_LDFLAGS}" \
     ./configure \
     --prefix=$INSTALL_DIR \
     --disable-shared \
     --enable-static \
     --without-python
    make install -j$NJOBS
fi

cd $BUILD_DIR/$TARGET_ARCH
if should_build LIBBLURAY && [ ! -d "libbluray" ]; then
    find ../src/ -type d -name "libbluray-*" | xargs -i cp -r {} ./libbluray
    start_build "libbluray"
    cd ./libbluray
    # Linux static link時にFFmpeg本体のdec_initと衝突するため、libbluray側を名前空間化する
    if [ "$MINGWDIR" = "" ]; then
        sed -i 's/\bdec_init\b/bluray_dec_init/g' src/libbluray/disc/dec.h src/libbluray/disc/dec.c src/libbluray/disc/disc.c
    fi
    autoreconf -fvi
    PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig \
    CFLAGS="${BUILD_CCFLAGS_SMALL}" \
    CPPFLAGS="${BUILD_CCFLAGS_SMALL}" \
    LDFLAGS="${BUILD_LDFLAGS}" \
     ./configure \
     --prefix=$INSTALL_DIR \
     --disable-shared \
     --enable-static \
     --disable-bdjava-jar \
     --disable-doxygen-doc \
     --disable-examples
    make -j$NJOBS
    make install
    if [ ! -f "${INSTALL_DIR}/lib/pkgconfig/libbluray.pc" ]; then
        echo "libbluray.pc is missing after install."
        find "${INSTALL_DIR}" -maxdepth 5 -name "libbluray.pc" -print || true
        exit 1
    fi
fi

cd $BUILD_DIR/$TARGET_ARCH
if should_build ARIBB24 && [ ! -d "aribb24" ]; then
    find ../src/ -type d -name "aribb24-*" | xargs -i cp -r {} ./aribb24
    start_build "aribb24"
    cd ./aribb24
    autoreconf -fvi
    PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig \
    CFLAGS="${BUILD_CCFLAGS_SMALL}" \
    CPPFLAGS="${BUILD_CCFLAGS_SMALL}" \
    LDFLAGS="${BUILD_LDFLAGS}" \
     ./configure \
     --prefix=$INSTALL_DIR \
     --disable-shared \
     --enable-static
    make install -j$NJOBS
    sed -i -e 's/Version: 1.0.3/Version: 1.0.4/g' ${INSTALL_DIR}/lib/pkgconfig/aribb24.pc
fi

cd $BUILD_DIR/$TARGET_ARCH
if should_build LIBARIBCAPTION && [ ! -d "libaribcaption" ]; then
    find ../src/ -type d -name "libaribcaption-*" | xargs -i cp -r {} ./libaribcaption
    start_build "libaribcaption"
    cd ./libaribcaption
    mkdir build && cd build
    CFLAGS="${BUILD_CCFLAGS_SMALL}" \
    CPPFLAGS="${BUILD_CCFLAGS_SMALL}" \
    LDFLAGS="${BUILD_LDFLAGS}" \
    cmake .. -G "${CMAKE_GENERATOR}" -DCMAKE_BUILD_TYPE=Release -DARIBCC_USE_FONTCONFIG=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR
    cmake --build . -j$NJOBS
    cmake --install .
    #sed -i -e 's/-lC:\//-l\/c\//g' ${INSTALL_DIR}/lib/pkgconfig/libaribcaption.pc
    # 下記のような絶対パス指定だとFFmpegの検出でリンク順が崩れるため、静的リンク指定へ正規化する
    #   -lC:/mingw64/.../libstdc++.a
    #   /usr/lib/gcc/x86_64-linux-gnu/*/libstdc++.a
    if [ "$MINGWDIR" = "" ] && [ -f "$LIBSTDCXX_A" ]; then
        sed -i -E \
            -e "s#-l[A-Z]:/.*/libstdc\\+\\+\\.a#${LIBSTDCXX_STATIC_FLAGS}#g" \
            -e "s#/usr/lib/gcc/[^ ]+/[0-9.]+/libstdc\\+\\+\\.a#${LIBSTDCXX_STATIC_FLAGS}#g" \
            -e "s#-lstdc\\+\\+#${LIBSTDCXX_STATIC_FLAGS}#g" \
            ${INSTALL_DIR}/lib/pkgconfig/libaribcaption.pc
    else
        sed -i -E \
            -e 's#-l[A-Z]:/.*/libstdc\+\+\.a#-lstdc++#g' \
            -e 's#/usr/lib/gcc/[^ ]+/[0-9.]+/libstdc\+\+\.a#-lstdc++#g' \
            ${INSTALL_DIR}/lib/pkgconfig/libaribcaption.pc
    fi
fi

cd $BUILD_DIR/$TARGET_ARCH
if should_build DAV1D && [ ! -d "dav1d" ]; then
    find ../src/ -type d -name "dav1d-*" | xargs -i cp -r {} ./dav1d
    start_build "dav1d"
    cd ./dav1d
    CC=gcc \
    CXX=g++ \
    CFLAGS="${BUILD_CCFLAGS}" \
    CPPFLAGS="${BUILD_CCFLAGS}" \
    LDFLAGS="${BUILD_LDFLAGS}" \
    meson build --buildtype release
    meson configure build/ --prefix=$INSTALL_DIR -Dbuildtype=release -Ddefault_library=static -Denable_examples=false -Denable_tests=false -Dc_args="${BUILD_CCFLAGS}"
    ninja -C build install
fi

cd $BUILD_DIR/$TARGET_ARCH
if should_build LIBVPL && [ ! -d "libvpl" ]; then
    find ../src/ -type d -name "libvpl-*" | xargs -i cp -r {} ./libvpl
    start_build "libvpl"
    cd libvpl
    #script/bootstrap
    cmake -G "${CMAKE_GENERATOR}" -B _build -DBUILD_SHARED_LIBS=OFF -DUSE_MSVC_STATIC_RUNTIME=ON -DCMAKE_BUILD_TYPE=Release -DINSTALL_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR
    cmake --build _build --config Release
    cmake --install _build --config Release
    # x86版の場合、$INSTALL_DIR/libに入るべきものが$INSTALL_DIR/lib/x86に入ってしまう
    # あとから強制的に移動する
    # vpl.pcのパスも移動に合わせる
    if [ $TARGET_ARCH = "x86" ]; then
        cp -r $INSTALL_DIR/lib/x86/* $INSTALL_DIR/lib/
        rm -rf $INSTALL_DIR/lib/x86
        sed -i -e 's/${pcfiledir}\/../${pcfiledir}/g' $INSTALL_DIR/lib/pkgconfig/vpl.pc
    fi
    if [ "$MINGWDIR" = "" ] && [ -f "$LIBSTDCXX_A" ]; then
        sed -i -e "s#^Libs:.*#Libs: -L\${libdir} -lvpl ${LIBSTDCXX_STATIC_FLAGS} -ldl#g" $INSTALL_DIR/lib/pkgconfig/vpl.pc
    else
        sed -i -e 's/-lvpl/-lvpl -lstdc++/g' $INSTALL_DIR/lib/pkgconfig/vpl.pc
    fi
    # ffmpegで参照するpkg-configをここで正規化しておく
    normalize_static_libstdcxx_pc_dir "$INSTALL_DIR/lib/pkgconfig"
fi

cd $BUILD_DIR/$TARGET_ARCH
if should_build LIBVPX && [ ! -d "libvpx" ]; then
    find ../src/ -type d -name "libvpx-*" | xargs -i cp -r {} ./libvpx
    start_build "libvpx"
    cd ./libvpx
    CFLAGS="${BUILD_CCFLAGS}" \
    CPPFLAGS="${BUILD_CCFLAGS}" \
    LDFLAGS="${BUILD_LDFLAGS}" \
     ./configure \
     --prefix=$INSTALL_DIR \
     --disable-shared \
     --enable-static \
     --disable-docs \
     --disable-examples \
     --disable-tools \
     --disable-unit-tests \
     --enable-vp9-highbitdepth \
     --enable-runtime-cpu-detect
    make install -j$NJOBS
fi

cd $BUILD_DIR/$TARGET_ARCH
if should_build NV_CODEC_HEADERS && [ ! -d "nv-codec-headers" ]; then
    find ../src/ -type d -name "nv-codec-headers-*" | xargs -i cp -r {} ./nv-codec-headers
    start_build "nv-codec-headers"
    cd nv-codec-headers
    make PREFIX=$INSTALL_DIR install
fi

cd $BUILD_DIR/$TARGET_ARCH
if should_build LIBXXHASH && [ ! -d "libxxhash" ]; then
    find ../src/ -type d -name "libxxhash-*" | xargs -i cp -r {} ./libxxhash
    start_build "libxxhash"
    cd ./libxxhash
    CC=gcc \
    CXX=g++ \
    CFLAGS="${BUILD_CCFLAGS} -DXXH_STATIC_LINKING_ONLY" \
    CPPFLAGS="${BUILD_CCFLAGS} -DXXH_STATIC_LINKING_ONLY" \
    LDFLAGS="${BUILD_LDFLAGS}" \
    PREFIX=$INSTALL_DIR \
    prefix=$INSTALL_DIR \
    DISPATCH=1 \
    make 
    PREFIX=$INSTALL_DIR \
    prefix=$INSTALL_DIR \
    make install
    if [ "$MINGWDIR" = "" ]; then
        # Linux静的リンク用途では共有ライブラリを除去
        rm -f ${INSTALL_DIR}/lib/libxxhash.so*
    fi
fi

cd $BUILD_DIR/$TARGET_ARCH
if should_build DOVI_TOOL && [ ! -d "dovi_tool" ]; then
    find ../src/ -type d -name "dovi_tool-*" | xargs -i cp -r {} ./dovi_tool
    start_build "dovi_tool"
    cd ./dovi_tool/dolby_vision
    cargo cinstall --target ${CARGOC_TARGET} --release --prefix=$INSTALL_DIR
    # dllを削除し、staticライブラリのみを残す
    if [ "$MINGWDIR" != "" ]; then
        rm $INSTALL_DIR/lib/dovi.dll.a
        rm $INSTALL_DIR/lib/dovi.def
        rm $INSTALL_DIR/bin/dovi.dll
    else
        rm $INSTALL_DIR/lib/libdovi.so*
    fi
    # static link向けにdovi.pcを編集
    LIBDOVI_STATIC_LIBS=`awk -F':' '/^Libs.private:/{print $2}' ${INSTALL_DIR}/lib/pkgconfig/dovi.pc`
    sed -i -e "s/-ldovi/-ldovi ${LIBDOVI_STATIC_LIBS}/g" ${INSTALL_DIR}/lib/pkgconfig/dovi.pc

    #dllからlibファイルを作成
    #cd target/${CARGOC_TARGET}/release
    #DOVI_DLL_FILENAME=dovi.dll
    #DOVI_DEF_FILENAME=dovi.def
    #DOVI_LIB_FILENAME=$(basename $DOVI_DEF_FILENAME .def).lib
    #lib.exe -machine:$TARGET_ARCH -def:$DOVI_DEF_FILENAME -out:$DOVI_LIB_FILENAME
fi

if should_build GLSLANG; then
  cd $BUILD_DIR/$TARGET_ARCH
  if [ ! -d "glslang" ]; then
      find ../src/ -type d -name "glslang-*" | xargs -i cp -r {} ./glslang
      start_build "glslang"
      cd ./glslang
      ./update_glslang_sources.py
      mkdir -p build && cd build
      PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig \
      CFLAGS="${BUILD_CCFLAGS}" \
      CPPFLAGS="${BUILD_CCFLAGS}" \
      LDFLAGS="${BUILD_LDFLAGS}" \
      cmake ../ -G "${CMAKE_GENERATOR}" -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DINSTALL_GTEST=OFF -DGLSLANG_TESTS=OFF
      make -j$NJOBS && make install
  fi
fi

cd $BUILD_DIR/$TARGET_ARCH
if should_build LIBJPEG_TURBO && [ ! -d "libjpeg-turbo" ]; then
    find ../src/ -type d -name "libjpeg-*" | xargs -i cp -r {} ./libjpeg-turbo
    start_build "libjpeg-turbo"
    cd ./libjpeg-turbo
    mkdir build && cd build
    PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig \
    CFLAGS="${BUILD_CCFLAGS}" \
    CPPFLAGS="${BUILD_CCFLAGS}" \
    LDFLAGS="${BUILD_LDFLAGS}" \
    cmake -G "${CMAKE_GENERATOR}" -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -DCMAKE_BUILD_TYPE=Release -DENABLE_SHARED=OFF -DENABLE_STATIC=ON ..
    make -j$NJOBS && make install
fi

cd $BUILD_DIR/$TARGET_ARCH
if should_build LCMS2 && [ ! -d "lcms2" ]; then
    find ../src/ -type d -name "lcms2*" | xargs -i cp -r {} ./lcms2
    start_build "lcms2"
    cd ./lcms2
    CC=gcc \
    CXX=g++ \
    PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig \
    CFLAGS="${BUILD_CCFLAGS}" \
    CPPFLAGS="${BUILD_CCFLAGS} -I${INSTALL_DIR}/include" \
    LDFLAGS="${BUILD_LDFLAGS} -L${INSTALL_DIR}/lib" \
    meson build --buildtype release --prefix=$INSTALL_DIR --libdir=lib -Ddefault_library=static -Dprefer_static=true -Dstrip=true -Dthreaded=false -Dfastfloat=false
    ninja -C build install
fi

cd $BUILD_DIR/$TARGET_ARCH
if should_build SHADERC && [ ! -d "shaderc" ]; then
    find ../src/ -type d -name "shaderc*" | xargs -i cp -r {} ./shaderc
    start_build "shaderc"
    cd ./shaderc
    if [ ! -d "third_party/spirv-tools" ] || [ ! -d "third_party/spirv-headers" ]; then
        "${PYTHON_BIN}" ./utils/git-sync-deps
    fi
    if [ ! -d "third_party/spirv-tools" ] || [ ! -d "third_party/spirv-headers" ]; then
        echo "shaderc dependencies are missing: third_party/spirv-tools or spirv-headers."
        exit 1
    fi
    if [ "$MINGWDIR" != "" ]; then
        patch -p1 < $PATCHES_DIR/shaderc_add_shaderc_util.diff
    fi
    mkdir build && cd build
    PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig \
    CFLAGS="${BUILD_CCFLAGS}" \
    CPPFLAGS="${BUILD_CCFLAGS} -I${INSTALL_DIR}/include" \
    LDFLAGS="${BUILD_LDFLAGS} -L${INSTALL_DIR}/lib" \
    cmake -GNinja -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=OFF -DSHADERC_SKIP_EXAMPLES=ON -DSHADERC_SKIP_TESTS=ON -DSHADERC_SKIP_COPYRIGHT_CHECK=ON -DINSTALL_GTEST=OFF ..
    ninja
    ninja install
    mv -f ${INSTALL_DIR}/lib/pkgconfig/shaderc_static.pc ${INSTALL_DIR}/lib/pkgconfig/shaderc.pc
    if [ "$MINGWDIR" = "" ]; then
        # Linux静的リンク用途では共有ライブラリを除去
        rm -f ${INSTALL_DIR}/lib/libshaderc_shared.so*
    fi
fi

cd $BUILD_DIR/$TARGET_ARCH
if should_build SPIRV_CROSS && [ ! -d "SPIRV-Cross" ]; then
    find ../src/ -type d -name "SPIRV-Cross*" | xargs -i cp -r {} ./SPIRV-Cross
    start_build "SPIRV-Cross"
    cd ./SPIRV-Cross
    mkdir build && cd build
    CC=gcc \
    CXX=g++ \
    PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig \
    CFLAGS="${BUILD_CCFLAGS}" \
    CPPFLAGS="${BUILD_CCFLAGS}" \
    LDFLAGS="${BUILD_LDFLAGS}" \
    cmake -G "${CMAKE_GENERATOR}" -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -DCMAKE_BUILD_TYPE=Release -DSPIRV_CROSS_ENABLE_TESTS=OFF -DSPIRV_CROSS_SHARED=OFF -DSPIRV_CROSS_CLI=OFF ..
    make -j$NJOBS && make install
    sed -i -e 's/-lspirv-cross-c/-lspirv-cross-c -lspirv-cross-msl -lspirv-cross-hlsl -lspirv-cross-cpp -lspirv-cross-glsl -lspirv-cross-util -lspirv-cross-core -lspirv-cross-reflect -lstdc++/g' ${INSTALL_DIR}/lib/pkgconfig/spirv-cross-c.pc
fi

cd $BUILD_DIR/$TARGET_ARCH
if should_build VULKAN_LOADER && [ ! -d "Vulkan-Loader" ]; then
    find ../src/ -type d -name "Vulkan-Loader*" | xargs -i cp -r {} ./Vulkan-Loader
    start_build "Vulkan-Loader"
    cd ./Vulkan-Loader
    patch -p1 < $PATCHES_DIR/vulkan_loader_static.diff
    mkdir build && cd build
    "${PYTHON_BIN}" ../scripts/update_deps.py --no-build
    cd Vulkan-Headers
    cmake -G "${CMAKE_GENERATOR}" -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -DCMAKE_BUILD_TYPE=Release -DVULKAN_HEADERS_ENABLE_MODULE=OFF
    make -j$NJOBS && make install
    cd ..
    CC=gcc \
    CXX=g++ \
    PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig \
    CFLAGS="${BUILD_CCFLAGS} -I${INSTALL_DIR}/include -DUNIX=OFF -DSTRSAFE_NO_DEPRECATE" \
    CPPFLAGS="${BUILD_CCFLAGS} -I${INSTALL_DIR}/include -DUNIX=OFF -DSTRSAFE_NO_DEPRECATE" \
    LDFLAGS="${BUILD_LDFLAGS} -L${INSTALL_DIR}/lib" \
    VULKAN_WSI_OPTIONS=""
    if [ "$MINGWDIR" = "" ]; then
        # Linux静的リンク用: X11/Wayland系WSI依存を無効化し、X11ヘッダ依存を避ける
        VULKAN_WSI_OPTIONS="-DBUILD_WSI_XCB_SUPPORT=OFF -DBUILD_WSI_XLIB_SUPPORT=OFF -DBUILD_WSI_WAYLAND_SUPPORT=OFF -DBUILD_WSI_DIRECTFB_SUPPORT=OFF"
    fi
    cmake -G "${CMAKE_GENERATOR}" -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DUNIX=OFF -DVULKAN_HEADERS_INSTALL_DIR=${INSTALL_DIR} ${VULKAN_WSI_OPTIONS} ..
    make -j$NJOBS && make install
    if [ "$MINGWDIR" = "" ]; then
        # 静的リンク用途では libvulkan.a を優先させるため共有ライブラリを除去
        rm -f ${INSTALL_DIR}/lib/libvulkan.so*
    fi
fi

cd $BUILD_DIR/$TARGET_ARCH
if should_build LIBPLACEBO && [ ! -d "libplacebo" ]; then
    find ../src/ -type d -name "libplacebo*" | xargs -i cp -r {} ./libplacebo
    start_build "libplacebo"
    cd ./libplacebo
    if [ "$MINGWDIR" != "" ]; then
        patch -p1 < $PATCHES_DIR/libplacebo_use_shaderc_combined.diff
        patch -p1 < $PATCHES_DIR/libplacebo_d3d11_build.diff
    fi
    if [ "$MINGWDIR" = "" ]; then
        LIBPLACEBO_D3D11_OPT="-Dd3d11=disabled"
    else
        LIBPLACEBO_D3D11_OPT="-Dd3d11=enabled"
    fi
    CC=gcc \
    CXX=g++ \
    PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig \
    CFLAGS="${BUILD_CCFLAGS}" \
    CPPFLAGS="${BUILD_CCFLAGS} -I${INSTALL_DIR}/include" \
    LDFLAGS="${BUILD_LDFLAGS} -L${INSTALL_DIR}/lib" \
    meson build --buildtype release --prefix=$INSTALL_DIR --libdir=lib ${LIBPLACEBO_D3D11_OPT} -Ddefault_library=static -Dprefer_static=true -Dstrip=true -Ddemos=false -Dtests=false
    ninja -C build install
    #下記のように変更しないと適切にリンクできない
    # C:/mingw64/mingw64/lib/libshlwapi.a -> -llibshlwapi
    sed -i -e "s/[A-Z]:\/.\+\/lib\/libshlwapi\.a/-lshlwapi/g" ${INSTALL_DIR}/lib/pkgconfig/libplacebo.pc
    sed -i -e "s/[A-Z]:\/.\+\/lib\/libversion\.a/-lversion/g" ${INSTALL_DIR}/lib/pkgconfig/libplacebo.pc
fi

if should_build LIBPLACEBO_DLL; then
    cd $BUILD_DIR/$TARGET_ARCH
    if [ ! -d "libplacebo_dll" ]; then
        find ../src/ -type d -name "libplacebo*" | xargs -i cp -r {} ./libplacebo_dll
        start_build "libplacebo_dll"
        cd ./libplacebo_dll
        if [ "$MINGWDIR" != "" ]; then
            patch -p1 < $PATCHES_DIR/libplacebo_use_shaderc_combined.diff
            patch -p1 < $PATCHES_DIR/libplacebo_d3d11_build.diff
        fi
        CC=gcc \
        CXX=g++ \
        PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig \
        CFLAGS="${BUILD_CCFLAGS}" \
        CPPFLAGS="${BUILD_CCFLAGS} -I${INSTALL_DIR}/include" \
        LDFLAGS="${BUILD_LDFLAGS} -L${INSTALL_DIR}/lib" \
        meson build --buildtype release --prefix=$INSTALL_DIR -Dd3d11=enabled -Ddefault_library=shared -Dprefer_static=false -Dstrip=true -Ddemos=false -Dtests=false
        sed -i 's/libstdc++.dll.a/libstdc++.a/g' build/build.ninja
        ninja -C build

        #dllからlib,defファイルを作成
        cd build/src
        LIBPLACEBO_DLL_FILENAME=$(basename `find ./libplacebo-*.dll`)
        LIBPLACEBO_DLL_FILENAME_WITHOUT_EXT=${LIBPLACEBO_DLL_FILENAME/.dll/}
        LIBPLACEBO_DEF_FILENAME=${LIBPLACEBO_DLL_FILENAME}.def
        LIBPLACEBO_DEF_FILENAME=${LIBPLACEBO_DEF_FILENAME/.dll.def/.def}
        echo ${LIBPLACEBO_DLL_FILENAME_WITHOUT_EXT}
        echo "dumpbin.exe /exports ${LIBPLACEBO_DLL_FILENAME} > ${LIBPLACEBO_DEF_FILENAME}.tmp" > dumpbin.bat
        eval "./dumpbin.bat"
        echo "LIBRARY ${LIBPLACEBO_DLL_FILENAME_WITHOUT_EXT}" > ${LIBPLACEBO_DEF_FILENAME}
        echo "EXPORTS" >> ${LIBPLACEBO_DEF_FILENAME}
        sed -n '/ordinal hint/,/Summary/p' ${LIBPLACEBO_DEF_FILENAME}.tmp | sed '/ordinal hint\|^$\|Summary/d' | awk '{print " "$4}' >> ${LIBPLACEBO_DEF_FILENAME}
        LIBPLACEBO_LIB_FILENAME=$(basename $LIBPLACEBO_DEF_FILENAME .def).lib
        lib.exe -machine:$TARGET_ARCH -def:$LIBPLACEBO_DEF_FILENAME -out:$LIBPLACEBO_LIB_FILENAME
        #cp `find ./.libs/libass-*.dll` .
    fi
fi

cd $BUILD_DIR/$TARGET_ARCH
if should_build ZIMG && [ ! -d "zimg" ]; then
    find ../src/ -type d -name "zimg*" | xargs -i cp -r {} ./zimg
    start_build "zimg"
    cd zimg
    ./autogen.sh
    
    CFLAGS="${BUILD_CCFLAGS}" \
    CXXFLAGS="${BUILD_CCFLAGS}" \
    LDFLAGS="${BUILD_LDFLAGS}" \
        ./configure \
        --prefix=$INSTALL_DIR \
        --disable-shared \
        --enable-static
    make -j$NJOBS && make install
fi

if should_build X264; then
    cd $BUILD_DIR/$TARGET_ARCH
    if [ ! -d "x264" ]; then
        ensure_test_yuv_files
        find ../src/ -type d -name "x264*" | xargs -i cp -r {} ./x264
        start_build "x264"
        cd x264
        X264_ENABLE_LTO=
        if [ $ENABLE_LTO = "TRUE" ]; then
            X264_ENABLE_LTO=--enable-lto
        fi
        patch < $HOME/patches/x264_makefile.diff
        PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig \
        ./configure \
         --prefix=$INSTALL_DIR \
         --enable-strip \
         --disable-ffms \
         --disable-gpac \
         --disable-lavf \
         --enable-static \
         --disable-shared \
         $X264_ENABLE_LTO \
         --bit-depth=all \
         --extra-cflags="${BUILD_CCFLAGS}" \
         --extra-ldflags="${BUILD_LDFLAGS}"
        make fprofiled VIDS="${YUVFILE}" -j$NJOBS && make install
    fi
fi

if should_build X265; then
    cd $BUILD_DIR/$TARGET_ARCH
    if [ ! -d "x265" ]; then
        ensure_test_yuv_files
        find ../src/ -type d -name "x265*" | xargs -i cp -r {} ./x265
        start_build "x265"
        cd x265
        patch -p 1 < $HOME/patches/x265_version.diff
        patch -p 1 < $HOME/patches/x265_zone_param.diff
        patch -p 0 < $HOME/patches/x265_json11.diff
        mkdir build/msys2 && cd build/msys2
        mkdir 8bit
        mkdir 12bit && cd 12bit
        cmake -G "${CMAKE_GENERATOR}" ../../../source \
            -DHIGH_BIT_DEPTH=ON \
            -DEXPORT_C_API=OFF \
            -DENABLE_SHARED=OFF \
            -DENABLE_ALPHA=ON \
            -DENABLE_MULTIVIEW=ON \
            -DENABLE_SCC_EXT=ON \
            -DENABLE_HDR10_PLUS=OFF \
            -DENABLE_CLI=OFF \
            -DMAIN12=ON \
            -DCMAKE_C_FLAGS="${BUILD_CCFLAGS} ${PROFILE_GEN_CC}" \
            -DCMAKE_CXX_FLAGS="${BUILD_CCFLAGS} ${PROFILE_GEN_CC}" \
            -DCMAKE_EXE_LINKER_FLAGS="${BUILD_LDFLAGS}"
        make -j${NJOBS} &
        
        cd ../
        mkdir 10bit && cd 10bit
        cmake -G "${CMAKE_GENERATOR}" ../../../source \
            -DHIGH_BIT_DEPTH=ON \
            -DEXPORT_C_API=OFF \
            -DENABLE_SHARED=OFF \
            -DENABLE_ALPHA=ON \
            -DENABLE_MULTIVIEW=ON \
            -DENABLE_SCC_EXT=ON \
            -DENABLE_HDR10_PLUS=ON \
            -DENABLE_CLI=OFF \
            -DCMAKE_C_FLAGS="${BUILD_CCFLAGS} ${PROFILE_GEN_CC}" \
            -DCMAKE_CXX_FLAGS="${BUILD_CCFLAGS} ${PROFILE_GEN_CC}" \
            -DCMAKE_EXE_LINKER_FLAGS="${BUILD_LDFLAGS} ${PROFILE_GEN_LD}"
        make -j${NJOBS} &

        cd ../8bit
        wait
        cp ../10bit/libx265.a libx265_main10.a
        cp ../12bit/libx265.a libx265_main12.a
        X265_EXTRA_LIB="x265_main10;x265_main12"
        cmake -G "${CMAKE_GENERATOR}" ../../../source \
            -DEXTRA_LIB="${X265_EXTRA_LIB}" \
            -DEXTRA_LINK_FLAGS=-L. \
            -DLINKED_10BIT=ON \
            -DLINKED_12BIT=ON \
            -DENABLE_SHARED=OFF \
            -DENABLE_ALPHA=ON \
            -DENABLE_MULTIVIEW=ON \
            -DENABLE_SCC_EXT=ON \
            -DENABLE_HDR10_PLUS=OFF \
            -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
            -DCMAKE_C_FLAGS="${BUILD_CCFLAGS} ${PROFILE_GEN_CC}" \
            -DCMAKE_CXX_FLAGS="${BUILD_CCFLAGS} ${PROFILE_GEN_CC}" \
            -DCMAKE_EXE_LINKER_FLAGS="${BUILD_LDFLAGS} ${PROFILE_GEN_LD}"
        make -j${NJOBS}

        #profileのための実行はシングルスレッドで行う
        ./x265 --pools none --frame-threads 1 --lookahead-slices 0 --input-res 1280x720 -o /dev/null --input "${YUVFILE}" --preset faster
        ./x265 --pools none --frame-threads 1 --lookahead-slices 0 --input-res 1280x720 -o /dev/null --input "${YUVFILE}" --preset fast
        ./x265 --pools none --frame-threads 1 --lookahead-slices 0 --input-res 1280x720 -o /dev/null --input "${YUVFILE}"
        ./x265 --pools none --frame-threads 1 --lookahead-slices 0 --input-res 1280x720 -o /dev/null --input "${YUVFILE}" --preset slow
        ./x265 --pools none --frame-threads 1 --lookahead-slices 0 --input-res 1280x720 -o /dev/null --input "${YUVFILE}" --preset slower
        ./x265 --pools none --frame-threads 1 --lookahead-slices 0 --input-res 1280x720 -o /dev/null --input "${YUVFILE_10}" --output-depth 10 --preset faster
        ./x265 --pools none --frame-threads 1 --lookahead-slices 0 --input-res 1280x720 -o /dev/null --input "${YUVFILE_10}" --output-depth 10 --preset fast
        ./x265 --pools none --frame-threads 1 --lookahead-slices 0 --input-res 1280x720 -o /dev/null --input "${YUVFILE_10}" --output-depth 10
        ./x265 --pools none --frame-threads 1 --lookahead-slices 0 --input-res 1280x720 -o /dev/null --input "${YUVFILE_10}" --output-depth 10 --preset slow
        ./x265 --pools none --frame-threads 1 --lookahead-slices 0 --input-res 1280x720 -o /dev/null --input "${YUVFILE_10}" --output-depth 10 --preset slower
        ./x265 --pools none --frame-threads 1 --lookahead-slices 0 --input-res 1280x720 -o /dev/null --input "${YUVFILE_10}" --output-depth 12 --preset faster
        ./x265 --pools none --frame-threads 1 --lookahead-slices 0 --input-res 1280x720 -o /dev/null --input "${YUVFILE_10}" --output-depth 12 --preset fast
        ./x265 --pools none --frame-threads 1 --lookahead-slices 0 --input-res 1280x720 -o /dev/null --input "${YUVFILE_10}" --output-depth 12
        ./x265 --pools none --frame-threads 1 --lookahead-slices 0 --input-res 1280x720 -o /dev/null --input "${YUVFILE_10}" --output-depth 12 --preset slow
        ./x265 --pools none --frame-threads 1 --lookahead-slices 0 --input-res 1280x720 -o /dev/null --input "${YUVFILE_10}" --output-depth 12 --preset slower
        
        cd ../12bit
        cmake -G "${CMAKE_GENERATOR}" ../../../source \
            -DHIGH_BIT_DEPTH=ON \
            -DEXPORT_C_API=OFF \
            -DENABLE_SHARED=OFF \
            -DENABLE_ALPHA=ON \
            -DENABLE_MULTIVIEW=ON \
            -DENABLE_SCC_EXT=ON \
            -DENABLE_HDR10_PLUS=OFF \
            -DSTATIC_LINK_CRT=ON \
            -DENABLE_CLI=OFF \
            -DMAIN12=ON \
            -DCMAKE_C_FLAGS="${BUILD_CCFLAGS} ${PROFILE_USE_CC}" \
            -DCMAKE_CXX_FLAGS="${BUILD_CCFLAGS} ${PROFILE_USE_CC}" \
            -DCMAKE_EXE_LINKER_FLAGS="${BUILD_LDFLAGS}"
        make -j${NJOBS} &
        
        cd ../10bit
        cmake -G "${CMAKE_GENERATOR}" ../../../source \
            -DHIGH_BIT_DEPTH=ON \
            -DEXPORT_C_API=OFF \
            -DENABLE_SHARED=OFF \
            -DENABLE_ALPHA=ON \
            -DENABLE_MULTIVIEW=ON \
            -DENABLE_SCC_EXT=ON \
            -DENABLE_HDR10_PLUS=ON \
            -DSTATIC_LINK_CRT=ON \
            -DENABLE_CLI=OFF \
            -DCMAKE_C_FLAGS="${BUILD_CCFLAGS} ${PROFILE_USE_CC}" \
            -DCMAKE_CXX_FLAGS="${BUILD_CCFLAGS} ${PROFILE_USE_CC}" \
            -DCMAKE_EXE_LINKER_FLAGS="${BUILD_LDFLAGS} ${PROFILE_GEN_LD}"
        make -j${NJOBS} &

        cd ../8bit
        wait
        cp ../10bit/libx265.a libx265_main10.a
        cp ../12bit/libx265.a libx265_main12.a
        X265_EXTRA_LIB="x265_main10;x265_main12"
        cmake -G "${CMAKE_GENERATOR}" ../../../source \
            -DEXTRA_LIB="${X265_EXTRA_LIB}" \
            -DEXTRA_LINK_FLAGS=-L. \
            -DLINKED_10BIT=ON \
            -DLINKED_12BIT=ON \
            -DSTATIC_LINK_CRT=ON \
            -DENABLE_SHARED=OFF \
            -DENABLE_ALPHA=ON \
            -DENABLE_MULTIVIEW=ON \
            -DENABLE_SCC_EXT=ON \
            -DENABLE_HDR10_PLUS=OFF \
            -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
            -DCMAKE_C_FLAGS="${BUILD_CCFLAGS} ${PROFILE_USE_CC}" \
            -DCMAKE_CXX_FLAGS="${BUILD_CCFLAGS} ${PROFILE_USE_CC}" \
            -DCMAKE_EXE_LINKER_FLAGS="${BUILD_LDFLAGS} ${PROFILE_USE_LD}"
        make -j${NJOBS}

        mv libx265.a libx265_main.a
        echo -n -e "create libx265.a\naddlib libx265_main.a\naddlib libx265_main10.a\naddlib libx265_main12.a\nsave\nend" | ar -M
        make install
        # static linkがうまくいくように書き換え
        sed -i -e 's/^Libs.private:.*/Libs.private: -lstdc++/g' $INSTALL_DIR/lib/pkgconfig/x265.pc
    fi
fi

if should_build XVIDCORE; then
    cd $BUILD_DIR/$TARGET_ARCH
    if [ ! -d "xvidcore" ]; then
        find ../src/ -type d -name "xvidcore*" | xargs -i cp -r {} ./xvidcore
        start_build "xvidcore"
        cd xvidcore/build/generic
        ./configure --help
        ./bootstrap.sh
        CFLAGS="${BUILD_CCFLAGS} -std=gnu17" \
        CPPFLAGS=${BUILD_CCFLAGS} \
        LDFLAGS=${BUILD_LDFLAGS} \
        ./configure --prefix=$INSTALL_DIR
        make -j${NUMBER_OF_PROCESSORS}
        cp ../../src/xvid.h $INSTALL_DIR/include/
        cp '=build/xvidcore.a' $INSTALL_DIR/lib/libxvidcore.a
    fi
fi

if should_build VVENC; then
    cd $BUILD_DIR/$TARGET_ARCH
    if [ ! -d "vvenc" ]; then
        find ../src/ -type d -name "vvenc*" | xargs -i cp -r {} ./vvenc
        start_build "vvenc"
        cd vvenc
        mkdir build && cd build
        CC=gcc \
        CXX=g++ \
        PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig \
        CFLAGS="${BUILD_CCFLAGS}" \
        CPPFLAGS="${BUILD_CCFLAGS}" \
        LDFLAGS="${BUILD_LDFLAGS}" \
        cmake -G "${CMAKE_GENERATOR}" -B build/release-static -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -DCMAKE_BUILD_TYPE=Release -DVVENC_INSTALL_FULLFEATURE_APP=OFF -DVVENC_ENABLE_THIRDPARTY_JSON=OFF -DVVENC_LIBRARY_ONLY=ON -DVVENC_ENABLE_WERROR=OFF -DBUILD_TESTING=OFF ..
        cmake --build build/release-static -j$NJOBS && cmake --build build/release-static --target install
        VVENC_PC_FILE="$INSTALL_DIR/lib/pkgconfig/libvvenc.pc"
        if [ ! -f "$VVENC_PC_FILE" ] && [ -f "$INSTALL_DIR/lib/pkgconfig/vvenc.pc" ]; then
            VVENC_PC_FILE="$INSTALL_DIR/lib/pkgconfig/vvenc.pc"
        fi
        if [ ! -f "$VVENC_PC_FILE" ]; then
            echo "vvenc pkg-config file not found."
            exit 1
        fi
        # static linkがうまくいくように書き換え
        if [ "$MINGWDIR" = "" ] && [ -f "$LIBSTDCXX_A" ]; then
            sed -i -e "s#^Libs:.*#Libs: -L\${libdir} -lvvenc ${LIBSTDCXX_STATIC_FLAGS}#g" "$VVENC_PC_FILE"
            sed -i -e 's#^Libs.private:.*#Libs.private: -lm -lgcc -lgcc#g' "$VVENC_PC_FILE"
        else
            sed -i -e 's/-lvvenc/-lvvenc -lstdc++/g' "$VVENC_PC_FILE"
        fi
    fi
fi

if should_build SVT_AV1; then
    cd $BUILD_DIR/$TARGET_ARCH
    if [ ! -d "svt-av1" ]; then
        ensure_test_yuv_files
        find ../src/ -type d -name "svt-av1*" | xargs -i cp -r {} ./svt-av1
        start_build "svt-av1"
        cd svt-av1
        mkdir -p build/msys2 && cd build/msys2
        SVTAV1_ENABLE_LTO=OFF
        if [ $ENABLE_LTO = "TRUE" ]; then
            SVTAV1_ENABLE_LTO=ON
        fi
        cmake -G "${CMAKE_GENERATOR}" \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_SHARED_LIBS=OFF \
            -DBUILD_TESTING=OFF \
            -DNATIVE=OFF \
            -DSVT_AV1_LTO=$SVTAV1_ENABLE_LTO \
            -DENABLE_NASM=ON \
            -DENABLE_AVX512=ON \
            -DCMAKE_ASM_NASM_COMPILER=nasm \
            -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
            -DCMAKE_C_FLAGS="${BUILD_CCFLAGS} ${PROFILE_GEN_CC} ${PROFILE_SVTAV1}" \
            -DCMAKE_CXX_FLAGS="${BUILD_CCFLAGS} ${PROFILE_GEN_CC} ${PROFILE_SVTAV1}" \
            -DCMAKE_EXE_LINKER_FLAGS="${BUILD_LDFLAGS} ${PROFILE_GEN_LD} ${PROFILE_SVTAV1}" \
            ../..
        make -j${NUMBER_OF_PROCESSORS}

        SVTAV1_ENC_APP="../../Bin/Release/SvtAv1EncApp"
        if [ -x "${SVTAV1_ENC_APP}.exe" ]; then
            SVTAV1_ENC_APP="${SVTAV1_ENC_APP}.exe"
        fi
        if [ ! -x "${SVTAV1_ENC_APP}" ]; then
            echo "SvtAv1EncApp not found: ${SVTAV1_ENC_APP}"
            exit 1
        fi

        "${SVTAV1_ENC_APP}" -w 1280 -h 720 --crf 30 --scd 1 --fps-num 30 --fps-denom 1 -b /dev/null -i ${YUVFILE}    --preset 4 -n 30 --asm avx512
        "${SVTAV1_ENC_APP}" -w 1280 -h 720 --crf 30 --scd 1 --fps-num 30 --fps-denom 1 -b /dev/null -i ${YUVFILE}    --preset 8 -n 30 --asm avx512
        "${SVTAV1_ENC_APP}" -w 1280 -h 720 --crf 30 --scd 1 --fps-num 30 --fps-denom 1 -b /dev/null -i ${YUVFILE}    --preset 4 -n 30 --asm avx2
        "${SVTAV1_ENC_APP}" -w 1280 -h 720 --crf 30 --scd 1 --fps-num 30 --fps-denom 1 -b /dev/null -i ${YUVFILE}    --preset 8 -n 30 --asm avx2
        "${SVTAV1_ENC_APP}" -w 1280 -h 720 --crf 30 --scd 1 --fps-num 30 --fps-denom 1 -b /dev/null -i ${YUVFILE_10} --preset 4 -n 30 --input-depth 10 --asm avx512
        "${SVTAV1_ENC_APP}" -w 1280 -h 720 --crf 30 --scd 1 --fps-num 30 --fps-denom 1 -b /dev/null -i ${YUVFILE_10} --preset 8 -n 30 --input-depth 10 --asm avx512
        "${SVTAV1_ENC_APP}" -w 1280 -h 720 --crf 30 --scd 1 --fps-num 30 --fps-denom 1 -b /dev/null -i ${YUVFILE_10} --preset 4 -n 30 --input-depth 10 --asm avx2
        "${SVTAV1_ENC_APP}" -w 1280 -h 720 --crf 30 --scd 1 --fps-num 30 --fps-denom 1 -b /dev/null -i ${YUVFILE_10} --preset 8 -n 30 --input-depth 10 --asm avx2

        cmake -G "${CMAKE_GENERATOR}" \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_SHARED_LIBS=OFF \
            -DBUILD_TESTING=OFF \
            -DNATIVE=OFF \
            -DSVT_AV1_LTO=$SVTAV1_ENABLE_LTO \
            -DENABLE_NASM=ON \
            -DENABLE_AVX512=ON \
            -DCMAKE_ASM_NASM_COMPILER=nasm \
            -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
            -DCMAKE_C_FLAGS="${BUILD_CCFLAGS} ${PROFILE_USE_CC} ${PROFILE_SVTAV1}" \
            -DCMAKE_CXX_FLAGS="${BUILD_CCFLAGS} ${PROFILE_USE_CC} ${PROFILE_SVTAV1}" \
            -DCMAKE_EXE_LINKER_FLAGS="${BUILD_LDFLAGS} ${PROFILE_USE_LD} ${PROFILE_SVTAV1}" \
            ../..
        make -j${NUMBER_OF_PROCESSORS} && make install
    fi
fi

# cd $BUILD_DIR/$TARGET_ARCH
# if [ ! -d "gmp" ]; then
    # find ../src/ -type d -name "gmp-*" | xargs -i cp -r {} ./gmp
    # start_build "gmp"
    # cd ./gmp
    # PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig \
    # CFLAGS="${BUILD_CCFLAGS_SMALL}" \
    # CPPFLAGS="${BUILD_CCFLAGS_SMALL}" \
    # LDFLAGS="${BUILD_LDFLAGS}" \
     # ./configure \
     # --prefix=$INSTALL_DIR \
     # --disable-shared \
     # --enable-static
     # make install -j$NJOBS
 # fi

# cd $BUILD_DIR/$TARGET_ARCH
# if [ ! -d "nettle" ]; then
    # find ../src/ -type d -name "nettle-*" | xargs -i cp -r {} ./nettle
    # start_build "nettle"
    # cd ./nettle
    # PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig \
    # CFLAGS="${BUILD_CCFLAGS_SMALL}" \
    # CPPFLAGS="${BUILD_CCFLAGS_SMALL}" \
    # LDFLAGS="${BUILD_LDFLAGS}" \
     # ./configure \
     # --prefix=$INSTALL_DIR \
     # --disable-shared \
     # --enable-static --disable-openssl
    # make install -j$NJOBS
# fi

# cd $BUILD_DIR/$TARGET_ARCH/gnutls
# PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig \
# CFLAGS="${BUILD_CCFLAGS_SMALL}" \
# CPPFLAGS="${BUILD_CCFLAGS_SMALL}" \
# LDFLAGS="${BUILD_LDFLAGS}" \
# ./configure \
# --prefix=$INSTALL_DIR \
# --disable-shared --disable-cxx \
# --disable-openssl-compatibility \
# --disable-doc --disable-gtk-doc-html \
# --with-included-libtasn1 --without-p11-kit
# sed -i.orig -e "/Libs.private:/s/$/ -lcrypt32/" lib/gnutls.pc
# make install -j$NJOBS

if [ $ENABLE_SWSCALE = "TRUE" ]; then
    SWSCALE_ARG="--enable-swscale"
else
    SWSCALE_ARG="--disable-swscale"
fi

if [ $FOR_FFMPEG4 = "TRUE" ]; then
    PKG_CONFIG_FLAGS=""
    FFMPEG5_CUDA_DISABLE_FLAGS=""
else
    PKG_CONFIG_FLAGS="--pkg-config-flags=\"--static\""
    FFMPEG5_CUDA_DISABLE_FLAGS=" --disable-cuda-nvcc --disable-cuda-llvm"
fi

if [ $TARGET_ARCH != "x86" ]; then
    ENCODER_LIBS=""
    if [ "${BUILD_LIB_VVENC}" = "TRUE" ]; then
        ENCODER_LIBS="${ENCODER_LIBS} --enable-libvvenc"
    fi
    if [ "${BUILD_LIB_SVT_AV1}" = "TRUE" ]; then
        ENCODER_LIBS="${ENCODER_LIBS} --enable-libsvtav1"
    fi
else
    ENCODER_LIBS=""
fi

if [ $ENABLE_GPL = "TRUE" ]; then
  GPL_LIBS="--enable-gpl --enable-libx264 --enable-libx265 --enable-libxvid"
else
  GPL_LIBS=""
fi

ARIB_LIBS=""
if [ "${BUILD_LIB_LIBARIBCAPTION}" = "TRUE" ]; then
    ARIB_LIBS="${ARIB_LIBS} --enable-libaribcaption"
fi
if [ "${BUILD_LIB_ARIBB24}" = "TRUE" ]; then
    ARIB_LIBS="${ARIB_LIBS} --enable-libaribb24"
fi

SOXR_LIBS=""
if [ "${BUILD_LIB_SOXR}" = "TRUE" ]; then
    SOXR_LIBS="--enable-libsoxr"
fi

# Linux静的リンク時、libsoxr等が要求するlibmを明示的に末尾へ渡す
FFMPEG_EXTRA_LIBS=""
if [ "$MINGWDIR" = "" ]; then
    FFMPEG_EXTRA_LIBS="--extra-libs=-lm"
fi

TWOLAME_LIBS=""
if [ "${BUILD_LIB_TWOLAME}" = "TRUE" ]; then
    TWOLAME_LIBS="--enable-libtwolame"
fi

# Linuxではlibiconvが不要/存在しないため、過去ビルド由来の-likonv混入を除去
if [ "$MINGWDIR" = "" ]; then
    sed -i 's/ -liconv//g' ${INSTALL_DIR}/lib/pkgconfig/*.pc 2>/dev/null || true
    # 既存成果物を再利用する場合でも、-static-libstdc++ を確実に静的libstdc++指定へ正規化
    normalize_static_libstdcxx_pc_dir "$INSTALL_DIR/lib/pkgconfig"
fi

# FFmpeg configure用pkg-config探索パス
PKG_CONFIG_PATH_FFMPEG=${INSTALL_DIR}/lib/pkgconfig
if [ "$MINGWDIR" = "" ]; then
    for pcdir in "${INSTALL_DIR}"/lib/*-linux-gnu/pkgconfig; do
        if [ -d "$pcdir" ]; then
            PKG_CONFIG_PATH_FFMPEG="${pcdir}:${PKG_CONFIG_PATH_FFMPEG}"
        fi
    done
fi

cd $BUILD_DIR/$TARGET_ARCH/$FFMPEG_DIR_NAME
if [ $FOR_AUDENC = "TRUE" ]; then
start_build "FFmpeg for Audenc"
pwd
PKG_CONFIG_PATH=${PKG_CONFIG_PATH_FFMPEG} \
./configure \
--prefix=${BUILD_DIR}/$FFMPEG_DIR_NAME/tmp/$TARGET_ARCH \
$PKG_CONFIG_FLAGS \
--arch="${FFMPEG_ARCH}" \
--target-os="${FFMPEG_TARGET_OS}" \
--enable-version3 \
--disable-doc \
$SWSCALE_ARG \
$FFMPEG_DISABLE_ASM \
$GPL_LIBS \
--disable-avdevice \
--disable-hwaccels \
--disable-devices \
--disable-debug \
--disable-shared \
--disable-amd3dnow \
--disable-amd3dnowext \
--disable-dxva2 \
--disable-d3d11va \
$FFMPEG5_CUDA_DISABLE_FLAGS \
--disable-xop \
--disable-fma4 \
--disable-network \
--disable-bsfs \
--enable-swresample \
--disable-protocols \
--enable-protocol="file,pipe,fd" \
--disable-decoders \
--enable-decoder="pcm*,adpcm*" \
--disable-demuxers \
--enable-demuxer="wav" \
--disable-encoders \
--enable-encoder="aac,ac3*,alac,adpcm*,eac3,flac,libmp3lame,libopus,libspeex,libtwolame,libmp3lame,libvorbis,mp2*,opus,pcm*,truehd,vorbis,wma*" \
--enable-libvorbis \
--enable-libspeex \
--enable-libmp3lame \
$TWOLAME_LIBS \
$SOXR_LIBS \
--enable-libopus \
--disable-filters \
--enable-filter=$CONFIGURE_AUDFILTER_LIST \
--enable-small \
--disable-mediafoundation \
--pkg-config-flags="--static" \
--extra-cflags="${BUILD_CCFLAGS} -Os -I${INSTALL_DIR}/include ${FFMPEG_SSE}" \
--extra-ldflags="${BUILD_LDFLAGS} -L${INSTALL_DIR}/lib" \
$FFMPEG_EXTRA_LIBS
elif [ $BUILD_EXE = "TRUE" ]; then
start_build "FFmpeg for Executable"
PKG_CONFIG_PATH=${PKG_CONFIG_PATH_FFMPEG} \
./configure \
--prefix=${BUILD_DIR}/$FFMPEG_DIR_NAME/tmp/$TARGET_ARCH \
$PKG_CONFIG_FLAGS \
--arch="${FFMPEG_ARCH}" \
--target-os="${FFMPEG_TARGET_OS}" \
--enable-version3 \
--disable-debug \
--disable-shared \
--disable-doc \
$SWSCALE_ARG \
$FFMPEG_DISABLE_ASM \
$ENCODER_LIBS \
$GPL_LIBS \
--disable-outdevs \
--disable-amd3dnow \
--disable-amd3dnowext \
--disable-xop \
--disable-fma4 \
--disable-w32threads \
$FFMPEG5_CUDA_DISABLE_FLAGS \
--enable-pthreads \
--enable-bsfs \
--enable-filters \
--enable-swresample \
--disable-decoder=vorbis \
--enable-libvorbis \
--enable-libspeex \
--enable-libmp3lame \
$TWOLAME_LIBS \
--enable-fontconfig \
--enable-libfribidi \
--enable-libfreetype \
$SOXR_LIBS \
--enable-libopus \
--enable-libass \
--enable-libdav1d \
--enable-libvpl \
--enable-libvpx \
--enable-libglslang \
--enable-libzimg \
--enable-libplacebo \
--enable-ffnvcodec \
--enable-nvdec \
--enable-cuvid \
--disable-mediafoundation \
--pkg-config-flags="--static" \
$ARIB_LIBS \
--extra-cflags="${BUILD_CCFLAGS} -I${INSTALL_DIR}/include ${FFMPEG_SSE}" \
--extra-ldflags="${BUILD_LDFLAGS} -L${INSTALL_DIR}/lib" \
$FFMPEG_EXTRA_LIBS
else
start_build "FFmpeg for Library"
FFMPEG_INSTALL_DIR=${BUILD_DIR}/$FFMPEG_DIR_NAME/tmp/$TARGET_ARCH
if [ "$MINGWDIR" = "" ]; then
    FFMPEG_INSTALL_DIR=$INSTALL_DIR
fi
PKG_CONFIG_PATH=${PKG_CONFIG_PATH_FFMPEG} \
./configure \
--prefix=${FFMPEG_INSTALL_DIR} \
$PKG_CONFIG_FLAGS \
--arch="${FFMPEG_ARCH}" \
--target-os="${FFMPEG_TARGET_OS}" \
--enable-version3 \
--disable-doc \
$SWSCALE_ARG \
$FFMPEG_DISABLE_ASM \
$GPL_LIBS \
$ENCODER_LIBS \
--disable-outdevs \
--disable-debug \
--enable-static \
--disable-amd3dnow \
--disable-amd3dnowext \
--disable-xop \
--disable-fma4 \
--disable-aesni \
--disable-w32threads \
--disable-dxva2 \
--disable-d3d11va \
$FFMPEG5_CUDA_DISABLE_FLAGS \
--enable-pthreads \
--enable-bsfs \
--enable-swresample \
--disable-shared \
--disable-decoder=vorbis \
--enable-libvorbis \
--enable-libspeex \
--enable-libmp3lame \
$TWOLAME_LIBS \
--enable-fontconfig \
--enable-libfribidi \
--enable-libfreetype \
$SOXR_LIBS \
--enable-libopus \
--enable-libbluray \
--enable-libass \
--enable-libdav1d \
--enable-libvpl \
--enable-libvpx \
--enable-ffnvcodec \
--enable-nvdec \
--enable-cuvid \
--disable-mediafoundation \
--pkg-config-flags="--static" \
$ARIB_LIBS \
--extra-cflags="${BUILD_CCFLAGS} -I${INSTALL_DIR}/include ${FFMPEG_SSE}" \
--extra-ldflags="${BUILD_LDFLAGS} -L${INSTALL_DIR}/lib" \
$FFMPEG_EXTRA_LIBS
fi
make clean && make -j$NJOBS && make install

if [ "$MINGWDIR" != "" ]; then
    mkdir -p $BUILD_DIR/$FFMPEG_DIR_NAME/include
    mkdir -p $BUILD_DIR/$FFMPEG_DIR_NAME/lib/$VC_ARCH
    cp -f -r $BUILD_DIR/$FFMPEG_DIR_NAME/tmp/$TARGET_ARCH/include/* $BUILD_DIR/$FFMPEG_DIR_NAME/include
    if [ -d "$BUILD_DIR/$FFMPEG_DIR_NAME/tmp/$TARGET_ARCH/bin" ]; then
    cp -f -r $BUILD_DIR/$FFMPEG_DIR_NAME/tmp/$TARGET_ARCH/bin/*     $BUILD_DIR/$FFMPEG_DIR_NAME/lib/$VC_ARCH
    fi
    cp -f -r $BUILD_DIR/$FFMPEG_DIR_NAME/tmp/$TARGET_ARCH/lib/*     $BUILD_DIR/$FFMPEG_DIR_NAME/lib/$VC_ARCH
    rm -rf   $BUILD_DIR/$FFMPEG_DIR_NAME/tmp

    if should_build LIBASS_DLL; then
        cp -f -r $BUILD_DIR/$TARGET_ARCH/libass_dll/libass/libass-*.dll $BUILD_DIR/$FFMPEG_DIR_NAME/lib/$VC_ARCH
        cp -f -r $BUILD_DIR/$TARGET_ARCH/libass_dll/libass/libass-*.def $BUILD_DIR/$FFMPEG_DIR_NAME/lib/$VC_ARCH
        cp -f -r $BUILD_DIR/$TARGET_ARCH/libass_dll/libass/libass-*.lib $BUILD_DIR/$FFMPEG_DIR_NAME/lib/$VC_ARCH
        cp -f -r $INSTALL_DIR/include/ass $BUILD_DIR/$FFMPEG_DIR_NAME/include
    fi

    if should_build LIBPLACEBO_DLL; then
        cp -f -r $BUILD_DIR/$TARGET_ARCH/libplacebo_dll/build/src/libplacebo-*.dll $BUILD_DIR/$FFMPEG_DIR_NAME/lib/$VC_ARCH
        cp -f -r $BUILD_DIR/$TARGET_ARCH/libplacebo_dll/build/src/libplacebo-*.def $BUILD_DIR/$FFMPEG_DIR_NAME/lib/$VC_ARCH
        cp -f -r $BUILD_DIR/$TARGET_ARCH/libplacebo_dll/build/src/libplacebo-*.lib $BUILD_DIR/$FFMPEG_DIR_NAME/lib/$VC_ARCH
        cp -f -r $INSTALL_DIR/include/libplacebo $BUILD_DIR/$FFMPEG_DIR_NAME/include
    fi
fi

cd $BUILD_DIR/src
SRC_7Z_FILENAME=ffmpeg_lgpl_src.7z
SRC_GPL_LIBS=
SRC_EXE_LIBS=
SRC_ENCODER_LIBS=
if [ ${ENABLE_GPL} != "FALSE" ]; then
  SRC_7Z_FILENAME=ffmpeg_gpl_src.7z
  SRC_GPL_LIBS="$BUILD_DIR/src/x264* $BUILD_DIR/src/x265* $BUILD_DIR/src/xvidcore*"
fi
if [ $TARGET_ARCH != "x86" ]; then
    SRC_ENCODER_LIBS="$BUILD_DIR/src/svt-av1* $BUILD_DIR/src/vvenc*"
fi
rm -f ${SRC_7Z_FILENAME}
echo "compressing src file..."

collect_existing_paths() {
    local out_var="$1"
    shift
    local files=()
    local pattern
    local matched
    for pattern in "$@"; do
        matched=()
        for f in $pattern; do
            if [ -e "$f" ]; then
                matched+=("$f")
            fi
        done
        if [ ${#matched[@]} -gt 0 ]; then
            files+=("${matched[@]}")
        fi
    done
    eval "$out_var=(\"\${files[@]}\")"
}

collect_existing_paths SRC_ARCHIVE_PATHS \
    "$BUILD_DIR/src/ffmpeg*" "$BUILD_DIR/src/opus*" "$BUILD_DIR/src/libogg*" "$BUILD_DIR/src/libvorbis*" \
    "$BUILD_DIR/src/lame*" "$BUILD_DIR/src/libsndfile*" "$BUILD_DIR/src/twolame*" "$BUILD_DIR/src/soxr*" "$BUILD_DIR/src/speex*" \
    "$BUILD_DIR/src/expat*" "$BUILD_DIR/src/freetype*" "$BUILD_DIR/src/harfbuzz*" "$BUILD_DIR/src/libunibreak*" \
    "$BUILD_DIR/src/libiconv*" "$BUILD_DIR/src/fontconfig*" \
    "$BUILD_DIR/src/libpng*" "$BUILD_DIR/src/libass*" "$BUILD_DIR/src/bzip2*" "$BUILD_DIR/src/libbluray*" \
    "$BUILD_DIR/src/glslang*" "$BUILD_DIR/src/zimg*" \
    "$BUILD_DIR/src/aribb24*" "$BUILD_DIR/src/libaribcaption*" "$BUILD_DIR/src/libxml2*" "$BUILD_DIR/src/dav1d*" \
    "$BUILD_DIR/src/libvpl*" "$BUILD_DIR/src/libvpx*" "$BUILD_DIR/src/nv-codec-headers*" \
    "$BUILD_DIR/src/libxxhash*" "$BUILD_DIR/src/shaderc*" "$BUILD_DIR/src/SPIRV-Cross*" \
    "$BUILD_DIR/src/dovi_tool*" "$BUILD_DIR/src/libjpeg-*" "$BUILD_DIR/src/lcms2*" "$BUILD_DIR/src/libplacebo*" "$BUILD_DIR/src/Vulkan-Loader*" \
    "$SRC_GPL_LIBS" "$SRC_EXE_LIBS" "$SRC_ENCODER_LIBS" \
    "$PATCHES_DIR/*"

if command -v 7z >/dev/null 2>&1; then
    7z a -y -t7z -mx=9 -mmt=off -x\!'*.tar.gz' -x\!'*.tar.bz2' -x\!'*.zip' -x\!'*.tar.xz' -xr\!'.git' ${SRC_7Z_FILENAME} \
    "${SRC_ARCHIVE_PATHS[@]}" \
    > /dev/null
else
    TAR_FILENAME=${SRC_7Z_FILENAME%.7z}.tar.xz
    echo "7z is not installed; creating ${TAR_FILENAME} (.tar.xz) with tar + xz..."
    rm -f "${TAR_FILENAME}"
    tar -cJf "${TAR_FILENAME}" \
        --exclude='*.tar.gz' --exclude='*.tar.bz2' --exclude='*.zip' --exclude='*.tar.xz' \
        --exclude='.git' --exclude='.git/*' \
        "${SRC_ARCHIVE_PATHS[@]}" \
        > /dev/null || { echo "tar failed"; exit 1; }
fi
