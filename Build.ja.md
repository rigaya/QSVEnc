
# QSVEncのビルド方法

- [Windows](./Build.ja.md#windows)
- Linux
  - [Linux (Ubuntu 20.04)](./Build.ja.md#linux-ubuntu-2004)
  - [Linux (Ubuntu 18.04)](./Build.ja.md#linux-ubuntu-1804)
  - [Linux (Fedora 32)](./Build.ja.md#linux-fedora-32)
  - [Intel Drivers for Linux](/Build.ja.md#Intel-Drivers-for-Linux)

## Windows 

### 0. 準備
ビルドには、下記のものが必要です。

- Visual Studio 2019
- [Avisynth](https://github.com/AviSynth/AviSynthPlus) SDK
- [VapourSynth](http://www.vapoursynth.com/) SDK
- Intel OpenCL SDK
- Intel Metric Framework SDK (Intel Platform Analysis Libraryに同梱)

Avisynth+とVapourSynthは、SDKがインストールされるよう設定してインストールします。

Avisynth+ SDKの"avisynth_c.h"とVapourSynth SDKの"VapourSynth.h", "VSScript.h"がVisual Studioのincludeパスに含まれるよう設定します。

includeパスは環境変数 "AVISYNTH_SDK" / "VAPOURSYNTH_SDK" で渡すことができます。

Avisynth+ / VapourSynthインストーラのデフォルトの場所にインストールした場合、下記のように設定することになります。
```Batchfile
setx AVISYNTH_SDK "C:\Program Files (x86)\AviSynth+\FilterSDK"
setx VAPOURSYNTH_SDK "C:\Program Files (x86)\VapourSynth\sdk"
```

Intel OpenCL SDKの環境変数、"INTELOCLSDKROOT"はインストーラにより自動的に設定されます。

Intel Metric Framework SDKの環境変数は、"INTEL_METRIC_FRAMEWORK_SDK"です。
このライブラリについては、cmakeを用いてVisual Studio 2015用にビルドすることが必要です。
それなりに面倒なうえ、このライブラリはGPU/MFXの使用率取得のみに使用されエンコードには関係ないので、
無効化して使わずにおくのもありです。
その場合は、QSVPipeline/rgy_version.hのマクロ "ENABLE_METRIC_FRAMEWORK" を 0 に変更してください。

さらにビルドに必要な[Caption2Ass_PCR](https://github.com/maki-rxrz/Caption2Ass_PCR)をcloneし、環境変数 "CAPTION2ASS_SRC" を設定します。

```Batchfile
git clone https://github.com/maki-rxrz/Caption2Ass_PCR <path-to-clone>
setx CAPTION2ASS_SRC <path-to-clone>/src
```

### 1. ソースのダウンロード

```Batchfile
git clone https://github.com/rigaya/QSVEnc --recursive
```

### 2. QSVEnc.auo / QSVEncC のビルド

QSVEnc.slnを開きます。

ビルドしたいものに合わせて、構成を選択してください。

|              |Debug用構成|Release用構成|
|:---------------------|:------|:--------|
|QSVEnc.auo (win32のみ) | Debug | Release |
|QSVEncC(64).exe | DebugStatic | RelStatic |


## Linux (Ubuntu 20.04)

### 0. ビルドに必要なもの

- C++17 Compiler
- Intel Driver
- git
- libraries
  - libva, libdrm, libmfx 
  - ffmpeg 4.x libs (libavcodec58, libavformat58, libavfilter7, libavutil56, libswresample3)
  - libass9
  - [Optional] VapourSynth

### 1. コンパイラ等のインストール

```Shell
sudo apt install build-essential libtool git
```

### 2. Intel ドライバのインストール
OpenCL関連は[こちらのリンク](https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-focal.html)に従ってインストールする。

```Shell
sudo apt-get install -y gpg-agent wget
wget -qO - https://repositories.intel.com/graphics/intel-graphics.key |
  sudo apt-key add -
sudo apt-add-repository \
  'deb [arch=amd64] https://repositories.intel.com/graphics/ubuntu focal main'
sudo apt-get update
sudo apt install intel-media-va-driver-non-free \
  intel-opencl-icd \
  intel-level-zero-gpu level-zero
sudo apt install opencl-headers
```

### 3. ビルドに必要なライブラリのインストール

```Shell
sudo apt install \
  libmfx1 \
  libmfx-dev \
  libmfx-tools \
  libva-drm2 \
  libva-x11-2 \
  libva-glx2 \
  libx11-dev \
  libigfxcmrt7 \
  libva-dev \
  libdrm-dev

sudo apt install ffmpeg \
  libavcodec-extra libavcodec-dev libavutil-dev libavformat-dev libswresample-dev libavfilter-dev \
  libass9 libass-dev
```

### 4. [オプション] VapourSynthのビルド
VapourSynthのインストールは必須ではありませんが、インストールしておくとvpyを読み込めるようになります。

必要のない場合は 5. QSVEncCのビルド に進んでください。

<details><summary>VapourSynthのビルドの詳細はこちら</summary>

#### 4.1 ビルドに必要なツールのインストール
```Shell
sudo apt install python3-pip autoconf automake libtool meson
```

#### 4.2 zimgのインストール
```Shell
git clone https://github.com/sekrit-twc/zimg.git
cd zimg
./autogen.sh
./configure
sudo make install -j16
cd ..
```

#### 4.3 cythonのインストール
```Shell
sudo pip3 install Cython
```

#### 4.4 VapourSynthのビルド
```Shell
git clone https://github.com/vapoursynth/vapoursynth.git
cd vapoursynth
./autogen.sh
./configure
make -j16
sudo make install

# vapoursynthが自動的にロードされるようにする
# "python3.x" は環境に応じて変えてください。これを書いた時点ではpython3.7でした
sudo ln -s /usr/local/lib/python3.x/site-packages/vapoursynth.so /usr/lib/python3.x/lib-dynload/vapoursynth.so
sudo ldconfig
```

#### 4.5 VapourSynthの動作確認
エラーが出ずにバージョンが表示されればOK。
```Shell
vspipe --version
```

#### 4.6 [おまけ] vslsmashsourceのビルド
```Shell
# lsmashのビルド
git clone https://github.com/l-smash/l-smash.git
cd l-smash
./configure --enable-shared
sudo make install -j16
cd ..
 
# vslsmashsourceのビルド
git clone https://github.com/HolyWu/L-SMASH-Works.git
# ffmpegのバージョンが合わないので、下記バージョンを取得する
cd L-SMASH-Works
git checkout -b 20200531 refs/tags/20200531
cd VapourSynth
meson build
cd build
sudo ninja install
cd ../../../
```

</details>

### 5. QSVとOpenCLの使用のため、ユーザーを下記グループに追加
```Shell
# QSV
sudo gpasswd -a ${USER} video
# OpenCL
sudo gpasswd -a ${USER} render
```

### 5. QSVEncCのビルド
```Shell
git clone https://github.com/rigaya/QSVEnc --recursive
cd QSVEnc
./configure
make -j8
```
動作するか確認します。
```Shell
./qsvencc --check-hw
```

うまく動作するようなら下記のように表示されます。
```
Success: QuickSyncVideo (hw encoding) available
```


## Linux (Ubuntu 18.04)

Ubuntu 18.04では、自分でlibmfx, libva, media-driverをビルド・インストールする必要があります。

### 0. ビルドに必要なもの

- C++17 Compiler
- Intel Driver
- git
- libraries
  - libva, libdrm, libmfx 
  - ffmpeg 4.x libs (libavcodec58, libavformat58, libavfilter7, libavutil56, libswresample3)
  - libass9
  - [Optional] VapourSynth

### 1. コンパイラ等のインストール

```Shell
sudo apt install build-essential meson automake libtool cmake pkg-config git
```

### 2. libvaのインストール

#### 2.1 libvaの依存するパッケージのインストール

```Shell
sudo apt-get install  libdrm-dev libx11-dev libxext-dev libxfixes-dev
```

#### 2.2 libvaのビルド

```Shell
git clone https://github.com/intel/libva.git
cd libva
./autogen.sh
./configure
```

下記のようなメッセージが出るはずです。
```
---------------------------------------------------------------
libva - 2.9.0 (VA-API 1.9.0)

Installation prefix .............. : /usr/local
Default driver path .............. : ${exec_prefix}/lib/dri
Extra window systems ............. : drm x11
Build documentation .............. : no
Build with messaging ............. : yes
---------------------------------------------------------------
```

ビルドし、インストールします。
```Shell
make -j8 && sudo make install
cd ..
```

### 3. Install libmfx
```Shell
git clone https://github.com/Intel-Media-SDK/MediaSDK msdk
cd msdk
mkdir build && cd build
cmake ..
make -j8 && sudo make install
cd ..
```

<details><summary>下記のファイル群がインストールされます。</summary>

```
Install the project...
-- Install configuration: "release"
-- Installing: /opt/intel/mediasdk/share/mfx/plugins.cfg
-- Installing: /opt/intel/mediasdk/lib/libmfx.so.1.34
-- Installing: /opt/intel/mediasdk/lib/libmfx.so.1
-- Installing: /opt/intel/mediasdk/lib/libmfx.so
-- Installing: /opt/intel/mediasdk/lib/pkgconfig/libmfx.pc
-- Installing: /opt/intel/mediasdk/include/mfx
-- Installing: /opt/intel/mediasdk/include/mfx/mfxpcp.h
-- Installing: /opt/intel/mediasdk/include/mfx/mfxplugin.h
-- Installing: /opt/intel/mediasdk/include/mfx/mfxpak.h
-- Installing: /opt/intel/mediasdk/include/mfx/mfxvstructures.h
-- Installing: /opt/intel/mediasdk/include/mfx/mfxsession.h
-- Installing: /opt/intel/mediasdk/include/mfx/mfxadapter.h
-- Installing: /opt/intel/mediasdk/include/mfx/mfxvideo++.h
-- Installing: /opt/intel/mediasdk/include/mfx/mfxmvc.h
-- Installing: /opt/intel/mediasdk/include/mfx/mfxscd.h
-- Installing: /opt/intel/mediasdk/include/mfx/mfxaudio.h
-- Installing: /opt/intel/mediasdk/include/mfx/mfxsc.h
-- Installing: /opt/intel/mediasdk/include/mfx/mfxaudio++.h
-- Installing: /opt/intel/mediasdk/include/mfx/mfxvp8.h
-- Installing: /opt/intel/mediasdk/include/mfx/mfxbrc.h
-- Installing: /opt/intel/mediasdk/include/mfx/mfxplugin++.h
-- Installing: /opt/intel/mediasdk/include/mfx/mfxstructures.h
-- Installing: /opt/intel/mediasdk/include/mfx/mfxvp9.h
-- Installing: /opt/intel/mediasdk/include/mfx/mfxjpeg.h
-- Installing: /opt/intel/mediasdk/include/mfx/mfxfei.h
-- Installing: /opt/intel/mediasdk/include/mfx/mfxdispatcherprefixedfunctions.h
-- Installing: /opt/intel/mediasdk/include/mfx/mfxastructures.h
-- Installing: /opt/intel/mediasdk/include/mfx/mfxdefs.h
-- Installing: /opt/intel/mediasdk/include/mfx/mfxcommon.h
-- Installing: /opt/intel/mediasdk/include/mfx/mfxvideo.h
-- Installing: /opt/intel/mediasdk/include/mfx/mfxfeihevc.h
-- Installing: /opt/intel/mediasdk/include/mfx/mfxla.h
-- Installing: /opt/intel/mediasdk/include/mfx/mfxenc.h
-- Installing: /opt/intel/mediasdk/include/mfx/mfxcamera.h
-- Installing: /opt/intel/mediasdk/lib/pkgconfig/mfx.pc
-- Installing: /opt/intel/mediasdk/share/mfx/samples/sample_decode
-- Installing: /opt/intel/mediasdk/share/mfx/samples/sample_encode
-- Installing: /opt/intel/mediasdk/share/mfx/samples/sample_fei
-- Installing: /opt/intel/mediasdk/share/mfx/samples/sample_hevc_fei
-- Installing: /opt/intel/mediasdk/share/mfx/samples/sample_hevc_fei_abr
-- Installing: /opt/intel/mediasdk/share/mfx/samples/sample_vpp
-- Installing: /opt/intel/mediasdk/share/mfx/samples/sample_multi_transcode
-- Installing: /opt/intel/mediasdk/share/mfx/samples/libsample_rotate_plugin.so
-- Installing: /opt/intel/mediasdk/share/mfx/samples/libvpp_plugin.a
-- Installing: /opt/intel/mediasdk/share/mfx/samples/libcttmetrics.so
-- Installing: /opt/intel/mediasdk/share/mfx/samples/metrics_monitor
-- Installing: /opt/intel/mediasdk/lib/libmfxhw64.so.1.34
-- Installing: /opt/intel/mediasdk/lib/libmfxhw64.so.1
-- Installing: /opt/intel/mediasdk/lib/libmfxhw64.so
-- Installing: /opt/intel/mediasdk/lib/pkgconfig/libmfxhw64.pc
-- Installing: /opt/intel/mediasdk/lib/mfx/libmfx_hevce_hw64.so
-- Installing: /opt/intel/mediasdk/lib/mfx/libmfx_hevc_fei_hw64.so
-- Installing: /opt/intel/mediasdk/lib/mfx/libmfx_vp9e_hw64.so
-- Installing: /opt/intel/mediasdk/lib/mfx/libmfx_h264la_hw64.so
-- Installing: /opt/intel/mediasdk/lib/mfx/libmfx_hevcd_hw64.so
-- Up-to-date: /opt/intel/mediasdk/lib/mfx/libmfx_hevcd_hw64.so
-- Installing: /opt/intel/mediasdk/lib/mfx/libmfx_vp8d_hw64.so
-- Installing: /opt/intel/mediasdk/lib/mfx/libmfx_vp9d_hw64.so
```

</details>

### 4. media driverのインストール

#### 4.1 gmmlibのビルド
```Shell
git clone https://github.com/intel/gmmlib.git
cd gmmlib
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
sudo make install
cd ../..
```

<details><summary>下記のファイル群がインストールされます。</summary>

```
-- Install configuration: ""
-- Installing: /usr/local/lib/dri/iHD_drv_video.so
-- Installing: /usr/local/lib/libigfxcmrt.so.7.2.0
-- Installing: /usr/local/lib/libigfxcmrt.so.7
-- Installing: /usr/local/lib/libigfxcmrt.so
-- Installing: /usr/local/include/igfxcmrt/cm_rt.h
-- Installing: /usr/local/include/igfxcmrt/cm_rt_g8.h
-- Installing: /usr/local/include/igfxcmrt/cm_rt_g9.h
-- Installing: /usr/local/include/igfxcmrt/cm_rt_g10.h
-- Installing: /usr/local/include/igfxcmrt/cm_rt_g11.h
-- Installing: /usr/local/include/igfxcmrt/cm_rt_g12_tgl.h
-- Installing: /usr/local/include/igfxcmrt/cm_hw_vebox_cmd_g10.h
-- Installing: /usr/local/include/igfxcmrt/cm_rt_def_os.h
-- Installing: /usr/local/include/igfxcmrt/cm_rt_api_os.h
-- Installing: /usr/local/include/igfxcmrt/cm_rt_extension.h
-- Installing: /usr/local/lib/pkgconfig/igfxcmrt.pc
```

</details>

#### 4.2 media driverのビルド
```
sudo apt install libdrm-dev xorg xorg-dev openbox libx11-dev libgl1-mesa-glx libgl1-mesa-dev
git clone https://github.com/intel/media-driver.git
mkdir build_media && cd build_media
cmake ../media-driver
make -j8 && sudo make install
cd ..
```

### 5. ffmpeg 4.xのライブラリのインストール
```Shell
sudo add-apt-repository ppa:jonathonf/ffmpeg-4
sudo apt update
sudo apt install ffmpeg \
  libavcodec-extra58 libavcodec-dev libavutil56 libavutil-dev libavformat58 libavformat-dev \
  libswresample3 libswresample-dev libavfilter-extra7 libavfilter-dev libass9 libass-dev
```

### 6. Intel OpenCLランタイムのインストール
OpenCL関連は[こちらのリンク](https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-bionic.html)に従ってインストールする。
```Shell
sudo apt-get install -y gpg-agent wget
wget -qO - https://repositories.intel.com/graphics/intel-graphics.key |
  sudo apt-key add -
sudo apt-add-repository \
  'deb [arch=amd64] https://repositories.intel.com/graphics/ubuntu bionic main'
sudo apt-get update
sudo apt-get install \
  intel-opencl \
  intel-level-zero-gpu level-zero
sudo apt-get install opencl-headers
```

### 7. [オプション] VapourSynthのビルド
VapourSynthのインストールは必須ではありませんが、インストールしておくとvpyを読み込めるようになります。

必要のない場合は 5. QSVEncCのビルド に進んでください。

<details><summary>VapourSynthのビルドの詳細はこちら</summary>

#### 7.1 ビルドに必要なツールのインストール
```Shell
sudo apt install python3-pip autoconf automake libtool meson
```

#### 7.2 zimgのインストール
```Shell
git clone https://github.com/sekrit-twc/zimg.git
cd zimg
./autogen.sh
./configure
sudo make install -j16
cd ..
```

#### 7.3 cythonのインストール
```Shell
sudo pip3 install Cython
```

#### 7.4 VapourSynthのビルド
```Shell
git clone https://github.com/vapoursynth/vapoursynth.git
cd vapoursynth
./autogen.sh
./configure
make -j16
sudo make install

# vapoursynthが自動的にロードされるようにする
# "python3.x" は環境に応じて変えてください。これを書いた時点ではpython3.7でした
sudo ln -s /usr/local/lib/python3.x/site-packages/vapoursynth.so /usr/lib/python3.x/lib-dynload/vapoursynth.so
sudo ldconfig
```

#### 7.5 VapourSynthの動作確認
エラーが出ずにバージョンが表示されればOK。
```Shell
vspipe --version
```

#### 7.6 [おまけ] vslsmashsourceのビルド
```Shell
# lsmashのビルド
git clone https://github.com/l-smash/l-smash.git
cd l-smash
./configure --enable-shared
sudo make install -j16
cd ..
 
# vslsmashsourceのビルド
git clone https://github.com/HolyWu/L-SMASH-Works.git
# ffmpegのバージョンが合わないので、下記バージョンを取得する
cd L-SMASH-Works
git checkout -b 20200531 refs/tags/20200531
cd VapourSynth
meson build
cd build
sudo ninja install
cd ../../../
```

</details>

### 8. QSVとOpenCLの使用のため、ユーザーを下記グループに追加
```Shell
# QSV
sudo gpasswd -a ${USER} video
# OpenCL
sudo gpasswd -a ${USER} render
```

### 9. QSVEncCのビルド
```Shell
git clone https://github.com/rigaya/QSVEnc --recursive
cd QSVEnc
./configure --extra-cxxflags="-I/opt/intel/mediasdk/include" --extra-ldflags="-L/opt/intel/mediasdk/lib"
make -j8
```
動作するか確認します。
```Shell
LD_LIBRARY_PATH=/opt/intel/mediasdk/lib ./qsvencc --check-hw
```

うまく動作するようなら下記のように表示されます。
```
Success: QuickSyncVideo (hw encoding) available
```


## Linux (Fedora 32)

### 0. ビルドに必要なもの

- C++17 Compiler
- Intel Driver
- git
- libraries
  - libva, libdrm, libmfx 
  - ffmpeg 4.x libs (libavcodec58, libavformat58, libavfilter7, libavutil56, libswresample3)
  - libass9
  - [Optional] VapourSynth

### 1. コンパイラ等のインストール

```Shell
sudo dnf install @development-tools
```

### 2. ビルドに必要なライブラリのインストール

```Shell
sudo dnf install https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm
sudo dnf install https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm

sudo dnf install libva-devel libva-X11-devel libdrm-devel intel-mediasdk intel-mediasdk-devel
sudo dnf install ffmpeg ffmpeg-devel
```

### 3. Intel Media ドライバとOpenCLランタイムのインストール

```Shell
#Media
sudo dnf install intel-media-driver
#OpenCL
sudo dnf install -y 'dnf-command(config-manager)'
sudo dnf config-manager \
  --add-repo \
  https://repositories.intel.com/graphics/rhel/8.3/intel-graphics.repo
sudo dnf update --refresh
sudo dnf install \
  intel-opencl \
  intel-media intel-mediasdk \
  level-zero intel-level-zero-gpu
sudo dnf install opencl-headers
```


### 4. [オプション] VapourSynthのビルド
VapourSynthのインストールは必須ではありませんが、インストールしておくとvpyを読み込めるようになります。

必要のない場合は 5. QSVEncCのビルド に進んでください。

<details><summary>VapourSynthのビルドの詳細はこちら</summary>

#### 4.1 ビルドに必要なツールのインストール
```Shell
sudo apt install python3-pip autoconf automake libtool meson
```

#### 4.2 zimgのインストール
```Shell
git clone https://github.com/sekrit-twc/zimg.git
cd zimg
./autogen.sh
./configure
sudo make install -j16
cd ..
```

#### 4.3 cythonのインストール
```Shell
sudo pip3 install Cython
```

#### 4.4 VapourSynthのビルド
```Shell
git clone https://github.com/vapoursynth/vapoursynth.git
cd vapoursynth
./autogen.sh
./configure
make -j16
sudo make install

# vapoursynthが自動的にロードされるようにする
# "python3.x" は環境に応じて変えてください。これを書いた時点ではpython3.7でした
sudo ln -s /usr/local/lib/python3.x/site-packages/vapoursynth.so /usr/lib/python3.x/lib-dynload/vapoursynth.so
sudo ldconfig
```

#### 4.5 VapourSynthの動作確認
エラーが出ずにバージョンが表示されればOK。
```Shell
vspipe --version
```

#### 4.6 [おまけ] vslsmashsourceのビルド
```Shell
# lsmashのビルド
git clone https://github.com/l-smash/l-smash.git
cd l-smash
./configure --enable-shared
sudo make install -j16
cd ..
 
# vslsmashsourceのビルド
git clone https://github.com/HolyWu/L-SMASH-Works.git
# ffmpegのバージョンが合わないので、下記バージョンを取得する
cd L-SMASH-Works
git checkout -b 20200531 refs/tags/20200531
cd VapourSynth
meson build
cd build
sudo ninja install
cd ../../../
```

</details>

### 5. QSVとOpenCLの使用のため、ユーザーを下記グループに追加
```Shell
# QSV
sudo gpasswd -a ${USER} video
# OpenCL
sudo gpasswd -a ${USER} render
```

### 6. QSVEncCのビルド
```Shell
git clone https://github.com/rigaya/QSVEnc --recursive
cd QSVEnc
./configure
make -j8
```
動作するか確認します。
```Shell
./qsvencc --check-hw
```

うまく動作するようなら下記のように表示されます。
```
Success: QuickSyncVideo (hw encoding) available
```

## Intel Drivers for Linux
各Linux distributionのIntelドライバのパッケージについては、Intel Media SDK Wikiの[こちら](https://github.com/Intel-Media-SDK/MediaSDK/wiki/Media-SDK-in-Linux-Distributions)を参照してください。

またドライバの詳細については、Ubuntuの例になりますが、[こちら]((https://github.com/Intel-Media-SDK/MediaSDK/wiki/Intel-media-stack-on-Ubuntu))をご覧ください。