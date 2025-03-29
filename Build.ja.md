
# QSVEncのビルド方法

- [Windows](./Build.ja.md#windows)
- Linux
  - [Linux (Ubuntu 20.04 - 24.04)](./Build.ja.md#linux-ubuntu-2004---2404)
  - [Linux (Ubuntu 18.04)](./Build.ja.md#linux-ubuntu-1804)
  - [Linux (Fedora 32)](./Build.ja.md#linux-fedora-32)
  - [Intel Drivers for Linux](/Build.ja.md#Intel-Drivers-for-Linux)

## Windows 

### 0. 準備
ビルドには、下記のものが必要です。

- Visual Studio 2022
- [Avisynth](https://github.com/AviSynth/AviSynthPlus) SDK
- [VapourSynth](http://www.vapoursynth.com/) SDK

Avisynth+とVapourSynthは、SDKがインストールされるよう設定してインストールします。

Avisynth+ SDKの"avisynth_c.h"とVapourSynth SDKの"VapourSynth.h", "VSScript.h"がVisual Studioのincludeパスに含まれるよう設定します。

includeパスは環境変数 "AVISYNTH_SDK" / "VAPOURSYNTH_SDK" で渡すことができます。

Avisynth+ / VapourSynthインストーラのデフォルトの場所にインストールした場合、下記のように設定することになります。
```Batchfile
setx AVISYNTH_SDK "C:\Program Files (x86)\AviSynth+\FilterSDK"
setx VAPOURSYNTH_SDK "C:\Program Files (x86)\VapourSynth\sdk"
```

ビルドに必要な[OpenCLのヘッダ](https://github.com/KhronosGroup/OpenCL-Headers.git)をcloneし、環境変数 "OPENCL_HEADERS" を設定します。

```Batchfile
git clone https://github.com/KhronosGroup/OpenCL-Headers.git <path-to-clone>
setx OPENCL_HEADERS <path-to-clone>
```

### 1. ソースのダウンロード

```Batchfile
git clone https://github.com/rigaya/QSVEnc --recursive
cd QSVEnc
curl -s -o ffmpeg_lgpl.7z -L https://github.com/rigaya/ffmpeg_dlls_for_hwenc/releases/download/20241102/ffmpeg_dlls_for_hwenc_20241102.7z
7z x -offmpeg_lgpl -y ffmpeg_lgpl.7z
```

### 2. QSVEnc.auo / QSVEncC のビルド

QSVEnc.slnを開きます。

ビルドしたいものに合わせて、構成を選択してください。

|              |Debug用構成|Release用構成|
|:---------------------|:------|:--------|
|QSVEnc.auo (win32のみ) | Debug | Release |
|QSVEncC(64).exe | DebugStatic | RelStatic |



## Linux (Ubuntu 20.04 - 24.04)

### 0. ビルドに必要なもの

- C++17 Compiler
- Intel Driver
- git
- cmake
- rust + cargo-c (libdoviビルド用)
- libraries
  - libva, libdrm 
  - ffmpeg 4.x - 7.x libs (libavcodec*, libavformat*, libavfilter*, libavutil*, libswresample*, libavdevice*)
  - libass9
  - [Optional] VapourSynth

### 1. コンパイラ等のインストール

- ビルドツールのインストール

  ```Shell
  sudo apt install build-essential libtool git cmake
  ```

- rust + cargo-cのインストール (libdovi, libhdr10plusビルド用)

  ```Shell
  sudo apt install libssl-dev curl pkgconf
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
    && . ~/.cargo/env \
    && cargo install cargo-c
  ```

### 2. Intel ドライバのインストール

### 1. Intel Media ドライバのインストール  
[こちらのリンク](https://dgpu-docs.intel.com/driver/client/overview.html)に沿って、ドライバをインストールします。

まず、必要なツールを導入します。

```Shell
sudo apt-get install -y gpg-agent wget
```

次に、Intelのリポジトリを追加します。

```Shell
# Ubuntu 24.04
# このステップは不要です。

# Ubuntu 22.04
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy unified" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list

# Ubuntu 20.04
wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg
echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics-keyring.gpg] https://repositories.intel.com/gpu/ubuntu focal client' | \
  sudo tee /etc/apt/sources.list.d/intel-graphics.list
```

最後にドライバのインストールを行います。

```
sudo apt update
sudo apt install intel-media-va-driver-non-free intel-opencl-icd
```

### 3. ビルドに必要なライブラリのインストール

```Shell
sudo apt install \
  libva-drm2 \
  libva-x11-2 \
  libva-glx2 \
  libx11-dev \
  libigfxcmrt7 \
  libva-dev \
  libdrm-dev \
  opencl-headers

sudo apt install ffmpeg \
  libavcodec-extra libavcodec-dev libavutil-dev libavformat-dev libswresample-dev libavfilter-dev libavdevice-dev \
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
git clone https://github.com/sekrit-twc/zimg.git --recursive
cd zimg
./autogen.sh
./configure
make && sudo make install
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
make && sudo make install

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
make && sudo make install
cd ..
 
# vslsmashsourceのビルド
git clone https://github.com/HolyWu/L-SMASH-Works.git
# ffmpegのバージョンが合わないので、下記バージョンを取得する
cd L-SMASH-Works
git checkout -b 20200531 refs/tags/20200531
cd VapourSynth
meson build
cd build
ninja && sudo ninja install
cd ../../../
```
</details>

### 5. [オプション] AvisynthPlusのビルド

AvisynthPlusのインストールは必須ではありませんが、インストールしておくとavsを読み込めるようになります。

必要のない場合は 7. NVEncCのビルド に進んでください。

<details><summary>AvisynthPlusのビルドの詳細はこちら</summary>
#### 5.1 ビルドに必要なツールのインストール
```Shell
sudo apt install cmake
```

#### 5.2 AvisynthPlusのインストール
```Shell
git clone https://github.com/AviSynth/AviSynthPlus.git
cd AviSynthPlus
mkdir avisynth-build && cd avisynth-build 
cmake ../
make && sudo make install
cd ../..
```

#### 5.3 [おまけ] lsmashsourceのビルド
```Shell
# lsmashのビルド
git clone https://github.com/l-smash/l-smash.git
cd l-smash
./configure --enable-shared
make && sudo make install
cd ..

# lsmashsourceのビルド
git clone https://github.com/HolyWu/L-SMASH-Works.git
cd L-SMASH-Works
# libavcodec の要求バージョンをクリアするためバージョンを下げる
git checkout -b 20200531 refs/tags/20200531
cd AviSynth
meson build
cd build
ninja && sudo ninja install
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
make
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

Ubuntu 18.04では、自分でlibva, media-driverをビルド・インストールする必要があります。

### 0. ビルドに必要なもの

- C++17 Compiler
- Intel Driver
- git
- cmake
- rust + cargo-c
- libraries
  - libva, libdrm 
  - ffmpeg 4.x libs (libavcodec58, libavformat58, libavfilter7, libavutil56, libswresample3, libavdevice58)
  - libass9
  - [Optional] VapourSynth

### 1. コンパイラ等のインストール

- ビルドツールのインストール

  ```Shell
  sudo apt install build-essential meson automake libtool cmake pkg-config git cmake
  ```

- rust + cargo-cのインストール

  ```Shell
  sudo apt install libssl-dev curl
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
    && . ~/.cargo/env \
    && cargo install cargo-c
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
make && sudo make install
cd ..
```

### 4. media driverのインストール

#### 4.1 gmmlibのビルド
```Shell
git clone https://github.com/intel/gmmlib.git
cd gmmlib
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make && sudo make install
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
make && sudo make install
cd ..
```

### 5. ffmpeg 4.xのライブラリのインストール
```Shell
sudo add-apt-repository ppa:jonathonf/ffmpeg-4
sudo apt update
sudo apt install ffmpeg \
  libavcodec-extra58 libavcodec-dev libavutil56 libavutil-dev libavformat58 libavformat-dev libavdevice58 libavdevice-dev \
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
git clone https://github.com/sekrit-twc/zimg.git --recursive
cd zimg
./autogen.sh
./configure
make && sudo make install
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
make && sudo make install

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
make && sudo make install
cd ..
 
# vslsmashsourceのビルド
git clone https://github.com/HolyWu/L-SMASH-Works.git
# ffmpegのバージョンが合わないので、下記バージョンを取得する
cd L-SMASH-Works
git checkout -b 20200531 refs/tags/20200531
cd VapourSynth
meson build
cd build
ninja && sudo ninja install
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
make
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
- cmake
- rust + cargo-c
- libraries
  - libva, libdrm 
  - ffmpeg 4.x libs (libavcodec58, libavformat58, libavfilter7, libavutil56, libswresample3, libavdevice58)
  - libass9
  - [Optional] VapourSynth

### 1. コンパイラ等のインストール

- コンパイラ等のインストール

  ```Shell
  sudo dnf install @development-tools cmake
  ```

- rust + cargo-cのインストール

  ```Shell
  sudo apt install libssl-dev curl
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
    && . ~/.cargo/env \
    && cargo install cargo-c
  ```

### 2. ビルドに必要なライブラリのインストール

```Shell
sudo dnf install https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm
sudo dnf install https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm

sudo dnf install libva-devel libX11-devel libdrm-devel
sudo dnf install ffmpeg ffmpeg-devel
```

### 3. Intel Media ドライバとOpenCLランタイムのインストール

```Shell
#Media
sudo dnf install intel-media-driver
#OpenCL
sudo dnf install -y 'dnf-command(config-manager)'
sudo dnf config-manager --add-repo https://repositories.intel.com/graphics/rhel/8.3/intel-graphics.repo
sudo dnf update --refresh
sudo dnf install intel-opencl intel-media intel-mediasdk level-zero intel-level-zero-gpu
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
git clone https://github.com/sekrit-twc/zimg.git --recursive
cd zimg
./autogen.sh
./configure
make && sudo make install
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
make && sudo make install

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
make && sudo make install
cd ..
 
# vslsmashsourceのビルド
git clone https://github.com/HolyWu/L-SMASH-Works.git
# ffmpegのバージョンが合わないので、下記バージョンを取得する
cd L-SMASH-Works
git checkout -b 20200531 refs/tags/20200531
cd VapourSynth
meson build
cd build
ninja && sudo ninja install
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
make
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