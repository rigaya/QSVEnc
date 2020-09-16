
# QSVEncのビルド方法

- [Windows](./Build.en.md#windows)
- Linux
  - [Linux (Ubuntu 20.04)](./Build.en.md#linux-ubuntu-2004)
  - [Intel Drivers for Linux](/Build.en.md#Intel-Drivers-for-Linux)

## Windows 

### 0. 準備
ビルドには、下記のものが必要です。

- Visual Studio 2019
- yasm
- Avisynth SDK
- VapourSynth SDK
- Intel OpenCL SDK
- Intel Metric Framework SDK (Intel Platform Analysis Libraryに同梱)

yasmはパスに追加しておきます。

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

さらにビルドに必要なCaption2Ass_PCRをcloneし、環境変数 "CAPTION2ASS_SRC" を設定します。

```Batchfile
git clone https://github.com/maki-rxrz/Caption2Ass_PCR <path-to-clone>
setx CAPTION2ASS_SRC Caption2Ass_PCR <path-to-clone>/src
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

- C++14 Compiler
- Intel Driver
- yasm
- git
- libraries
  - libva, libdrm, libmfx 
  - ffmpeg 4.x libs (libavcodec58, libavformat58, libavfilter7, libavutil56, libswresample3)
  - libass9
  - [Optional] VapourSynth

### 1. コンパイラ等のインストール

```Shell
sudo apt install build-essential libtool git yasm
```

### 2. Intel ドライバのインストール

```Shell
sudo apt install intel-media-va-driver-non-free
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
git checkout -b 20200531 refs/tags/20200531
cd L-SMASH-Works/VapourSynth
meson build
cd build
sudo ninja install
cd ../../../
```

</details>

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

## Intel Drivers for Linux
各Linux distributionのIntelドライバのパッケージについては、Intel Media SDK Wikiの[こちら](https://github.com/Intel-Media-SDK/MediaSDK/wiki/Media-SDK-in-Linux-Distributions)を参照してください。

またドライバの詳細については、Ubuntuの例になりますが、[こちら]((https://github.com/Intel-Media-SDK/MediaSDK/wiki/Intel-media-stack-on-Ubuntu))をご覧ください。