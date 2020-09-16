
# How to build QSVEnc

- [Windows](./Build.en.md#windows)
- Linux
  - [Linux (Ubuntu 20.04)](./Build.en.md#linux-ubuntu-2004)
  - [Intel Drivers for Linux](/Build.en.md#Intel-Drivers-for-Linux)

## Windows

### 0. Requirements
To build QSVEnc, components below are required.

- Visual Studio 2019
- yasm
- Avisynth SDK
- VapourSynth SDK
- Intel OpenCL SDK
- Intel Metric Framework SDK (included in Intel Platform Analysis Library)

Please set yasm to your environment PATH.

Install Avisynth+ and VapourSynth, with the SDKs.

Then, "avisynth_c.h" of the Avisynth+ SDK and "VapourSynth.h" of the VapourSynth SDK should be added to the include path of Visual Studio.

These include path can be passed by environment variables "AVISYNTH_SDK" and "VAPOURSYNTH_SDK".

With default installation, environment variables could be set as below.
```Batchfile
setx AVISYNTH_SDK "C:\Program Files (x86)\AviSynth+\FilterSDK"
setx VAPOURSYNTH_SDK "C:\Program Files (x86)\VapourSynth\sdk"
```

The environment variable for Intel OpenCL SDK "INTELOCLSDKROOT" should be set by the installer.

The environment variable for Intel Metric Framework SDK is "INTEL_METRIC_FRAMEWORK_SDK".
As this library is only used for getting GPU/MFX usage, you might want to just disable this feature and skip building this library,
by setting macro ENABLE_METRIC_FRAMEWORK to 0 in QSVPipeline/rgy_version.h.

You will also need source code of Caption2Ass_PCR.

```Batchfile
git clone https://github.com/maki-rxrz/Caption2Ass_PCR <path-to-clone>
setx CAPTION2ASS_SRC Caption2Ass_PCR <path-to-clone>/src
```


### 1. Download source code

```Batchfile
git clone https://github.com/rigaya/QSVEnc --recursive
```

### 2. Build QSVEncC.exe / QSVEnc.auo

Finally, open QSVEnc.sln, and start build of QSVEnc by Visual Studio.

|  |For Debug build|For Release build|
|:--------------|:--------------|:--------|
|QSVEnc.auo (win32 only) | Debug | Release |
|QSVEncC(64).exe | DebugStatic | RelStatic |


## Linux (Ubuntu 20.04)

### 0. Requirements

- C++14 Compiler
- Intel Driver
- yasm
- git
- libraries
  - libva, libdrm, libmfx 
  - ffmpeg 4.x libs (libavcodec58, libavformat58, libavfilter7, libavutil56, libswresample3)
  - libass9
  - [Optional] VapourSynth

### 1. Install build tools

```Shell
sudo apt install build-essential libtool git yasm
```

### 2. Install Intel driver

```Shell
sudo apt install intel-media-va-driver-non-free
```

### 3. Install required libraries

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

### 4. [Optional] Install VapourSynth
VapourSynth is required only if you need VapourSynth(vpy) reader support.  

Please go on to [5. Build QSVEncC] if you don't need vpy reader.

<details><summary>How to build VapourSynth</summary>

#### 4.1 Install build tools for VapourSynth
```Shell
sudo apt install python3-pip autoconf automake libtool meson
```

#### 4.2 Install zimg
```Shell
git clone https://github.com/sekrit-twc/zimg.git
cd zimg
./autogen.sh
./configure
sudo make install -j16
cd ..
```

#### 4.3 Install cython
```Shell
sudo pip3 install Cython
```

#### 4.4 Install VapourSynth
```Shell
git clone https://github.com/vapoursynth/vapoursynth.git
cd vapoursynth
./autogen.sh
./configure
make -j16
sudo make install

# Make sure vapoursynth could be imported from python
# Change "python3.x" depending on your encironment
sudo ln -s /usr/local/lib/python3.x/site-packages/vapoursynth.so /usr/lib/python3.x/lib-dynload/vapoursynth.so
sudo ldconfig
```

#### 4.5 Check if VapourSynth has been installed properly
Make sure you get version number without errors.
```Shell
vspipe --version
```

#### 4.6 [Option] Build vslsmashsource
```Shell
# Install lsmash
git clone https://github.com/l-smash/l-smash.git
cd l-smash
./configure --enable-shared
sudo make install -j16
cd ..
 
# Install vslsmashsource
git clone https://github.com/HolyWu/L-SMASH-Works.git
cd L-SMASH-Works/VapourSynth
meson build
cd build
sudo ninja install
cd ../../../
```

</details>

### 5. Build QSVEncC
```Shell
git clone https://github.com/rigaya/QSVEnc --recursive
cd QSVEnc
./configure
make -j8
```
Check if it works properly.
```Shell
./qsvencc --check-hw
```

You shall get results below if Quick Sync Video works properly.
```
Success: QuickSyncVideo (hw encoding) available
```

## Intel Drivers for Linux
Please refer to Intel Media SDK Wiki for [driver packages for Linux distributions](https://github.com/Intel-Media-SDK/MediaSDK/wiki/Media-SDK-in-Linux-Distributions) and [further infomation about Intel media stack on Ubuntu](https://github.com/Intel-Media-SDK/MediaSDK/wiki/Intel-media-stack-on-Ubuntu).