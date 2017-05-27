
# How to build QSVEnc
by rigaya  

## 0. Requirements
To build QSVEnc, components below are required.

- Visual Studio 2015
- yasm
- Avisynth SDK
- VapourSynth SDK

Please set yasm to your environment PATH.

## 1. Download source code

```Batchfile
git clone https://github.com/rigaya/QSVEnc --recursive
```

## 2. Build ffmpeg dll

QSVEncC requires ffmpeg dlls, and it should be placed as the structure below.
```
QSVEnc root
 |-QSVEnc
 |-QSVEncC
 |-QSVEncCore
 |-QSVEncSDK
 |-<others>...
 `-ffmpeg_lgpl
    |- include
    |   |-libavcodec
    |   |  `- libavcodec header files
    |   |-libavfilter
    |   |  `- libavfilter header files
    |   |-libavformat
    |   |  `- libavfilter header files
    |   |-libavutil
    |   |  `- libavutil header files
    |   `-libswresample
    |      `- libswresample header files
    `- lib
        |-win32 (for win32 build)
        |  `- avocdec, avfilter, avformat, avutil, swresample
        |     x86 lib & dlls
        `- x64 (for x64 build)
           `- avocdec, avfilter, avformat, avutil, swresample
              x64 lib & dlls
```

One of the way to build ffmpeg dlls is to use msys+mingw, and when Visual Studio's environment path is set, ffmpeg will build dlls & libs on shared lib build.

For example, if you need x64 build, you can set Visual Studio's environment path be calling vcvarsall.bat before msys.bat call.

```Batchfile
call "%VS140COMNTOOLS%\..\..\VC\vcvarsall.bat" x64
call msys.bat
```
Then, by configuring with options below, dlls will be built automatically.
```
--enable-shared --enable-swresample
```


## 3. QSVEnc.auo / QSVEncC のビルド

After preparations are done, open QSVEnc.sln, and set headers below in the include path.

 - "avisynth_c.h"、
 - "VapourSynth.h", "VSScript.h"

Finally, start build of QSVEnc by Visual Studio.

||For Debug build|For Release build|
|:--------------|:----------------------------------|
|QSVEnc.auo (win32 only) | Debug | Release |
|QSVEncC(64).exe | DebugStatic | RelStatic |
