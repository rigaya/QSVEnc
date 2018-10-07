
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

## 2. Build QSVEncC.exe / QSVEnc.auo

After preparations are done, open QSVEnc.sln, and set headers below in the include path.

 - "avisynth_c.h"、
 - "VapourSynth.h", "VSScript.h"

Finally, start build of QSVEnc by Visual Studio.

|  |For Debug build|For Release build|
|:--------------|:--------------|:--------|
|QSVEnc.auo (win32 only) | Debug | Release |
|QSVEncC(64).exe | DebugStatic | RelStatic |
