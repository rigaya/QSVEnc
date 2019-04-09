
# How to build QSVEnc
by rigaya  

## 0. Requirements
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


## 1. Download source code

```Batchfile
git clone https://github.com/rigaya/QSVEnc --recursive
```

## 2. Build QSVEncC.exe / QSVEnc.auo

Finally, open QSVEnc.sln, and start build of QSVEnc by Visual Studio.

|  |For Debug build|For Release build|
|:--------------|:--------------|:--------|
|QSVEnc.auo (win32 only) | Debug | Release |
|QSVEncC(64).exe | DebugStatic | RelStatic |
