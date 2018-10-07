
# QSVEncのビルド方法
by rigaya  

## 0. 準備
ビルドには、下記のものが必要です。

- Visual Studio 2015
- yasm
- Avisynth SDK
- VapourSynth SDK

yasmはパスに追加しておきます。

## 1. ソースのダウンロード

```Batchfile
git clone https://github.com/rigaya/QSVEnc --recursive
```

## 2. QSVEnc.auo / QSVEncC のビルド

QSVEnc.slnを開きます。

Avisynth SDKの"avisynth_c.h"、
VapourSynth SDKの"VapourSynth.h", "VSScript.h"が
includeパスに含まれるよう、Visual Studio設定した後、ビルドしてください。

ビルドしたいものに合わせて、構成を選択してください。

|              |Debug用構成|Release用構成|
|:---------------------|:------|:--------|
|QSVEnc.auo (win32のみ) | Debug | Release |
|QSVEncC(64).exe | DebugStatic | RelStatic |
