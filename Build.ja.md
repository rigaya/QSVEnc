
# QSVEncのビルド方法
by rigaya  

## 0. 準備
ビルドには、下記のものが必要です。

- Visual Studio 2015
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

## 1. ソースのダウンロード

```Batchfile
git clone https://github.com/rigaya/QSVEnc --recursive
```

## 2. QSVEnc.auo / QSVEncC のビルド

QSVEnc.slnを開きます。

ビルドしたいものに合わせて、構成を選択してください。

|              |Debug用構成|Release用構成|
|:---------------------|:------|:--------|
|QSVEnc.auo (win32のみ) | Debug | Release |
|QSVEncC(64).exe | DebugStatic | RelStatic |
