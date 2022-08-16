
# QSVEnc
by rigaya

[![Build Windows Releases](https://github.com/rigaya/QSVEnc/actions/workflows/build_releases.yml/badge.svg)](https://github.com/rigaya/QSVEnc/actions/workflows/build_releases.yml)  [![Build Linux Packages](https://github.com/rigaya/QSVEnc/actions/workflows/build_packages.yml/badge.svg)](https://github.com/rigaya/QSVEnc/actions/workflows/build_packages.yml)  

このソフトウェアは、IntelのCPUに搭載されているHWエンコーダ(QSV)の画質や速度といった性能の実験を目的としています。
単体で動作するコマンドライン版とAviutlの出力プラグイン版があります。  

- QSVEncC.exe  
  単体で動作するコマンドライン版です。本項で説明します。

- [QSVEnc.auo](./QSVEnc_auo_readme.md)  
  Intel Media SDK を使用してエンコードを行う[Aviutl](http://spring-fragrance.mints.ne.jp/aviutl/)の出力プラグインの使用方法については、[こちら](./QSVEnc_auo_readme.md)を参照してください。

## ダウンロード & 更新履歴
[こちら](https://github.com/rigaya/QSVEnc/releases)

## インストール
インストール方法は[こちら](./Install.ja.md)。

## ビルド
ビルド方法は[こちら](./Build.ja.md)。

## 想定動作環境
### Windows
Windows 10/11 (x86/x64) (QSVEnc.auo / QSVEncC.exe x86版)  
Windows 10/11 (x64) (QSVEncC.exe x64版)  
Aviutl 1.00 以降 (QSVEnc.auo)  

### Linux
CentOS/Readhat系 (QSVEncC)  
Debian/Ubuntu系 (QSVEncC)  
  そのほかのディストリビューションでも動作する可能性があります。

## QSVEnc 使用にあたっての注意事項
無保証です。自己責任で使用してください。  
QSVEncを使用したことによる、いかなる損害・トラブルについても責任を負いません。

## QSVEncCの導入方法
[QSVEncC 導入・使用方法について＞＞](http://rigaya34589.blog135.fc2.com/blog-entry-704.html)

## QSVEncCの使用方法とオプション
[QSVEncCのオプションの説明＞＞](./QSVEncC_Options.ja.md)

## 使用出来る主な機能
### QSVEnc/QSVEncC共通
- QuickSyncVideoを使用したエンコード
   - H.264/AVC
   - H.265/HEVC (8bit/10bit)
   - MPEG2
   - VP9 (8bit/10bit)
- QuickSyncVideoの各エンコードモード
   - CQP       固定量子化量
   - CBR       固定ビットレート
   - VBR       可変ビットレート
   - AVBR      適応的可変ビットレート
   - QVBR      品質ベース可変ビットレート
   - Lookahead 先行探索レート制御
   - LA-HRD    先行探索レート制御 (HRD互換)
   - ICQ       固定品質モード
   - LA-ICQ    先行探索付き固定品質モード
   - VCM       ビデオ会議モード
- エンコード品質の7段階による指定
- インタレ保持エンコード (PAFF方式)
- シーンチェンジ検出
- SAR比指定
- colormatrix等の指定
- H.264 Level / Profileの指定
- Blurayエンコードモード
- 最大ビットレート等の指定
- 最大GOP長の指定
- AVX2までを使用した処理の高速化
- エンコード結果のSSIM/PSNRを計算
- VPP機能
  - Media Functionを使用した高速フィルタリング
    - リサイズ
    - インタレ解除 (normal / bob / it)
    - エッジ強調
    - ノイズ低減
    - Image Stablizier
  - CUDAによるGPUフィルタリング
    - インタレ解除
      - afs (自動フィールドシフト)
      - nnedi
    - decimate
    - mpdecimate
    - delogo
    - 字幕焼きこみ
    - 色空間変換 (x64版のみ)
      - hdr2sdr
    - リサイズ  
      - bilinear
      - spline16, spline36, spline64
      - lanczos2, lanczos3, lanczos4
    - 回転 / 反転
    - パディング(黒帯)の追加
    - バンディング低減
    - ノイズ除去
      - knn (K-nearest neighbor)
      - pmd (正則化pmd法)
    - 輪郭・ディテール強調
      - unsharp
      - edgelevel (エッジレベル調整)
      - warpsharp

### QSVEnc
- 音声エンコード
- 音声及びチャプターとのmux機能
- 自動フィールドシフト対応

### QSVEncC
- QSVデコードに対応
  - MPEG2
  - H.264/AVC
  - HEVC
  - VP8
  - VP9
- QSVによるデコード/VPP/エンコードをそれぞれ自由な組み合わせで使用可能
- エンコードなしの出力も可能
- avi(vfw), avs, vpy, y4m, rawなど各種形式に対応
- libavcodec/libavformatを利用した音声処理に対応
- libavcodec/libavformatを利用し、muxしながら出力が可能

## 各GPUで使用可能な機能
下記はGPU、GPUドライバ、QSVEncのバージョンにより異なります。

| CPU Gen     | GPU Gen |  Windows                                                 | Linux | 
|:--          |:--      |:--                                                      |:--    |
| SandyBridge | Gen6     | [i5 2410M](./GPUFeatures/QSVEnc_SND_i5_2410M_Win.txt)   |  |
| IvyBridge   | Gen7     |                                                         |  |
| Haswell     | Gen7.5   | [i3 4170](./GPUFeatures/QSVEnc_HSW_i3_4170_Win.txt)     |  |
| Broadwell   | Gen8     | [i7 5500U](./GPUFeatures/QSVEnc_BDW_i7_5500U_Win.txt)   | [i7 5500U](./GPUFeatures/QSVEnc_BDW_i7_5500U_Ubuntu2004.txt)  |
| SkyLake     | Gen9     |                                                         |  |
| KabyLake    | Gen9.5   | [i7 7700K](./GPUFeatures/QSVEnc_KBL_i7_7700K_Win.txt)   | [i7 7700K](./GPUFeatures/QSVEnc_KBL_i7_7700K_Ubuntu2204.txt)  |
| CoffeeLake  | Gen9.5   |                                                         |  |
| CommetLake  | Gen9.5   |                                                         |  |
| IceLake     | Gen11    | [i5 1035G7](./GPUFeatures/QSVEnc_ICL_i5_1035G7_Win.txt) | [i5 1035G7](./GPUFeatures/QSVEnc_ICL_i5_1035G7_Ubuntu2004.txt)  |
| RocketLake  | Gen11    | [i7 11700K](./GPUFeatures/QSVEnc_RKL_i7_11700K_Win.txt) | [i7 11700K](./GPUFeatures/QSVEnc_RKL_i7_11700K_Ubuntu2204.txt)  |
| AlderLake   | Gen12    | [i9 12900K](./GPUFeatures/QSVEnc_ADL_i9_12900K_Win.txt) | [i9 12900K](./GPUFeatures/QSVEnc_ADL_i9_12900K_Ubuntu2004.txt) |
|             | DG2      | [Arc A380](./GPUFeatures/QSVEnc_DG2_Arc_A380_Win.txt)   |  |

## ソースコードについて
- MITライセンスです。
- 本ソフトウェアでは、
  [oneVPL](https://github.com/oneapi-src/oneVPL/),
  [ffmpeg](https://ffmpeg.org/),
  [libass](https://github.com/libass/libass),
  [tinyxml2](http://www.grinninglizard.com/tinyxml2/),
  [ttmath](http://www.ttmath.org/),
  [clRNG](https://github.com/clMathLibraries/clRNG),
  [dtl](https://github.com/cubicdaiya/dtl),
  [Caption2Ass](https://github.com/maki-rxrz/Caption2Ass_PCR)を使用しています。
  これらのライセンスにつきましては、該当ソースのヘッダ部分や、license.txtをご覧ください。

### ソースの構成
Windows ... VCビルド  
Linux ... makefile + gcc/clangビルド  

文字コード: UTF-8-BOM  
改行: CRLF  
インデント: 空白x4  
