
# QSVEnc
by rigaya

このソフトウェアは、IntelのCPUに搭載されているHWエンコーダ(QSV)の画質や速度といった性能の実験を目的としています。
Aviutlの出力プラグイン版と単体で動作するコマンドライン版があります。  

QSVEnc.auo … Intel Media SDK を使用してエンコードを行うAviutlの出力プラグインです。

QSVEncC.exe … 上記のコマンドライン版です。

## ダウンロード & 更新履歴
[rigayaの日記兼メモ帳＞＞](http://rigaya34589.blog135.fc2.com/blog-category-10.html)

## 基本動作環境
### CPU
[QSVが利用可能なCPU一覧＞＞](#qsv_cpu_list)

### Windows
Windows 7, 8, 8.1, 10 (x86/x64) (QSVEnc.auo / QSVEncC.exe x86版)  
Windows 7, 8, 8.1, 10 (x64) (QSVEncC.exe x64版)  
Aviutl 0.99g4 以降 (QSVEnc.auo)  

### Linux
CentOS/Readhat系 (QSVEncC)  
Debian/Ubuntu系 (QSVEncC)  
  そのほかのディストリビューションでも動作する可能性があります。

## QSVEnc 使用にあたっての注意事項
無保証です。自己責任で使用してください。  
QSVEncを使用したことによる、いかなる損害・トラブルについても責任を負いません。

## QSVEncCの導入方法
[QSVEncC 導入・使用方法について＞＞](http://rigaya34589.blog135.fc2.com/blog-entry-704.html)

## QSVEncCのオプション
[QSVEncCのオプションの説明＞＞](http://rigaya34589.blog135.fc2.com/blog-entry-337.html)


## 使用出来る主な機能
### QSVEnc/QSVEncC共通
- QuickSyncVideoを使用したエンコード
   - H.264/AVC
   - H.265/HEVC (8bit/10bit)
   - MPEG2
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
- Vpp機能 -- GPUを使用した高速フィルタリング
   - リサイズ
   - インタレ解除 (normal / bob / it)
   - エッジ強調
   - ノイズ低減
   - Image Stablizier
   - 回転
   - ロゴ消し(delogo) (CPU処理)
   - 字幕焼き込み

### QSVEnc
- 音声エンコード
- 音声及びチャプターとのmux機能
- 自動フィールドシフト対応

### QSVEncC
- QSVデコードに対応
  - MPEG2
  - H.264/AVC
  - HEVC
  - VC-1
  - VP8
  - VP9
- QSVによるデコード/VPP/エンコードをそれぞれ自由な組み合わせで使用可能
- エンコードなしの出力も可能
- avi(vfw), avs, vpy, y4m, rawなど各種形式に対応
- libavcodec/libavformatを利用した音声処理に対応
- libavcodec/libavformatを利用し、muxしながら出力が可能

<a name ="qsv_cpu_list">
## QSVの使用できるCPU一覧

○　... 使用可能 / × ... 使用不可 / ? ... 型番、ドライバにより異なる

|Core系 CPU	        || QSVの使用|
|:--|:--|:-:|
|Penrynn以前	    |Core2 xxx	|×|
|Nehalem/LynnField	|Core i3/i5/i7 xxx	|×|
|SandyBridge	    |Core i3/i5/i7 2xxx	|○|
|SandyBridge-E	    |Core i7 3xxx	|×|
|IvyBridge	        |Core i3/i5/i7 3xxx	|○|
|IvyBridge-E	    |Core i7 4xxx	|×|
|Haswell	        |Core i3/i5/i7 4xxx	|○|
|Haswell	        |Pentium G3xxx	|○|
|Haswell	        |Celeron G1xxx	|○|
|Haswell-E	        |Core i7 5xxx	|×|
|Broadwell	        |Core M/i3/i5/i7 5xxx	|○|
|Skylake	        |Core i3/i5/i7 6xxx	|○|
|Skylake	        |Pentium G44xx/G450x/G452x	|○|
|Skylake	        |Celeron G390x/G392x	|○|
|Kabylake	        |Core i3/i5/i7 7xxx	|○|
|Kabylake	        |Pentium G456x/G46xx	|○|
|Kabylake	        |Celeron G393x/G395x	|○|
|Skylake-X	        |Core i7/i9 78xx/79xx	|×|
|Coffeelake	        |Core i3/i5/i7 8xxx	|○|
|Coffeelake	        |Pentium G5xxx	|○|
|Coffeelake	        |Celeron G49xx	|○|
|Atom系 CPU	|||
|Bonnell	        |Atom	        |×|
|Saltwell	        |Atom	        |×|
|Silvermont	        |Pentium N3xxx/J2xxx	|?|
|Silvermont	        |Celeron N2xxx/J1xxx	|?|
|Silvermont	        |Atom Z3xxx/C2xxx	    |?|
|Airmont	        |Pentium N3xxx/J3xxx	|○|
|Airmont	        |Celeron N3xxx/J3xxx	|○|
|Airmont	        |Atom x7/x5/x3	|○|
|GeminiLake	        |Pentium N5xxx/J5xxx	|○|
|GeminiLake	        |Celeron N4xxx/J4xxx	|○|


## Intel Media SDKとAPIの対応関係
|Media SDK API バージョン	        | Media SDK バージョン|
|:--|:--|
|API v1.26 | Intel Media SDK 2018 R1 |
|API v1.23 | Intel Media SDK 2017 R1 |
|API v1.19 | Intel Media SDK 2016 Update 2 |
|API v1.17 | Intel Media SDK 2016 |
|API v1.16 | Intel Media SDK 2015 Update 2.1 |
|API v1.15 | Intel Media SDK 2015 Update 2 |
|API v1.13 | Intel Media SDK 2015 Update 1 |
|API v1.11 | Intel Media SDK 2015 |
|API v1.10 | Intel Media SDK 2014 R2 for Server (有料?) |
|API v1.9  | Intel Media SDK 2014 R2 for Client |
|API v1.8  | Intel Media SDK 2014 |
|API v1.7  | Intel Media SDK 2013 R2　|
|API v1.6  | Intel Media SDK 2013 |
|API v1.4  | Intel Media SDK 2012 R3 |
|API v1.4  | Intel Media SDK 2012 R2 |
|API v1.3  | Intel Media SDK 2012 (Intel Media SDK v3.0) |
|API v1.1  | Intel Media SDK v2.0  |

## ソースコードについて
- MITライセンスです。
- 本ソフトウェアでは、
  [ffmpeg](https://ffmpeg.org/),
  [libass](https://github.com/libass/libass),
  [tinyxml2](http://www.grinninglizard.com/tinyxml2/),
  [ttmath](http://www.ttmath.org/),
  [dtl](https://github.com/cubicdaiya/dtl)を使用しています。
  これらのライセンスにつきましては、該当ソースのヘッダ部分や、license.txtをご覧ください。

### ソースの構成
Windows ... VCビルド  
Linux ... makefile + gcc/clangビルド  

文字コード: UTF-8-BOM  
改行: CRLF  
インデント: 空白x4  
