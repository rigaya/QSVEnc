
# QSVEnc
by rigaya

**[日本語版はこちら＞＞](./Readme.ja.md)**

This software is meant to investigate performance and image quality of HW encoder (QSV) of Intel.
There are 2 types of software developed, one is command line version that runs independently, and the nother is a output plug-in of [Aviutl](http://spring-fragrance.mints.ne.jp/aviutl/).

- QSVEncC.exe ... Command line version supporting transcoding.  
- QSVEnc.auo ... Output plugin for [Aviutl](http://spring-fragrance.mints.ne.jp/aviutl/).

## Downloads & update history
[rigayaの日記兼メモ帳＞＞](http://rigaya34589.blog135.fc2.com/blog-category-10.html)

## System Requirements
### CPU
[CPUs supporting QSV＞＞](#qsv_cpu_list)

### Windows
Windows 7, 8, 8.1, 10 (x86/x64)  
Aviutl 0.99g4 or later (QSVEnc.auo)  

### Linux
CentOS/Readhat (QSVEncC)  
Debian/Ubuntu (QSVEncC)  
  It may be possible to run on other distributions (not tested).


## Precautions for using QSVEnc
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.


## Major Features
### Common to QSVEnc / QSVEncC
- Encoding using QuickSyncVideo
   - H.264/AVC
   - H.265/HEVC (8bit/10bit)
   - MPEG2
- Encode mode of QuickSyncVideo
   - CQP       (Fixed Quantization)
   - CBR       (Constant bitrate)
   - VBR       (Variable bitrate)
   - AVBR      (Adaptive Variable bitrate)
   - QVBR      (Quality based Variable bitrate)
   - LA        (Lookahead mode)
   - LA-HRD    (HRD compatible Lookahead mode)
   - ICQ       (Constant Quality)
   - LA-ICQ    (Constant Quality with Lookahead)
   - VCM       (Video Conference mode)
- Quality preset of 7 steps
- Interlaced encoding (by PAFF)
- supports setting of codec profile & level, SAR, colormatrix, maxbitrate, GOP len, etc...
- supports various vpp(video pre-processing) filters
   - resize
   - deinterlace (normal / bob / it)
   - detail enhancement
   - denoise
   - image stablizier
   - rotation
   - delogo
   - sub hard-burn

### QSVEncC
- Supports QSV(hw) decoding
  - MPEG2
  - H.264/AVC
  - HEVC
  - VC-1
  - VP8
  - VP9
- Supports various formats such as avs, vpy, y4m, and raw
- Supports demux/muxing using libavformat
- Supports decode using libavcodec

### QSVEnc.auo (Aviutl plugin)
- Audio encoding
- Mux audio and chapter
- afs (Automatic field shift) support

<a name ="qsv_cpu_list">
## CPUs supporting QSV

○ ... QSV avail  
× ... QSV unavail  
?  ... differs by SKU or driver versions  

|Intel Core CPUs	        || QSV Availability|
|:--|:--|:-:|
|Penrynn or before	    |Core2 xxx	|×|
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
|Intel Atom CPUs	|||
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


## Intel Media SDK and API
|Media SDK API version	        | Media SDK version|
|:--|:--|
|API v1.26 | Intel Media SDK 2018 R1 |
|API v1.23 | Intel Media SDK 2017 R1 |
|API v1.19 | Intel Media SDK 2016 Update 2 |
|API v1.17 | Intel Media SDK 2016 |
|API v1.16 | Intel Media SDK 2015 Update 2.1 |
|API v1.15 | Intel Media SDK 2015 Update 2 |
|API v1.13 | Intel Media SDK 2015 Update 1 |
|API v1.11 | Intel Media SDK 2015 |
|API v1.9  | Intel Media SDK 2014 R2 for Client |
|API v1.8  | Intel Media SDK 2014 |
|API v1.7  | Intel Media SDK 2013 R2　|
|API v1.6  | Intel Media SDK 2013 |
|API v1.4  | Intel Media SDK 2012 R3 |
|API v1.4  | Intel Media SDK 2012 R2 |
|API v1.3  | Intel Media SDK 2012 (Intel Media SDK v3.0) |
|API v1.1  | Intel Media SDK v2.0  |

## QSVEnc source code
- MIT license.
- This software depends on
  [ffmpeg](https://ffmpeg.org/),
  [libass](https://github.com/libass/libass),
  [tinyxml2](http://www.grinninglizard.com/tinyxml2/),
  [ttmath](http://www.ttmath.org/),
  [dtl](https://github.com/cubicdaiya/dtl).
  For these licenses, please see the header part of the corresponding source and NVEnc_license.txt.

### ソースの構成
Windows ... VC build  
Linux ... makefile + gcc/clang build    

Character code: UTF-8-BOM  
Line feed: CRLF  
Indent: blank x4  
