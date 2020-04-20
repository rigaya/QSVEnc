
# QSVEncC オプションリスト


## コマンドラインの例


### 基本的なコマンドの表記
```Batchfile
QSVEncC.exe [Options] -i <filename> -o <filename>
```

### もっと実用的なコマンド
#### qsvデコードを使用する例
```Batchfile
QSVEncC --avhw -i "<mp4(H.264/AVC) file>" -o "<outfilename.264>"
```

#### qsvデコードを使用する例 (インタレ保持)
```Batchfile
QSVEncC --avhw --interlace tff -i "<mp4(H.264/AVC) file>" -o "<outfilename.264>"
```

#### avs(Avisynth)の例 (avsやvpyはvfw経由でも読み込み可能です)
```Batchfile
QSVEncC -i "<avsfile>" -o "<outfilename.264>"
```

#### パイプ利用の例
```Batchfile
avs2pipemod -y4mp "<avsfile>" | QSVEncC --y4m -i - -o "<outfilename.264>"
```

#### ffmpegからパイプ渡し

```Batchfile
ffmpeg -y -i "<ソース動画>" -an -pix_fmt yuv420p -f yuv4mpegpipe - | QSVEncC --y4m -i - -o "<outfilename.264>"
```

#### ffmpegから映像と音声を両方パイプ渡したい
--> "nut"フォーマットでくるんで受け渡しするとよいでしょう
```Batchfile
ffmpeg -y -i "<input>" <options for ffmpeg> -codec:a copy -codec:v rawvideo -pix_fmt yuv420p -f nut - | NVEncC --avsw -i - --audio-codec aac -o "<outfilename.mp4>"
```


#### raw H.264/ESのmux
H.264/ESで出力し、mp4に格納したり、AAC音声とmuxする場合には、L-SMASHを使って、

```Batchfile
muxer.exe -i "<raw H.264/ES file>" -i "<ADTS-AAC>" -o "<muxed mp4 file>"
```

としてAAC音声と多重化できます。音声がALACの場合には、

```Batchfile
muxer.exe -i "<raw H.264/ES file>" -o "<video mp4file>"
remuxer.exe -i "<video mp4file>" -i "<m4a(ALAC in mp4)file>" -o "<muxed mp4 file>"
```

のように2段階のステップが必要です。

同様にmkvtoolnixに含まれるmkvmergeでmuxし、mkvに格納することもできます。


## オプションの指定方法

```
-<短縮オプション名>、--<オプション名> <引数>  
引数なしの場合は単体で効果を発揮。

引数のタイプは
- なし
- <int>　　 整数で指定
- <float>　小数点で指定
- <string> 文字列で指定

引数の [ ] 内は、省略可能です。

--(no-)xxx
と付いている場合は、--no-xxxとすることで、--xxxと逆の効果を得る。  
例1: --xxx : xxxを有効にする → --no-xxx: xxxを無効にする  
例2: --xxx : xxxを無効にする → --no-xxx: xxxを有効にする
```

## 表示系オプション

### -h,-? --help
ヘルプの表示

### -v, --version
バージョンの表示

### --check-hw
ハードウェアエンコの可否の表示。

### --check-features
QSVEncの使用可能なエンコード機能を表示する。

### --check-features-html [&lt;string&gt;]
QSVEncの使用可能なエンコード機能情報を指定したファイルにhtmlで出力する。
特に指定がない場合は、"qsv_check.html"に出力する。

### --check-environment
QSVEncCの認識している環境情報を表示

### --check-codecs, --check-decoders, --check-encoders
利用可能な音声コーデック名を表示

### --check-profiles &lt;string&gt;
指定したコーデックの利用可能な音声プロファイル名を表示

### --check-formats
利用可能な出力フォーマットを表示

### --check-protocols
利用可能なプロトコルを表示

### --check-filters
利用可能な音声フィルタを表示

### --check-avversion
dllのバージョンを表示

## エンコードの基本的なオプション

### -c, --codec &lt;string&gt;
エンコードするコーデックの指定
 - h264 (デフォルト)
 - hevc
 - mpeg2
 - raw

### -o, --output &lt;string&gt;
出力ファイル名の表示、"-"でパイプ出力

### -i, --input &lt;string&gt;
入力ファイル名の設定、"-"でパイプ入力

QSVEncの入力方法は下の表のとおり。入力フォーマットをしてしない場合は、拡張子で自動的に判定される。

| 使用される読み込み |  対象拡張子 |
|:---|:---|          
| Avisynthリーダー    | avs |
| VapourSynthリーダー | vpy |
| aviリーダー         | avi |
| y4mリーダー         | y4m |
| rawリーダー         | yuv |
| avhw/avswリーダー | それ以外 |

| 入力方法の対応色空間 | yuv420 | yuy2 | yuv422 | yuv444 | rgb24 | rgb32 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| raw | ○ |  |  |  |  |  |
| y4m | ◎ |  | ◎ | ◎ |  |  |
| avi | ○ | ○ |  |  | ○ | ○ |
| avs | ◎ | ○ | ◎ | ◎ |  |  |
| vpy | ◎ |  | ◎ | ◎ |  |  |
| avhw | ◎ |  |  |  |  |  |
| avsw | ◎ |  | ◎ | ◎ | ○ | ○ |

◎ … 8bit / 9bit / 10bit / 12bit / 14bit / 16bitに対応  
○ … 8bitのみ対応

### --raw
入力をraw形式に設定する。
入力解像度、入力fpsの設定が必要。

### --y4m
入力をy4m(YUV4MPEG2)形式として読み込む。

### --avi
入力ファイルをaviファイルとして読み込む。

### --avs
入力ファイルをAvisynthで読み込む。

### --vpy
### --vpy-mt
入力ファイルをVapourSynthで読み込む。vpy-mtはマルチスレッド版。

### --avsw
avformat + sw decoderを使用して読み込む。
ffmpegの対応するほとんどのコーデックを読み込み可能。

### --avhw
avformat + QSV decoderを使用して読み込む。
デコードからエンコードまでを一貫してGPUで行うため高速。

| コーデック | 対応状況 |
|:---|:---:|
| MPEG1      | ○ |
| MPEG2      | ○ |
| H.264/AVC  | ○ |
| H.265/HEVC | ○ |
| VP8        | × |
| VP9        | ○ |
| VC-1       | × |
| WMV3/WMV9  | × |

### --interlace &lt;string&gt;
**入力**フレームがインターレースかどうかと、そのフィールドオーダーを設定する。

[--vpp-deinterlace](#--vpp-deinterlace-string)によりQSVEncC内でインタレ解除を行ったり、そのままインタレ保持エンコードを行う。(インタレ保持エンコードはH.264のみ)

- none ... プログレッシブ
- tff ... トップフィールドファースト
- bff ... ボトムフィールドファースト

### --crop &lt;int&gt;,&lt;int&gt;,&lt;int&gt;,&lt;int&gt;
左、上、右、下の切り落とし画素数。

### --fps &lt;int&gt;/&lt;int&gt; or &lt;float&gt;
入力フレームレートの設定。raw形式の場合は必須。

### --input-res &lt;int&gt;x&lt;int&gt;
入力解像度の設定。raw形式の場合は必須。

### --output-res &lt;int&gt;x&lt;int&gt;
出力解像度の設定。入力解像度と異なる場合、自動的にHW/GPUリサイズを行う。

指定がない場合、入力解像度と同じになり、リサイズは行われない。

_特殊な値について_
- 0 ... 入力解像度と同じ
- 縦横のどちらかを負の値  
  アスペクト比を維持したまま、片方に合わせてリサイズ。ただし、その負の値で割り切れる数にする。

```
例: 入力が1280x720の場合
--output-res 1024x576 -> 通常の指定方法
--output-res 960x0    -> 960x720にリサイズ (0のほうは720のまま)
--output-res 1920x-2  -> 1920x1080にリサイズ (アスペクト比が維持できるように調整)
```


## エンコードモードのオプション

デフォルトはCQP(固定量子化量)。

### --cqp &lt;int&gt; or &lt;int&gt;:&lt;int&gt;:&lt;int&gt;　(CQP, 固定量子化量)
CQP(固定量子化量)でエンコードを行う。&lt;Iフレーム&gt;:&lt;Pフレーム&gt;:&lt;Bフレーム&gt;のQP値を設定。
設定値は低い値ほど高品質になる。
基本的にQP値は I &lt; P &lt; B になるように設定することをおすすめ。

### --cbr &lt;int&gt;  (CBR, 固定ビットレート)
### --vbr &lt;int&gt;  (VBR, 可変ビットレート)
### --avbr &lt;int&gt; (AVBR, 適応的可変ビットレート)
### --la &lt;int&gt;   (LA, 先行探索レート制御, lookahead)
### --la-hrd &lt;int&gt; (LA-HRD, 先行探索レート制御 (HRD互換), lookahead)
### --vcm &lt;int&gt; (VCM, ビデオ会議モード)
ビットレートをkbps単位で指定してエンコードを行う。
AVBRはAPI v1.3以降で、より柔軟なビットレート配分が行える。
Lookaheadは更に多くのフレームをあらかじめ解析し、より最適なビットレート配分を行う。

### --qvbr &lt;int&gt;, --qvbr-q &lt;int&gt; (QVBR, 品質ベース可変ビットレート)
### --qvbr-qで指定した品質(デフォルト 23)をベースに、--qvbrで指定したビットレートでエンコードを行う。
設定値は低い値ほど高品質になる。

### --icq &lt;int&gt; (ICQ, 固定品質モード: デフォルト 23)
### --la-icq &lt;int&gt; (LA-ICQ, 先行探索付き固定品質モード: デフォルト 23)
固定品質系のモード。設定値は低い値ほど高品質になる。

### --fallback-rc
使用できないレート制御モードが指定された場合に、エラー終了するのではなく、自動的により一般的にサポートされるレート制御モードにフォールバックする。ビットレート指定系なら最終的にvbrを、品質指定系なら最終的にcqpを使用する。

**エンコードモードの選択について**
CBR/VBR/AVBRなどのモードは高速かかわりに画質が悪い傾向があり、容量を喰うところで映像が破綻しやすい。
画質を維持しながらエンコードするには、なるべく固定品質系や先行探索レート制御系のオプション(ICQ, LA-ICQ, QVBR, LA等)を使用したほうが良いと思う。
ある程度容量が膨れてもよい場合には、低めのQP値に指定したCQPモードが高速でよいかもしれない。

ただし、特定機器用エンコード、例えばBD用エンコードのような、
上限ビットレートを気にする場合があるときには、序湧現ビットレートを指定可能なVBR/AVBRモードを使用する必要がある。


## フレームバッファのオプション

**フレームバッファの種類**  

| OS  | システムメモリ | ビデオメモリ |
|:---|:---:|:---:|
| Windows | system | d3d9 / d3d11 |
| Linux   | system | va           |

フレームバッファの種類の決定は、デフォルトでは下記のように自動で行われる。

**Windows**  
<u>QSVエンコード使用時:</u>  
基本的には、より高速なd3d9メモリを使用するが、dGPUが存在するシステムでは、d3d9メモリが使用できないため、d3d11メモリを使用する。

<u>QSVエンコードを使用しない場合 (デコードのみの場合):</u>  
ビデオメモリはQSVでの処理は高速だが、そのフレームをCPU側に転送するのが非常に遅い。このため、エンコードをせず、デコードやVPPの結果を他のアプリケーションに渡す場合、systemメモリを使用する。

**Linux**  
安定性確保のため、systemメモリを使用する。


### --disable-d3d (Win)
### --disable-va (Linux)
ハードウェアエンコ時にバッファとしてビデオメモリでなく、CPU側のメモリを使用する。

### --d3d9
d3d9ビデオメモリを使用する。 (Windows)

### --d3d11
d3d11ビデオメモリを使用する。 (Windows)

### --va
vaビデオメモリを使用する。(Linux)


## その他のオプション

### --fixed-func
従来の部分的にGPU EUを使用したエンコードではなく、エンコードの全工程で固定回路(Fixed Func)を使用し、完全HWエンコを行う。
GPUに負荷をかけることなく、低電力でエンコード可能だが、品質はやや劣る。

### --max-bitrate &lt;int&gt;
最大ビットレート(kbps単位)。

### --vbv-bufsize &lt;int&gt;
VBVバッファサイズ (kbps単位).

### --avbr-unitsize &lt;int&gt;
AVBRモード時のビットレート配分単位を、100フレーム単位で指定する。デフォルトは90(=9000フレーム)。Intel Media SDKにもあまり説明がなく、正直よくわからない。

### --qp-min &lt;int&gt; or &lt;int&gt;:&lt;int&gt;:&lt;int&gt;
最小QP値を&lt;Iフレーム&gt;:&lt;Pフレーム&gt;:&lt;Bフレーム&gt;で設定する。
ビットレート指定のエンコードモード使用時のみ有効。設定したQP値より低いQP値は使用されなくなる。

ビットレート指定モードなどで、静止画などの部分で過剰にビットレートが割り当てられることがあるのを抑制したりするのに使用する。

### --qp-max &lt;int&gt; or &lt;int&gt;:&lt;int&gt;:&lt;int&gt;
最大QP値を&lt;Iフレーム&gt;:&lt;Pフレーム&gt;:&lt;Bフレーム&gt;設定する。
ビットレート指定のエンコードモード使用時のみ有効。設定したQP値より高いQP値は使用されなくなる。

指定したビットレートを超えてでも、動画のどんな部分でもある程度の品質を維持したい場合に使用する。

### --qp-offset &lt;int&gt;[:&lt;int&gt;][:&lt;int&gt;]...
ピラミッド参照の階層ごとに使用するQPのオフセットを指定する。(デフォルト = 0)

例:ピラミッド参照の第1層目のQPを+1、 第2層目を+2。
```
--qp-offset 1:2
```

### -u, --quality &lt;string&gt;
エンコード品質と速度のバランスの設定。速度重視か画質重視かを決める。数字でもOKで、bestが1で、fastestが7。
```
best, higher, high, balanced(default), fast, faster, fastest
```

### --la-depth &lt;int&gt;
先行探索レート制御を使用した場合に、あらかじめ分析するフレームの枚数を指定する。(10-100)  
インタレ保持の場合には、上限は半分の50となる。

### --la-window-size &lt;int&gt; 0(自動)
先行探索レート制御使用時の、上限ビットレートの計算幅をフレームの枚数で指定する。--max-bitrateと組み合わせて使用する。

### --la-quality &lt;string&gt;
先行探索の品質を設定する。slowになるほど遅くなるが品質が向上する。
mediumやfastでは、先行探索を縮小されたフレームで行い、高速に処理する。
- auto (default)
- fast ... x1/4の解像度で高速な分析を行う。
- medium ... x1/2の解像度で分析を行う。
- slow　... 等倍のフレームで高品質な分析を行う。

### --mbbrc
マクロブロック単位でのレート制御を有効にする。デフォルトでは"-u"オプションに従い自動的にオン/オフが切り替わる。

### --i-adapt
適応的なIフレーム挿入を有効化する。

### --b-adapt
適応的なBフレーム挿入を有効化する。

### --strict-gop
固定GOP長を強制する。

### --gop-len &lt;int&gt;
最大GOP長。

### -b, --bframes &lt;int&gt;
連続Bフレーム数。

### --ref &lt;int&gt;
参照距離を設定する。QSVEncではあまり増やしても品質は向上しない。

### --b-pyramid
Bフレームピラミッド参照を有効にする。

### --weightb
重み付きBフレームを使用する。

### --weightp
重み付きPフレームを使用する。

### --adapt-ltr
Adaptive LTRを有効にする。

### --mv-scaling &lt;string&gt;
動きベクトルのコストの調整。
- 0 ... 動きベクトルのコストを0として見積もる
- 1 ... 動きベクトルのコストをx1/2として見積もる
- 2 ... 動きベクトルのコストをx1/4として見積もる
- 3 ... 動きベクトルのコストをx1/8として見積もる

### --slices &lt;int&gt;
スライス数。指定なし、あるいは0で自動。

### --level &lt;string&gt;
エンコードするコーデックのLevelを指定する。指定しない場合は自動的に決定される。
```
h264:  auto, 1, 1b, 1.1, 1.2, 1.3, 2, 2.1, 2.2, 3, 3.1, 3.2, 4, 4.1, 4.2, 5, 5.1, 5.2
hevc:  auto, 1, 2, 2.1, 3, 3.1, 4, 4.1, 5, 5.1, 5.2, 6, 6.1, 6.2
mpeg2: auto, low, main, high, high1440
```

### --profile &lt;string&gt;
エンコードするコーデックのプロファイルを指定する。指定しない場合は自動的に決定される。
```
h264:  auto, baseline, main, high, high444
hevc:  auto, main, main10, main444
mpeg2: auto, Simple, Main, High
```

### --tier &lt;string&gt;
コーデックのtierを指定する。
```
hevc:  main, high
```

### --sar &lt;int&gt;:&lt;int&gt;
SAR比 (画素アスペクト比) の指定。

### --dar &lt;int&gt;:&lt;int&gt;
DAR比 (画面アスペクト比) の指定。

### --fullrange
フルレンジYUVとしてエンコードする。

### --colorrange &lt;string&gt;
"--colorrange full"は"--fullrange"に同じ。
"auto"を指定することで、入力ファイルの値をそのまま反映できます。([avhw](#--avhw)/[avsw](#--avsw)読み込みのみ)
```
  limited, full, auto
```

### --videoformat &lt;string&gt;
"auto"を指定することで、入力ファイルの値をそのまま反映できます。([avhw](#--avhw)/[avsw](#--avsw)読み込みのみ)
```
  undef, auto, ntsc, component, pal, secam, mac
```
### --colormatrix &lt;string&gt;
"auto"を指定することで、入力ファイルの値をそのまま反映できます。([avhw](#--avhw)/[avsw](#--avsw)読み込みのみ)
```
  undef, auto, bt709, smpte170m, bt470bg, smpte240m, YCgCo, fcc, GBR, bt2020nc, bt2020c
```
### --colorprim &lt;string&gt;
"auto"を指定することで、入力ファイルの値をそのまま反映できます。([avhw](#--avhw)/[avsw](#--avsw)読み込みのみ)
```
  undef, auto, bt709, smpte170m, bt470m, bt470bg, smpte240m, film, bt2020
```
### --transfer &lt;string&gt;
"auto"を指定することで、入力ファイルの値をそのまま反映できます。([avhw](#--avhw)/[avsw](#--avsw)読み込みのみ)
```
  undef, auto, bt709, smpte170m, bt470m, bt470bg, smpte240m, linear,
  log100, log316, iec61966-2-4, bt1361e, iec61966-2-1,
  bt2020-10, bt2020-12, smpte2084, smpte428, arib-std-b67
```  

### --chromaloc &lt;int&gt; or "auto"
出力データのchroma location flagを 0 ～ 5 の範囲で指定する。  
デフォルト = 0 (unspecified)

### --max-cll &lt;int&gt;,&lt;int&gt; [HEVCのみ]
MaxCLL and MaxFall を nits で指定する。"copy"とすると入力ファイルの値を出力ファイルにそのまま設定します。
```
例1: --max-cll 1000,300
例2: --max-cll copy  # 入力ファイルから値をコピー
```

### --master-display &lt;string&gt; [HEVCのみ]
Mastering display data の設定。"copy"とすると入力ファイルの値を出力ファイルにそのまま設定します。
```
例1: --master-display G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)
例2: --master-display copy  # 入力ファイルから値をコピー
```

### --aud
Access Unit Delimiter NALを挿入する。

### --pic-struct
picture timing SEIを挿入する。

### --bluray [H.264のみ]
Bluray用出力を行う。(デフォルト: オフ)  
Bluray互換となるよう、一部の内部パラメータを自動的に調整する。BD用エンコードの場合は必須。また、ビットレート指定系のエンコモードしか使用できない(CQP/VQPは使用できない)。

### --repartition-check
小さなpartitionからの予測を有効にする。 [H.264のみ]

### --trellis &lt;string&gt; [H.264のみ]
H.264の歪みレート最適化の適用範囲を指定する。
- auto(default)
- off
- i
- ip
- all

### --no-deblock
H.264のデブロックフィルタを無効化する。[H.264]

### --tskip
transform skip の有効化。[HEVC]

### --sao &lt;string&gt;
SAOのモードの切り替え。 [HEVC]
- auto    ... デフォルト
- none    ... SAOの無効化
- luma    ... 輝度成分についてSAOを使用
- chroma  ... 色差成分についてSAOを使用
- all     ... 輝度 & 色差成分についてSAOを使用

### --ctu &lt;int&gt;
エンコードで使用するctuサイズの最大値を指定。16, 32, 64のいずれか。 [HEVC]

## 入出力 / 音声 / 字幕などのオプション

### --input-analyze &lt;int&gt;
libavが読み込み時に解析するファイルの時間を秒で指定。デフォルトは5。
音声トラックなどが正しく抽出されない場合、この値を大きくしてみてください(例:60)。

### --trim &lt;int&gt;:&lt;int&gt;[,&lt;int&gt;:&lt;int&gt;][,&lt;int&gt;:&lt;int&gt;]...
指定した範囲のフレームのみをエンコードする。

```
例1: --trim 0:1000,2000:3000    (0～1000フレーム目, 2000～3000フレーム目をエンコード)
例2: --trim 2000:0              (2000～最終フレームまでをエンコード)
```

### --seek [&lt;int&gt;:][&lt;int&gt;:]&lt;int&gt;[.&lt;int&gt;]
書式は、hh:mm:ss.ms。"hh"や"mm"は省略可。
高速だが不正確なシークをしてからエンコードを開始する。正確な範囲指定を行いたい場合は[--trim](#--trim-intintintintintint)で行う。
```
例1: --seek 0:01:15.400
例2: --seek 1:15.4
例3: --seek 75.4
```

### --input-format &lt;string&gt;
avhw/avswリーダー使用時に、入力のフォーマットを指定する。

### -f, --output-format &lt;string&gt;
muxerに出力フォーマットを指定して出力する。

出力フォーマットは出力拡張子から自動的に決定されるので、通常、特に指定する必要はないが、このオプションで出力フォーマットを強制できる。

使用可能なフォーマットは[--check-formats](#--check-formats)で確認できる。H.264/HEVCをElementary Streamで出力する場合には、"raw"を指定する。

### --video-track &lt;int&gt;
エンコード対象の映像トラックの選択。avsw/avhwリーダー使用時のみ有効。
 - 1  ... 最も高解像度の映像トラック (デフォルト)
 - 2  ... 2番目に高解像度の映像トラック
    ...
 - -1 ... 最も低解像度の映像トラック
 - -2 ... 2番目に低解像度の映像トラック

### --video-streamid &lt;int&gt;
エンコード対象の映像トラックをstream idで選択。

### --video-tag &lt;string&gt;
映像のcodec tagの指定。
```
 -o test.mp4 -c hevc --video-tag hvc1
```

### --audio-copy [&lt;int&gt;[,&lt;int&gt;]...]
音声をそのままコピーしながら映像とともに出力する。avhw/avswリーダー使用時のみ有効。

tsなどでエラーが出るなどしてうまく動作しない場合は、[--audio-codec](#--audio-codec-intstring)で一度エンコードしたほうが安定動作するかもしれない。

```
例: トラック番号#1,#2を抽出
--audio-copy 1,2
```

### --audio-codec [[&lt;int&gt;?]&lt;string&gt;[:&lt;string&gt;=&lt;string&gt;][,&lt;string&gt;=&lt;string&gt;][#&lt;string&gt;=&lt;string&gt;][,&lt;string&gt;=&lt;string&gt;]...]
音声をエンコードして映像とともに出力する。使用可能なコーデックは[--check-encoders](#--check-codecs---check-decoders---check-encoders)で確認できる。

[&lt;int&gt;]で、抽出する音声トラック(1,2,...)を指定することもできる。

また、
- ":"のあとに[&lt;string&gt;=&lt;string&gt;]で音声エンコーダのオプション
- "#"のあとに[&lt;string&gt;=&lt;string&gt;]で音声デコーダのオプション
を指定できる。
```
例1: 音声をmp3に変換
--audio-codec libmp3lame

例2: 音声の第2トラックをaacに変換
--audio-codec 2?aac

例3: aacエンコーダのパラメータ"aac_coder"に低ビットレートでより高品質な"twoloop"を指定
--audio-codec aac:aac_coder=twoloop

例2: 音声デコーダにdual_mono_mode=mainを指定
--audio-codec aac#dual_mono_mode=main
```

### --audio-bitrate [&lt;int&gt;?]&lt;int&gt;
音声をエンコードする際のビットレートをkbpsで指定する。

[&lt;int&gt;]で、抽出する音声トラック(1,2,...)を指定することもできる。
```
例1: --audio-bitrate 192   (音声を192kbpsで変換)
例2: --audio-bitrate 2?256 (音声の第2トラックを256kbpsで変換)
```

### --audio-profile [&lt;int&gt;?]&lt;string&gt;
音声をエンコードする際、そのプロファイルを指定する。

### --audio-stream [&lt;int&gt;?][&lt;string1&gt;][:&lt;string2&gt;]
音声チャンネルの分離・統合などを行う。
--audio-streamが指定された音声トラックは常にエンコードされる。(コピー不可)
,(カンマ)で区切ることで、入力の同じトラックから複数のトラックを生成できる。

##### 書式
&lt;int&gt;に処理対象のトラックを指定する。

&lt;string1&gt;に入力として使用するチャンネルを指定する。省略された場合は入力の全チャンネルを使用する。

&lt;string2&gt;に出力チャンネル形式を指定する。省略された場合は、&lt;string1&gt;のチャンネルをすべて使用する。

```
例1: --audio-stream FR,FL
デュアルモノから左右のチャンネルを2つのモノラル音声に分離する。

例2: --audio-stream :stereo
どんな音声もステレオに変換する。

例3: --audio-stream 2?5.1,5.1:stereo
入力ファイルの第２トラックを、5.1chの音声を5.1chとしてエンコードしつつ、ステレオにダウンミックスしたトラックを生成する。
実際に使うことがあるかは微妙だが、書式の紹介例としてはわかりやすいかと。
```

##### 使用できる記号
```
mono       = FC
stereo     = FL + FR
2.1        = FL + FR + LFE
3.0        = FL + FR + FC
3.0(back)  = FL + FR + BC
3.1        = FL + FR + FC + LFE
4.0        = FL + FR
4.0        = FL + FR + FC + BC
quad       = FL + FR + BL + BR
quad(side) = FL + FR + SL + SR
5.0        = FL + FR + FC + SL + SR
5.1        = FL + FR + FC + LFE + SL + SR
6.0        = FL + FR + FC + BC + SL + SR
6.0(front) = FL + FR + FLC + FRC + SL + SR
hexagonal  = FL + FR + FC + BL + BR + BC
6.1        = FL + FR + FC + LFE + BC + SL + SR
6.1(front) = FL + FR + LFE + FLC + FRC + SL + SR
7.0        = FL + FR + FC + BL + BR + SL + SR
7.0(front) = FL + FR + FC + FLC + FRC + SL + SR
7.1        = FL + FR + FC + LFE + BL + BR + SL + SR
7.1(wide)  = FL + FR + FC + LFE + FLC + FRC + SL + SR
```

### --audio-samplerate [&lt;int&gt;?]&lt;int&gt;
音声のサンプリング周波数をHzで指定する。
[&lt;int&gt;]で、抽出する音声トラック(1,2,...)を指定することもできる。
```
例1: --audio-bitrate 44100   (音声を44100Hzに変換)
例2: --audio-bitrate 2?22050 (音声の第2トラックを22050Hzに変換)
```

### --audio-resampler &lt;string&gt;
音声チャンネルのmixやサンプリング周波数変換に使用されるエンジンの指定。
- swr  ... swresampler (デフォルト)
- soxr ... sox resampler (libsoxr)

### --audio-delay [&lt;int&gt;?]&lt;int&gt;
音声に設定する遅延をms単位で指定する。

### --audio-file [&lt;int&gt;][&lt;string&gt;?]&lt;string&gt;
指定したパスに音声を抽出する。出力フォーマットは出力拡張子から自動的に決定する。avhw/avswリーダー使用時のみ有効。

[&lt;int&gt;]で、抽出する音声トラック(1,2,...)を指定することもできる。
```
例: test_out2.aacにトラック番号#2を抽出
--audio-file 2?"test_out2.aac"
```

[&lt;string&gt;]では、出力フォーマットを指定することができる。
```
例: 拡張子なしでもadtsフォーマットで出力
--audio-file 2?adts:"test_out2"  
```

### --audio-filter [&lt;int&gt;?]&lt;string&gt;
音声に音声フィルタを適用する。適用可能なフィルタは[こちら](https://ffmpeg.org/ffmpeg-filters.html#Audio-Filters)。


[&lt;int&gt;]で、抽出する音声トラック(1,2,...)を指定することもできる。

```
例1: --audio-filter volume=0.2     (音量を下げる例)
例2: --audio-filter 2?volume=-4db  (第2トラックの音量を下げる例)
```

### --audio-ignore-decode-error &lt;int&gt;
指定した連続する音声のデコードエラーの数をカウントし、閾値以内ならエラーを無視して処理を継続し、エラーの箇所は無音に置き換える。

デフォルトは10。 0とすれば、1回でもデコードエラーが起これば処理を中断してエラー終了する。

### --audio-source &lt;string&gt;[:[&lt;int&gt;?][;&lt;param1&gt;=&lt;value1&gt;][;&lt;param2&gt;=&lt;value2&gt;]...][:...]
外部音声ファイルをmuxする。

**パラメータ** 
- copy  
  音声トラックをそのままコピーする。

- codec=&lt;string&gt;  
  音声トラックを指定のコーデックにエンコードする。

- profile=&lt;string&gt;  
  音声エンコード時のプロファイルを指定する。

- bitrate=&lt;int&gt;  
  音声エンコード時のビットレートをkbps単位で指定する。
  
- samplerate=&lt;int&gt;  
  音声エンコード時のサンプリングレートをHz単位で指定する。

- enc_prm=&lt;string&gt;  
  音声エンコード時のパラメータを指定する。

- filter=&lt;string&gt;  
  音声エンコード時のフィルタを指定する。

```
例1: --audio-source "<audio_file>":copy
例2: --audio-source "<audio_file>":codec=aac
例3: --audio-source "<audio_file>":1?codec=aac;bitrate=256:2?codec=aac;bitrate=192
```

### --chapter &lt;string&gt;
指定したチャプターファイルを読み込み反映させる。
nero形式、apple形式、matroska形式に対応する。--chapter-copyとは併用できない。

nero形式  
```
CHAPTER01=00:00:39.706
CHAPTER01NAME=chapter-1
CHAPTER02=00:01:09.703
CHAPTER02NAME=chapter-2
CHAPTER03=00:01:28.288
CHAPTER03NAME=chapter-3
```

apple形式 (UTF-8であること)  
```
<?xml version="1.0" encoding="UTF-8" ?>
  <TextStream version="1.1">
   <TextStreamHeader>
    <TextSampleDescription>
    </TextSampleDescription>
  </TextStreamHeader>
  <TextSample sampleTime="00:00:39.706">chapter-1</TextSample>
  <TextSample sampleTime="00:01:09.703">chapter-2</TextSample>
  <TextSample sampleTime="00:01:28.288">chapter-3</TextSample>
  <TextSample sampleTime="00:01:28.289" text="" />
</TextStream>
```

matroska形式 (UTF-8であること)  
[その他のサンプル&gt;&gt;](https://github.com/nmaier/mkvtoolnix/blob/master/examples/example-chapters-1.xml)
```
<?xml version="1.0" encoding="UTF-8"?>
<Chapters>
  <EditionEntry>
    <ChapterAtom>
      <ChapterTimeStart>00:00:00.000</ChapterTimeStart>
      <ChapterDisplay>
        <ChapterString>chapter-0</ChapterString>
      </ChapterDisplay>
    </ChapterAtom>
    <ChapterAtom>
      <ChapterTimeStart>00:00:39.706</ChapterTimeStart>
      <ChapterDisplay>
        <ChapterString>chapter-1</ChapterString>
      </ChapterDisplay>
    </ChapterAtom>
    <ChapterAtom>
      <ChapterTimeStart>00:01:09.703</ChapterTimeStart>
      <ChapterDisplay>
        <ChapterString>chapter-2</ChapterString>
      </ChapterDisplay>
    </ChapterAtom>
    <ChapterAtom>
      <ChapterTimeStart>00:01:28.288</ChapterTimeStart>
      <ChapterTimeEnd>00:01:28.289</ChapterTimeEnd>
      <ChapterDisplay>
        <ChapterString>chapter-3</ChapterString>
      </ChapterDisplay>
    </ChapterAtom>
  </EditionEntry>
</Chapters>
```

### --chapter-copy
チャプターをコピーする。

### --chapter-no-trim
チャプター読み込みの際、trimを反映させず、そのまま適用する。

### --sub-source &lt;string&gt;
指定のファイルから字幕を読み込みmuxする。

### --sub-copy [&lt;int&gt;[,&lt;int&gt;]...]
字幕をコピーする。avhw/avswリーダー使用時のみ有効。
[&lt;int&gt;[,&lt;int&gt;]...]で、抽出する字幕トラック(1,2,...)を指定することもできる。

対応する字幕は、PGS/srt/txt/ttxtなど。

```
例: 字幕トラック #1と#2をコピー
--sub-copy 1,2
```

### --sub-codec [&lt;int&gt;?]&lt;string&gt;
字幕トラックを指定のコーデックにエンコードする。

### --caption2ass &lt;string&gt;
caption2assによる字幕抽出処理を行い、動画にmuxして出力する。別途 "Caption.dll" が必要。

mp4にmuxする際は、必ずsrt形式を選択してください。内部でさらにmov_textに変換してmuxしますが、ass形式を選択するとmp4へのmuxがうまく動作しません。

**出力フォーマット**
- srt (デフォルト)
- ass

### --data-copy [&lt;int&gt;[,&lt;int&gt;]...]
データストリームをコピーする。avhw/avswリーダー使用時のみ有効。

### --input-option &lt;string1&gt;:&lt;string2&gt;
avsw/avhwでの読み込み時にオプションパラメータを渡す。&lt;string1&gt;にオプション名、&lt;string2&gt;にオプションの値を指定する。

```
Example: Blurayのplaylist 1を読み込み
-i bluray:D:\ --input-option palylist:1
```

### -m, --mux-option &lt;string1&gt;:&lt;string2&gt;
mux時にオプションパラメータを渡す。&lt;string1&gt;にオプション名、&lt;string2&gt;にオプションの値を指定する。

```
例: HLS用の出力
-i <input> -o test.m3u8 -f hls -m hls_time:5 -m hls_segment_filename:test_%03d.ts --gop-len 30
```

### --avsync &lt;string&gt;
  - cfr (default)  
    入力はCFRを仮定し、入力ptsをチェックしない。

  - forcecfr  
    入力ptsを見ながら、CFRに合うようフレームの水増し・間引きを行い、音声との同期が維持できるようにする。主に、入力がvfrやRFFなどのときに音ズレしてしまう問題への対策。

  - vfr  
    入力に従い、フレームのタイムスタンプをそのまま引き渡す。avsw/avhwリーダによる読み込みの時のみ使用可能。また、--trimとは併用できない。

## vppオプション


### --vpp-deinterlace &lt;string&gt;
GPUによるインタレ解除を使用する。"normal", "bob"はわりときれいに解除されるが、"it"はあまりきれいに解除できない。

- none   ... インタレ解除を行わない
- normal ... 標準的な60i→30pインタレ解除。
- bob    ... 60i→60pインタレ解除。
- it     ... inverse telecine

### --vpp-denoise &lt;int&gt;
GPUによるノイズ除去を行う。0 - 100 の間でノイズ除去の強さを指定する。

### --vpp-mctf [ "auto" or &lt;int&gt; ]
動き補償付き時間軸ノイズ除去を行う。引数を省略した場合、あるいは"auto"を指定した場合は、
自動的にノイズ除去の強さが調整される。また、1(弱) - 20(強) の間でノイズ除去の強さを指定することもできる。

### --vpp-detail-enhance &lt;int&gt;
GPUによるディテールの強調を行う。0 - 100 の間でディテール強調の強さを指定する。

### --vpp-image-stab &lt;string&gt;
image stabilizerのモードの指定。
- none
- upscale
- box

### --vpp-rotate &lt;int&gt;
映像を指定した角度で回転させる。90°, 180°, 270° から選択。動作にはd3d11モードであることが必要。

### --vpp-mirror &lt;string&gt;
映像を胸像反転させる。
- h ... 水平方向の反転。
- v ... 垂直方向の反転。

### --vpp-half-turn &lt;string&gt;
非常に遅く、実験用。

### --vpp-resize &lt;string&gt;
リサイズのアルゴリズムを指定する。

| オプション名 | 説明 |
|:---|:---|
| auto  | 自動的に適切なものを選択 |
| simple | HWによるシンプルなリサイズ |
| fine | 高品質なリサイズ |


### --vpp-colorspace [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...  
色空間変換を行う。

**パラメータ**
- matrix=&lt;from&gt;:&lt;to&gt;  
  
```
  bt709, smpte170m
```

- range=&lt;from&gt;:&lt;to&gt;  
```
  limited, full, auto
```


```
例1: BT.709(fullrange) -> BT.601 への変換
--vpp-colorspace matrix=smpte170m:bt709,range=full:limited
```

### --vpp-delogo &lt;string&gt;
ロゴファイルを指定する。".lgd",".ldp",".ldp2"に対応。

### --vpp-delogo-select &lt;string&gt;
ロゴパックの場合に、使用するロゴを以下のいずれかで指定する。

- ロゴ名
- インデックス (1,2,...)
- 自動選択用iniファイル
```
 [LOGO_AUTO_SELECT]
 logo<連番数字>=<マッチパターン>,<リストに表示されているロゴ名(完全一致!)>
 ```

 例:
 ```ini
[LOGO_AUTO_SELECT]
logo1= (NHK-G).,NHK総合 1440x1080
logo2= (NHK-E).,NHK-E 1440x1080
logo3= (MX).,TOKYO MX 1 1440x1080
logo4= (CTC).,チバテレビ 1440x1080
logo5= (NTV).,日本テレビ 1440x1080
logo6= (TBS).,TBS 1440x1088
logo7= (TX).,TV東京 50th 1440x1080
logo8= (CX).,フジテレビ 1440x1088
logo9= (BSP).,NHK BSP v3 1920x1080
logo10= (BS4).,BS日テレ 1920x1080
logo11= (BSA).,BS朝日 1920x1080
logo12= (BS-TBS).,BS-TBS 1920x1080
logo13= (BSJ).,BS Japan 1920x1080
logo14= (BS11).,BS11 1920x1080 v3
```

### --vpp-delogo-pos &lt;int&gt;:&lt;int&gt;
1/4画素精度のロゴ位置の調整。Aviutlで言うところの &lt;位置 X&gt;:&lt;位置 Y&gt;。

### --vpp-delogo-depth &lt;int&gt;
ロゴの透明度の補正。デフォルト128。Aviutlで言うところの &lt;深度&gt;。

### --vpp-delogo-y  &lt;int&gt;
### --vpp-delogo-cb &lt;int&gt;
### --vpp-delogo-cr &lt;int&gt;
ロゴの各色成分の補正。Aviutlで言うところの &lt;Y&gt;, &lt;Cb&gt;, &lt;Cr&gt;。


## 制御系のオプション

### -a, --async-depth &lt;int&gt;
QSVパイプラインに先行投入するフレーム数を指定する。

QSVのパイプライン(Decode, VPP, Encode)に指定量のフレームを余剰に投入する。これによりパイプラインの並列動作を容易にし、QSV/GPUの稼働率を向上させ、処理が高速化する。デフォルトでは自動で決定され、4 + 追加のパイプライン段数×2となる。(たとえば、エンコードのみなら4、エンコードとデコードなら6...)
多くすると高速化する可能性もあるが、メモリ使用量が増えるほか、キャッシュ効率が悪くなり、遅くなる可能性もある。

### --output-buf &lt;int&gt;
出力バッファサイズをMB単位で指定する。デフォルトは8、最大値は128。0で使用しない。

出力バッファはディスクへの書き込みをアプリケーション側で制御し、バッファサイズ分たまるまで書き出さないようにする。
これにより、ディスクへの書き込み単位が大きくなり、エンコード中のディスクへの読み書きの混在を防ぎ、高速化が期待できる。
またディスク上でのファイルの断片化の抑止も期待できる。

一方、あまり大きく設定しすぎると、逆に遅くなることがあるので注意。基本的にはデフォルトのままで良いと思われる。

file以外のプロトコルを使用する場合には、この出力バッファは使用されず、この設定は反映されない。
また、出力バッファ用のメモリは縮退確保するので、必ず指定した分確保されるとは限らない。

### --mfx-thread &lt;int&gt;
QSVパイプライン駆動用のスレッド数を2以上の値から指定する。(デフォルト: -1 ( = 自動))

### --output-thread &lt;int&gt;
出力用のスレッドを使用するかどうかを指定する。
- -1 ... 自動(デフォルト)
-  0 ... 使用しない
-  1 ... 使用する  
出力スレッドを使用すると、メモリ使用量が増加するが、エンコード速度が向上する場合がある。

### --min-memory
QSVEncCの使用メモリ量を最小化する。下記オプションに同じ。
```
--output-thread 0 --audio-thread 0 --mfx-thread 2 -a 1 --input-buf 1 --output-buf 
```

### --(no-)timer-period-tuning
Windowsのタイマー精度を向上させ、高速化する。いわゆるtimeBeginPeriod(1)。Windowsのみ。

### --log &lt;string&gt;
ログを指定したファイルに出力する。

### --log-level &lt;string&gt;
ログ出力の段階を選択する。不具合などあった場合には、--log-level debug --log log.txtのようにしてデバッグ用情報を出力したものをコメントなどで教えていただけると、不具合の原因が明確になる場合があります。
- error ... エラーのみ表示
- warn ... エラーと警告を表示
- info ... 一般的なエンコード情報を表示、デフォルト
- debug ... デバッグ情報を追加で出力
- trace ... フレームごとに情報を出力

### --benchmark &lt;string&gt;
ベンチマークモードを実行し、結果を指定されたファイルに出力する。

### --bench-quality "all" or <int>[,<int>][,<int>]...
ベンチマークの対象とする"--quality"のリスト。デフォルトは"best,balanced,fastest"。"all"とすると7種類のすべての品質設定についてベンチマークを行う。

### --max-procfps &lt;int&gt;
エンコード速度の上限を設定。デフォルトは0 ( = 無制限)。
複数本QSVEncでエンコードをしていて、ひとつのストリームにCPU/GPUの全力を奪われたくないというときのためのオプション。
```
例: 最大速度を90fpsに制限
--max-procfps 90
```

### --lowlatency
エンコード遅延を低減するモード。最大エンコード速度(スループット)は低下するので、通常は不要。

### --perf-monitor [&lt;string&gt;][,&lt;string&gt;]...
エンコーダのパフォーマンス情報を出力する。パラメータとして出力したい情報名を下記から選択できる。デフォルトはall (すべての情報)。

```
 all          ... monitor all info
 cpu_total    ... cpu total usage (%)
 cpu_kernel   ... cpu kernel usage (%)
 cpu_main     ... cpu main thread usage (%)
 cpu_enc      ... cpu encode thread usage (%)
 cpu_in       ... cpu input thread usage (%)
 cpu_out      ... cpu output thread usage (%)
 cpu_aud_proc ... cpu aud proc thread usage (%)
 cpu_aud_enc  ... cpu aud enc thread usage (%)
 cpu          ... monitor all cpu info
 gpu_load    ... gpu usage (%)
 gpu_clock   ... gpu avg clock
 vee_load    ... gpu video encoder usage (%)
 gpu         ... monitor all gpu info
 queue       ... queue usage
 mem_private ... private memory (MB)
 mem_virtual ... virtual memory (MB)
 mem         ... monitor all memory info
 io_read     ... io read  (MB/s)
 io_write    ... io write (MB/s)
 io          ... monitor all io info
 fps         ... encode speed (fps)
 fps_avg     ... encode avg. speed (fps)
 bitrate     ... encode bitrate (kbps)
 bitrate_avg ... encode avg. bitrate (kbps)
 frame_out   ... written_frames
```

### --perf-monitor-interval &lt;int&gt;
[--perf-monitor](#--perf-monitor-stringstring)でパフォーマンス測定を行う時間間隔をms単位で指定する(50以上)。デフォルトは 500。