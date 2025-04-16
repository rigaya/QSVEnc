---------------------------------------------------


	QSVEnc + QSVEncC
	 by rigaya

---------------------------------------------------

QSVEnc は、
Intel Media SDK を使用してエンコードを行うAviutlの出力プラグインです。
IntelMediaSDKのsample_encode.exeを改造し、x264guiEx 1.xxに突っ込みました。
QuickSyncVideoによるハードウェア高速エンコードを目指します。

QSVEncC は、上記のコマンドライン版です。
コマンドラインオプションについては、下記urlを参照ください。
https://github.com/rigaya/QSVEnc/blob/master/QSVEncC_Options.ja.md


【基本動作環境】
Windows 10/11 (x86/x64) (QSVEnc.auo / QSVEncC.exe x86版)
Windows 10/11 (x64) (QSVEncC.exe x64版)
Aviutl 1.00 以降 (QSVEnc.auo)

【ハードウェアエンコード動作環境】
上記「基本動作環境」に加え、
QuickSyncVideo回路の有効なCPUと対応したマザーボード
などですが、他にも色々条件があるかもなので頑張ってください。
注意点としては
・WinXP非対応
・Windows Home Server とかもダメっぽいです。(持ってないのでわからんけども)
  WHS用のIntelグラフィックスドライバが対応してない…模様。


【QSVEnc 使用にあたっての注意事項】
無保証です。自己責任で使用してください。
QSVEncを使用したことによる、いかなる損害・トラブルについても責任を負いません。
バグがたくさんあり、よく止まります。ご了承下さい。

【QSVEnc 再配布(二次配布)について】
このファイル(QSVEnc_readme.txt)とIntel Media SDK EULA.rtfと一緒に配布してください。念のため。
まあできればアーカイブまるごとで。

【導入方法】
※ 下記リンク先では図も使用して説明していますので、よりわかりやすいかもしれません。
   https://github.com/rigaya/QSVEnc/blob/master/QSVEnc_auo_readme.md#QSVEnc-の-aviutl-への導入更新

1.
ダウンロードしたAviutl_QSVEnc_7.xx.zipを開きます。

2.
zipファイル内のフォルダすべてをAviutlフォルダにコピーします。

3.
Aviutlを起動します。

4.
環境によっては、ウィンドウが表示され必要なモジュールのインストールが行われます。
その際、この不明な発行元のアプリがデバイスに変更を加えることを許可しますか? と出ることがありますが、
「はい」を選択してください。

5.
「その他」>「出力プラグイン情報」にQSVEnc 7.xxがあるか確かめます。
ここでQSVEncの表示がない場合、
- zipファイル内のフォルダすべてをコピーできていない
- 必要なモジュールのインストールに失敗した
  - この不明な発行元のアプリがデバイスに変更を加えることを許可しますか? で 「はい」を選択しなかった
  - (まれなケース) ウイルス対策ソフトにより、必要な実行ファイルが削除された
などの原因が考えられます。


【削除方法】
※ 下記リンク先では図も使用して説明していますので、よりわかりやすいかもしれません。
   https://github.com/rigaya/QSVEnc/blob/master/QSVEnc_auo_readme.md#QSVEnc-の-aviutl-からの削除

・Aviutlのpulginsフォルダ内から下記フォルダとファイルを削除してください。
  - [フォルダ] QSVEnc_stg
  - [ファイル] QSVEnc.auo
  - [ファイル] QSVEnc.conf (存在する場合)
  - [ファイル] QSVEnc(.ini)
  - [ファイル] auo_setup.auf

【QSVEncによるエンコードの注意点】
・基本的にQSVEncCのx86版のほうがx64版より高速です。

【使用出来る主な機能】
 [QSVEnc/QSVEncC共通]
・QuickSyncVideoを使用したエンコード
   - H.264/AVC
   - H.265/HEVC
   - MPEG2
・QuickSyncVideoの各エンコードモード
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
・エンコード品質の7段階による指定
・インタレ保持エンコード (PAFF方式)
・シーンチェンジ検出
・SAR比指定
・colormatrix等の指定
・H.264 Level / Profileの指定
・Blurayエンコードモード
・最大ビットレート等の指定
・最大GOP長の指定
・AVX2までを使用した処理の高速化
・Vpp機能 -- GPUを使用した高速フィルタリング
   - リサイズ
   - インタレ解除 (normal / bob / it)
   - エッジ強調
   - ノイズ低減
   - Image Stablizier
[QSVEnc]
・音声エンコード
・音声及びチャプターとのmux機能
・自動フィールドシフト対応
[QSVEncC]
・QSVデコードに対応。
・QSVによるデコード/VPP/エンコードをそれぞれ自由な組み合わせで使用可能
・エンコードなしの出力も可能。
・avi(vfw), avs, vpy, y4m, rawなど各種形式に対応
・libavcodec/libavformatを利用した音声処理に対応
・libavcodec/libavformatを利用し、muxしながら出力が可能

ほか。

【iniファイルによる拡張】
QSVEnc.iniを書き換えることにより、
音声エンコーダやmuxerのコマンドラインを変更できます。
また音声エンコーダを追加することもできます。

デフォルトの設定では不十分だと思った場合は、
iniファイルの音声やmuxerのコマンドラインを調整してみてください。

【コンパイル環境】
VC++ 2022 Community

【dtlの使用表記】
本プログラムは、dtlライブラリを内部で使用しています。
https://github.com/cubicdaiya/dtl

【tinyxml2の使用表記】
本プログラムは、tinyxml2を内部で使用しています。
http://www.grinninglizard.com/tinyxml2/index.html

【検証環境 ～2011.09.30】
Win7 x64
Core i5 2500 + H61
4GB RAM
Intel Graphics Driver 8.15.10.2376
Intel Graphics Driver 8.15.10.2509

【検証環境 2011.10.01～】
Win7 x64
Core i5 2500 + Z68
GeForce 8600GT as Primary GPU
4GB RAM
Intel Graphics Driver 8.15.10.2509

【検証環境 2012.01～】
Win7 x64
Core i5 2500 + Z68
4GB RAM
Intel Graphics Driver 8.15.10.2509
Intel Graphics Driver 8.15.10.2559 (API v1.1)
Intel Graphics Driver 8.15.10.2626 (API v1.3)
Intel Graphics Driver 8.15.10.2653 (API v1.3)

【検証環境 2012.05～】
Win7 x64
Core i7 3770K + ASUS P8Z77-M
16GB RAM
Intel Graphics Driver 8.15.10.2696 (API v1.3)
Intel Graphics Driver 8.15.10.2729 (API v1.4)
Intel Graphics Driver 8.15.10.2753 (API v1.4)
Intel Graphics Driver 8.15.10.2761 (API v1.4)
Intel Graphics Driver 8.15.10.2867 (API v1.4)
Intel Graphics Driver 9.17.10.2932 (API v1.4)
Intel Graphics Driver 9.17.10.3071 (API v1.4)

【検証環境 2013.06～】
Win7 x64
Core i7 4770K + Asrock Z87 Extreme4
16GB RAM
Intel Graphics Driver  9.18.10.3165 (API v1.6)
Intel Graphics Driver  9.18.10.3187 (API v1.6)
Intel Graphics Driver  9.18.10.3220 (API v1.7)
Intel Graphics Driver  9.18.10.3257 (API v1.7)
Intel Graphics Driver 10.18.10.3336 (API v1.7)
Intel Graphics Driver 10.18.10.3354 (API v1.7)
Intel Graphics Driver 10.18.10.3412 (API v1.7)
Intel Graphics Driver 10.18.10.3464 (API v1.8)
Intel Graphics Driver 10.18.10.3496 (API v1.8)
Intel Graphics Driver 10.18.10.3612 (API v1.10)

【検証環境 2014.10～】
Win8.1 x64
Core i7 4770K + ASUS Z87 Gryphon
16GB RAM
Intel Graphics Driver 10.18.10.3907 (API v1.10)
Intel Graphics Driver 10.18.10.3960 (API v1.11)
Intel Graphics Driver 10.18.14.4080 (API v1.13)
Intel Graphics Driver 10.18.14.4156 (API v1.14)
Intel Graphics Driver 10.18.14.4170 (API v1.14)
Intel Graphics Driver 10.18.14.4222 (API v1.15)
Intel Graphics Driver 10.18.14.4235 (API v1.15)

【検証環境 2015.08～】
Win8.1 x64
Core i7 6700K + Asrock Z170 Extreme7+
16GB RAM
Intel Graphics Driver 10.18.15.4256 (API v1.16)
Intel Graphics Driver 10.18.15.4268 (API v1.16)
Intel Graphics Driver 10.18.15.4274 (API v1.16)
Intel Graphics Driver 10.18.15.4279 (API v1.16)
Intel Graphics Driver 10.18.15.4294 (API v1.16)
Intel Graphics Driver 20.19.15.4300 (API v1.16)
Intel Graphics Driver 20.19.15.4312 (API v1.17)
Intel Graphics Driver 20.19.15.4326 (API v1.17)
Intel Graphics Driver 20.19.15.4352 (API v1.17)
Intel Graphics Driver 20.19.15.4380 (API v1.17)
Intel Graphics Driver 20.19.15.4444 (API v1.17)
Intel Graphics Driver 20.19.15.4454 (API v1.17)
Intel Graphics Driver 20.19.15.4463 (API v1.19)
Intel Graphics Driver 21.20.16.4471 (API v1.19)

【検証環境 2016.07～】
Win10 x64
Core i7 6700K + ASUS Z170-PRO
16GB RAM
Intel Graphics Driver 20.19.15.4463 (API v1.19)

【検証環境 2017.01～】
Win10 x64
Core i7 7700K + Asrock Z270 Extreme4
16GB RAM
Intel Graphics Driver 21.20.16.4534 (API v1.19)


【検証環境 2017.05～】
Win8.1 x64
Core i3 4170 + Asrock Z97 Gryphon
16GB RAM
Intel Graphics Driver 4624 (API v1.19)

【検証環境 2017.05～】
Win10 x64
Core i7 7700K + Asrock Z270 Gaming-ITX/ac
16GB RAM
Intel Graphics Driver 21.20.16.4534 (API v1.19)
Intel Graphics Driver 21.20.16.4678 (API v1.23)
Intel Graphics Driver 21.20.16.4901 (API v1.25)
Intel Graphics Driver 21.20.16.4982 (API v1.25)
Intel Graphics Driver 23.20.16.5018 (API v1.26)
Intel Graphics Driver 25.20.100.6326 (API v1.27)
Intel Graphics Driver 25.20.100.6373 (API v1.27)
Intel Graphics Driver 25.20.100.7000 (API v1.27)
Intel Graphics Driver 25.20.100.7327 (API v1.30)
Intel Graphics Driver 27.20.100.8190 (API v1.32)
Intel Graphics Driver 27.20.100.8681 (API v1.33)

【検証環境 2021.04～】
Win10 x64
Core i7 11700K + GIGABYTE Z590I AORUS ULTRA
16GB RAM
Intel Graphics Driver 27.20.100.9466 (API v1.34)
Intel Graphics Driver 30.0.100.9809  (API v2.03)
Intel Graphics Driver 30.0.100.9955  (API v2.05)

【検証環境 2021.11～】
Win11 x64
Core i9 12900K + MSI Z690 Tomahawk DDR4 WIFI
32GB RAM
Intel Graphics Driver 30.0.100.1002  (API v2.05)
Intel Graphics Driver 30.0.101.1191  (API v2.06)

【検証環境 2022.08～】
Win11 x64
Core i9 12900K + MSI Z690 Tomahawk DDR4 WIFI
Arc A380
32GB RAM
Intel Graphics Driver 30.0.101.3221  (API v2.07)
Intel Graphics Driver 30.0.101.3259  (API v2.07)
Intel Graphics Driver 31.0.101.3276  (API v2.07)
Intel Graphics Driver 31.0.101.3430  (API v2.07)
Intel Graphics Driver 31.0.101.3491  (API v2.07)

【Intel Media SDKとAPIの対応関係】
API v1.32 … Intel Media SDK 2020 R1
API v1.29 … Intel Media SDK 2019 R1
API v1.27 … Intel Media SDK 2018 R2
API v1.26 … Intel Media SDK 2018 R1
API v1.23 … Intel Media SDK 2017 R1
API v1.19 … Intel Media SDK 2016 Update 2
API v1.17 … Intel Media SDK 2016
API v1.16 … Intel Media SDK 2015 Update 2.1
API v1.15 … Intel Media SDK 2015 Update 2
API v1.13 … Intel Media SDK 2015 Update 1
API v1.11 … Intel Media SDK 2015
API v1.9  … Intel Media SDK 2014 R2 for Client 44
API v1.8  … Intel Media SDK 2014
API v1.7  … Intel Media SDK 2013 R2
API v1.6  … Intel Media SDK 2013
API v1.4  … Intel Media SDK 2012 R3
API v1.4  … Intel Media SDK 2012 R2
API v1.3  … Intel Media SDK 2012 (Intel Media SDK v3.0)
API v1.1  … Intel Media SDK v2.0


【どうでもいいメモ】
2025.04.16 (7.86)
- 無制限にOpenCLのビルドがスレッドを立てるのを防ぐため、スレッドプールで実行するように。
- AV1のVBVバッファサイズの表示を改善。
- --parallelの安定性向上。
- --log-levelにGPUの自動選択の状況について表示する gpu_select を追加。
- --vpp-libplacebo-tonemappingのtonemapping_functionで、st2094-10とst2094-40が指定できなかった問題を修正。

2025.04.03 (7.85)
- 字幕やデータトラップがある際に、mux時に先頭付近での映像と音声の混ざり具合を改善。
- "failed to run h264_mp4toannexb bitstream filter" というエラーが出るとフリーズしてしまう問題を修正。
  きちんとエラー終了するように。
- 入力色フォーマットとして、uyvyに対応。
- 並列数が1以下ならparallelを自動で無効化するように。

2025.03.22 (7.84)
- ファイル分割並列エンコード機能を追加。 (--parallel)
- 言語コード指定時に、ISO 639-2のTコードでの指定も可能なように拡張。
- 入力ファイルによっては、まれに--seekを使用するとtimestampが不適切になってしまっていたのを修正。
- --qp-min, --qp-max片方のみの指定だと適切に設定されない問題を修正。
- 不必要なdovi rpuの変換をしないように。
- vpp-deinterlace bob使用時にRFFなどで最初のフレームがプログレッシブだとpts=0のフレームが2枚生成されエラー終了してしまう問題を修正。
- libmfx1をUbuntu 24.04 debの依存ライブラリとして追加。

2025.03.04 (7.83)
- AV1エンコードで--dolby-vision-rpu使用時の処理を改善。

2025.03.01 (7.82)
- 7.80からavswで一部のコーデックがデコードできなくなっていた問題を修正。
- --dolby-vision-profileに10.0, 10.1, 10.2, 10.4の選択肢を追加。
- --dolby-vision-profileがavhw/avsw以外で動作しなかった問題を修正。
- Linux環境でIntel GPUが複数ある場合のhw検出を改善。

2025.02.18 (7.81)
- 7.80でavswが動作しなくなっていたのを修正。

2025.02.17 (7.80)
- mp4/mkv/ts出力等でchromaloc指定が意図した値にならないことがある問題を修正。
- dolby vision profile 4への処理を改善。

2025.01.23 (7.79)
- AACを--audio-copyしてmkv出力すると、音声が再生できないファイルができることがある問題を修正。

2025.01.08 (7.78)
- SAR比が設定されていない(例えば0:0)と、mp4 muxerの出力する"tkhd: Track Header Box"(L-SMASH boxdumper)、"Visual Track layout"(mp4box -info)のwidthが0になってしまう問題を回避。

2025.01.06 (7.77)
- --vpp-libplacebo-tonemappingで一部のパラメータが正常に動作しない問題を修正。
- tsファイルなどで途中からエンコードする場合に、OpenGOPが使用されているとtrim位置がずれてしまう問題を修正。
  trim補正量の計算時にOpenGOPで最初のキーフレーム後にその前のフレームが来るケースを考慮できていなかった。
- --trimでAviutlのtrimエクスポートプラグインの表記を受け取れるように。

2025.01.03 (7.76)
- 7.75の"--dolby-vision-rpuをファイルから読む場合に壊してしまっていたのを修正。"がまだ修正できていなかったのを再修正。
- エンコーダのPCを起動してから2回目以降の起動速度を向上。
- --device autoでのGPU自動選択を改善。
  より積極的に他のGPUを使用するように。

2024.11.24 (7.75)
- --dolby-vision-rpuをファイルから読む場合に壊してしまっていたのを修正。
- --vpp-libplacebo-debandのgrain_y, grain_cの読み取りが行われない問題を修正。
- --vpp-libplacebo-debandのgrain_cのヘルプを修正。
- --dolby-vision-rpuと--dhdr10-infoの併用に対応。

2024.11.22 (7.74)
- --dolby-vision-profileで対象外のプロファイルも読み込めていた問題を修正。
- --dolby-vision-rpu使用時にレターボックス部分をcropをした場合にそれを反映させるオプションを追加。 ( --dolby-vision-rpu-prm crop )
- --dolby-visionに関するモード制限を解除。
- ログ表示の細かな変更等。

2024.11.12 (7.73)
- --dolby-vision-rpu copyを使用して長時間のエンコードを行うと、エンコード速度が著しく低下していくのを改善し、速度を維持し続けられるように。
- AV1出力時に--dhdr10-infoを使用した時の出力を改善。
- 入力ファイルの字幕のタイムスタンプが入れ違いになっている場合にエラーが発生していたのを、ソートしなおして解決するように変更。
- --vpp-tweakをなにもしない設定で実行した時クラッシュするのを回避。

2024.11.02 (7.72)
[QSVEncC]
- --dhdr10-infoの実装を変更し、Linuxでの動作に対応。
  hdr10plus_gen.exeを使用する代わりにlibhdr10plusを使用するように変更。
- 入力ファイルにdoviがない場合に、--dolby-vision-rpuを指定するとエラー終了する問題を修正。
- --dhdr10-infoがraw出力時に動作しなくなっていたのを修正。
- 7.71で入力ファイルのSAR比が反映されていなかった問題を修正。

2024.10.27 (7.71)
[QSVEncC]
- libplaceboによるバンディング低減フィルタを追加。(--vpp-libplacebo-deband)
- libplaceboによるtone mappingフィルタを追加。(--vpp-libplacebo-tonemapping)
- libplaceboのcustom shaderを使用したフィルタを追加。 (--vpp-libplacebo-shader)
- --dolby-vision-rpu copy使用時に、入力ファイルのdolby vision profile 7のとき、
  libdoviを使用して自動的にdolby vision profile 8に変換するように。 
- --dhdr10-infoが動作しなくなっていたのを修正。

2024.09.24 (7.70)
[QSVEncC]
- libvplを更新し、2.13に対応。
- libplaceboによるリサイズフィルタを追加。(Windows x64版)
- 使用するffmpegのライブラリを更新。(Windows版)
  - ffmpeg     7.0    -> 20240902
  - dav1d      1.4.1  -> 1.4.3
  - libvpl     2.11.0 -> 2.12.0
  - libvpx     2.14.0
  - MMT/TLV対応パッチを組み込み (mmtsを読み込み可能に)
- --vpp-smoothの誤ったhelpを修正。

[QSVEnc.auo]
- 6時間を超える長さのwav出力に対応。
- ffmpeg系の音声出力で6時間を超える音声の出力に対応。
- ffmpegのlibopusを使用したopusエンコードを追加。
- VC runtimeインストーラを更新。

2024.08.20 (7.69)
- RGB出力機能を追加。(--output-csp rgb)
- Dolby Vision profileのコピー機能を追加。(--dolby-vision-profile copy)
- Dolby Vision rpu metadataのコピー機能を追加。(--dolby-vision-rpu copy)
- H.264/HEVCのヘッダがうまく取得できない場合、最初のパケットから取得するように。( #196 )
- mkvに入っているAV1がそのままのヘッダだと、Decodeに失敗する場合があるのを修正。
- 音声のmux用のバッファ不足になり、音声が同時刻の映像と違うfragmentにmuxされる問題を修正。
- --vpp-transformでフレームサイズが64で割り切れない場合に、不正なメモリアクセスが生じる問題を修正。

2024.07.04 (7.68)
[QSVEncC]
- oneVPLベースからlibvplベースに変更。
- Ubuntu 24.04用のパッケージを追加。
- API 2.11に対応。
- API 2.11で追加されたAI Super Resolutionを追加。(--vpp-resize mfx-ai-superres)
  が、まだドライバは未対応の模様。
- --vpp-tweakにチャネルごとの制御を追加。

[QSVEnc.auo]
- Windowsの登録拡張子の状況によっては、意図せず出力拡張子が設定されず、muxされなくなってしまう問題を回避。

2024.06.29 (7.67)
[QSVEncC]
- RGBなaviファイルからエンコードすると、フレームの上下が入れ替わってしまうことがある問題を修正。

[QSVEnc.auo]
- Nsyw様に提供いただき、中国語翻訳を更新。
- 拡張編集使用時に映像と音声の長さが異なる場合には、警告を出して一時中断し、処理を継続するか判断してもらうよう変更。

2024.06.08 (7.66)
- 新たなノイズ除去フィルタを追加。(--vpp-fft3d)

2024.05.24 (7.65)
- 新たなインタレ解除フィルタを追加。(--vpp-decomb)

2024.05.12 (7.64)
- ffmpeg 7.0に更新。(Windows版)
  - ffmpeg     6.1    -> 7.0
  - libpng     1.4.0  -> 1.4.3
  - expat      2.5.0  -> 2.6.2
  - opus       1.4    -> 1.5.2
  - libxml2    2.12.0 -> 2.12.6
  - dav1d      1.3.0  -> 1.4.1
  - libvpl     2.11.0 (new!)
  - nv-codec-headers 12.2.72.0 (new!)
- avswで使用するデコーダを指定可能に。
- --audio-bitrateの指定がないとき、デフォルトのビットレートを設定するのではなく、コーデックに任せるように。
- --audio-bitrateあるいは--audio-copyで指定のない音声/字幕/データトラックは処理しないように。
- QSVEnc 7.62以降のside_dataの扱いが誤っており、--master-display  copy/--max-cll copyが正常に行われていなかった問題を修正。

2024.04.28 (7.63)
- 新たなノイズ除去フィルタを追加。(--vpp-nlmeans)
- --audio-resamplerを拡張し、文字列でパラメータ設定できるように。
- --output-resにSAR比を無視して計算するオプションを追加。
- subgroupでの同期を使用してvpp-smooth/vpp-denoise-dctをわずかに高速化。

2024.03.17 (7.62)
- HEVCではFFを使用できる場合にはFFを優先して使用するように。
- RGB3→RGB4変換のAVX2版の不具合を修正。
- 10bit出力時に、--vpp-smoothの強度が8bit出力時と一致しなかったのを修正。
- --avsyncのデフォルト値を実態に合わないcfrからautoに変更。
- 音声を品質で指定するオプションを追加。 ( --audio-quality )
- Linux環境で権限不足でデバイスをopenできなかった場合のメッセージを改善。
- 存在しないドライブに出力すると異常終了する問題を修正。

2024.02.20 (7.61)
- Resizable BARが無効の状態でArc GPUでエンコードした際に、GPUクロックが低下し
  エンコードが著しく遅くなる問題を回避。(--avoid-idle-clock)

2024.02.16 (7.60)
- 7.59で、getWorkSurfに失敗することがある問題を修正。
- 7.56以降で--disable-d3dが正常に動作しなかった問題を修正。
- その他コード整理。

2024.02.11 (7.59)
- ノイズ除去フィルタを追加。(--vpp-denoise-dct)
- avhw以外の処理を高速化。
- 各タスクの時間計測機能を追加。(--task-perf-monitor)

2024.01.04 (7.58)
- spline系のresizeが正常に動作しない可能性があったのを修正。

2023.12.08 (7.57)
- ffmpeg 6.1に更新。(Windows版)
  - ffmpeg 5.1 -> 6.1
  - libpng 1.3.9 -> 1.4.0
  - opus 1.3.1 -> 1.4
  - libsndfile 1.2.0 -> 1.2.2
  - libxml2 2.10.3 -> 2.12.0
  - dav1d 1.0.0 -> 1.3.0
  - libaribcaption 1.1.1 (new!)

- --caption2assを廃止。
  --sub-codec ass#sub_type=ass,ass_single_rect=true で同等の処理が可能。

2023.12.03 (7.56)
- --seek使用時の進捗表示を改善。
- --disable-opencl指定時にOpenCL情報を使用してGPU情報を表示しないようにして高速化。
- DX11のチェック時にはこれを指定することでDX9のチェックを抑止し、セッション初期化を高速化。
- --option-files指定時に対象ファイルが空だと異常終了する問題を修正。

2023.11.28 (7.55)
- 初期化のコードにAVX2命令が混入しているらしく、AVX2に対応しない環境で動作しないのを回避。
- セッション初期化を高速化。 (Windows)
- --audio-delayを小数点で渡せるように変更。
- ts出力時にaacをコピーする時の処理を改善。

2023.11.19 (7.54)
- 7.53で追加した--dynamic-rcが正常に動作しない場合があったのを修正。

2023.11.18 (7.53)
- 動的にレート制御モードを変更するオプションを追加。(--dynamic-rc)
- --vpp-afsでrffの考慮をデフォルトで有効に。
- --vpp-yadif/--vpp-nnediでbob化する際、--vpp-rffなしでrffな動画を処理するとtimestamp周りの計算がおかしくなりエラー終了する問題を修正。
- mfxのインタレ解除を使用しているとき、途中でフィールド順が変わったときにGPU busyに陥る問題を回避。
- インタレ解除を指定したが、インタレ設定がされていない場合、自動的に--interlace auto相当の動作にするように。
- エンコードを遅くする大量のメッセージを抑制。
- --check-features/--check-hwのログ出力を拡張。

2023.11.04 (7.52)
[QSVEncC]
- rffを展開するフィルタを追加。(--vpp-rff)
- QSVEnc 7.50で、--b-pyramidがデフォルトで無効になっていたのを元に戻す。

2023.10.28 (7.51)
[QSVEncC]
- ICQモードをデフォルトに。
- --fallback-rcをデフォルトで有効に。

[QSVEnc.auo]
- 外部音声エンコーダを使用すると、muxerの制限でAV1が出力できない問題を修正。
  外部muxerの使用を廃止し、内部muxerを使用するよう変更した。
- AV1時のCQPの上限を255に。
  ただ、基本的には固定品質(ICQ)の使用がおすすめ。

2023.10.18 (7.50)
[QSVEncC]
- 音声フィルターの切り替えがエンコード中に発生する時に--thread-audio > 1で異常終了する問題を修正。
- --log-levelにquietを追加。
- --check-featuresの処理を改良。
- 新しいAVChannelLayout APIに対応(Windows版)。

2023.10.01 (7.49)
[QSVEncC]
- Linuxで--deviceによるデバイス選択に対応。
- --vpp-afs, --vpp-nnedi, --vpp-yadif, --vpp-padのエラーメッセージを拡充。
- --vpp-decimateで最終フレームのタイムスタンプが適切に計算されず、異常終了する問題を修正。

2023.08.08 (7.48)
[QSVEncC]
- --hyper-mode on時には--open-gopを無効に。
- マルチGPU環境での--hyper-mode on指定時の動作を改善。
- --video-tagの指定がない場合、HEVCでは再生互換性改善のため、 "hvc1"をデフォルトとする。
  (libavformatのデフォルトは"hev1")

2023.07.24 (7.47)
[QSVEncC]
- 4575ドライバで--check-featuresが正しく動作しない問題を修正。
- --audio-streamをavs読み込み時にも対応。
- perceptual pre encode filterを有効にするオプションを追加。(--vpp-perc-pre-enc)

2023.06.24 (7.46)
[QSVEncC]
- --vpp-denoiseのauto_xx等が動作するよう修正。
- 入力のcolorprimの指定がない場合に、--vpp-colorspaceのhdr2sdrが正常に動作しない問題を修正。
- --max-framesize-i, --max-framesize-pを追加。

[QSVEnc.auo]
- faw処理時に音声がブツブツ切れる場合があったのを修正。

2023.06.20 (7.45)
[QSVEncC]
--tile-row, --tile-col, --max-framesizeを追加。

[QSVEnc.auo]
- QSVEnc 7.43のfaw処理に問題があり、異常終了してしまうケースがあったのを修正。

2023.06.xx (7.44)
- tsファイルで--audio-copyを使用すると、"AAC bitstream not in ADTS format and extradata missing"というエラーが出るのを回避。

2023.06.04 (7.43)
[QSVEncC]
- HEVCエンコードでcolormatrix等の指定時の出力を改善。
- 複数のデバイスからGPUを選択する際、OpenCLの初期化に失敗したデバイスの優先度を落とすように。

[QSVEnc.auo]
- faw2aac.auo/fawcl.exeがなくても内蔵機能でfawを処理できるように。

2023.06.02 (7.42)
- OpenCLが正常にインストールされていない環境でエラー回避するように。
- ログ出力の調整機能を拡張。

2023.05.18 (7.41)
- 4369ドライバでエンコードできない問題を修正。

2023.05.14 (7.40)
[QSVEncC]
- dshowのカメラ入力等に対応。
- --audio-source, --sub-sourceのコマンドラインの区切りを変更。
- --vpp-colorspaceの調整。

2023.05.10 (7.39)
[QSVEncC]
- OpenCLの初期化が失敗した場合でも処理を続行するよう変更。
- RGY_PRIM_ST432_1についての定義を追加。
- 音声処理をトラックごとに並列化。

2023.05.02 (7.38)
[QSVEncC]
- --audio-source/--sub-sourceでファイルのフォーマット等を指定できるように。
- libavdeviceのサポートを追加。
- timestampが0で始まらない音声を--audio-sourceで読み込むと、映像と正しく同期が取れない問題を修正。
- Linux版(Ubuntu 22.04)の依存パッケージの変更。
  libmfx-gen1.2 -> libmfxgen1

2023.04.12 (7.37)
[QSVEncC]
- oneVPL API 2.9に対応。
- エンコード品質をチューニングするオプションを追加。(--tune)

2023.03.29 (7.36)
[QSVEncC]
- mkvの中のAV1をhwデコードできずFailed to DecodeHeaderエラーで終了してしまうのを回避。
- 色空間情報の記載のないy4mファイルの色がおかしくなるのを回避。
- Linux環境で、 pgs_frame_mergeが見つからないというエラーを回避。
- 縮小時のbilinear補間の精度を向上させるとともに、bicubicを追加。
- OpenCLのビルドログの調整。
- 音声・字幕のtimestampに負の値が入ることがあったのを回避。
- デバイス選択時の4GPUまでの制限を解除。
- --gpu-copyオプションを追加。

[QSVEnc.auo]
- 出力する動画の長さが短い場合の警告を追加。
- QSVが利用可能かチェックする際、64bitOSでは64bit版のQSVEncCを使用してチェックするように。

2023.03.07 (7.35)
[QSVEncC]
- 色調を指定したカーブに従って変更するオプションを追加。(--vpp-curves)
- --vpp-overlayを最後のフィルタに変更。
- OpenCLの初期化に失敗した場合に、OpenCLなしで可能な限り処理を続行するように。
- GPU使用率の集計方法を改善。
- --ctuを対応していない世代のGPUで指定した場合のエラー終了を回避。
- ファイルがシークしやすくなるよう--open-gopの取り扱いを改善。

[QSVEnc.auo]
- オブジェクトエクスプローラからドラッグドロップしたファイルがある場合の出力に対応。

2023.02.13 (7.34)
[QSVEncC]
- フレーム時刻をタイムコードファイルから設定するオプションを追加。(--tcfile-in)
- 時間精度を指定するオプションを追加。(--timebase)
- --audio-profileが変更できなかった時に警告を表示するように。
- うまく情報が取得できなかった場合に--check-hwが異常終了する問題を修正。

2023.02.09 (7.33)
[QSVEncC]
- 7.30からPGS字幕のコピーがうまく動作しなくなっていた問題を修正。

2023.02.07 (7.32)
[QSVEncC]
- 7.30から --vpp-resize spline16, spline36, spline64を使用すると、意図しない線やノイズが入る問題を修正。

2023.02.06 (7.31)
[QSVEncC]
- --vpp-afsにrffオプション使用時の誤検出でRFF部が12fpsになることがあったのを修正。

2023.02.05 (7.30)
[QSVEncC]
- ffmpegのライブラリを更新 (Windows版)
  ffmpeg     5.0    -> 5.1
  libpng     1.3.8  -> 1.3.9
  expat      2.4.4  -> 2.5.0
  libsndfile 1.0.31 -> 1.2.0
  libxml2    2.9.12 -> 2.10.3
  libbluray  1.3.0  -> 1.3.4
  dav1d      0.9.2  -> 1.0.0
- --sub-sourceでPGS字幕を読み込むと正常にmuxできない問題を回避。
- --vpp-afsにrffオプションを追加。
- --check-hwでエンコードをサポートするコーデック名も表示するように。
- --check-hwや--check-featuresのログ出力を--log-levelで制御できるように。

2023.01.30 (7.29)
[QSVEncC]
- 画像を焼き込むフィルタを追加。 (--vpp-overlay)
- lowlatency向けにmuxの挙動を調整。
- 動画ファイルに添付ファイルをつけるオプションを追加。 (--attachement-source)
- --perf-monitorでビットレート情報等が出力されないのを修正。
- 音声エンコードスレッド (--thread-audio 1) が動作しなくなっていた問題を修正。

2023.01.22 (7.28)
[QSVEncC]
- --scenario-infoの誤字の修正。

2023.01.22 (7.27)
[QSVEncC]
- --vpp-decimateに複数のフレームをdropさせるオプションを追加。
- AV1のmaster-displayの取り扱いを改善。
- maxcllあるいはmastering displayの片方がないときに、AV1エンコードで適切でないデータが発行されていた問題の修正。
- 言語による--audio-copyの指定が適切に動作していなかった問題を修正。

2023.01.21 (7.26)
[QSVEncC]
- シナリオ情報を渡して画質の最適化を行うオプションを追加。(--scenario-info)
- vbvbufsizeに65535以上の値を指定するとおかしな値が設定されることがある問題を修正。
- dolby-vision-profileを使用した場合でも、指定したchromaloc等が優先されるよう動作を変更。
- OpenGOPの際にIdrIntervalを変更しないように。
- lookahead depthの最小値制限を外す。

2023.01.11 (7.25)
[QSVEncC]
- リモートデスクトップ使用時に初期化に失敗する問題を修正。
- デバッグログを強化。
- lookahead depthはVBR/CBRでも有効な場合があるので、ログに表示するように。
- trellisはH.264エンコード時のみ表示するよう変更。

[QSVEnc.auo]
- エラーメッセージの文字化けを修正。

2022.11.12 (7.24)
[QSVEncC]
- エンコードするフレーム数を指定するオプション(--frames)を追加。
- --fpsで入力フレームレートを強制できるように。
- --vpp-tweakにswapuvオプションを追加。
- --vpp-subburnにfoced flag付きの字幕のみ焼きこむオプションを追加。
- 出力ファイル名が.rawの場合もraw出力とするように。

2022.11.01 (7.23)
[QSVEncC]
- 7.22で--check-featuresのレート制御モードの表示順序が入れ替わっていたのを修正。

[QSVEnc.auo]
- 7.22で適切なレート制御モードを選択できなくなっていたのを修正。

2022.10.30 (7.22)
[QSVEncC]
- --vpp-afsとAV1エンコードを組み合わせた場合に正常に動作しない問題を修正。
- 指定したオプションにより適切なGPUを自動的に選択するように。

[QSVEnc.auo]
- AV1のビット深度を指定する設定欄を追加。

2022.09.21 (7.21)
[QSVEncC]
- --master-display/--maxcll/--dhdr10plusをAV1エンコードに対応。
- 新たなオプションを追加。
  - --repeat-headers
  - --intra-refresh-cycle
  - --hevc-gpb
- --atcseiの情報をログに追加。
- AV1のCQP/ICQの上限を255に変更。
- AV1のGopRefDistのデフォルトを8に変更。
- --vpp-debandでOpenCLコンパイルエラーが生じる問題を修正。
- Arc GPUでOpenCLフィルタとavhw以外の読み込みを組み合わせたときのエラーを修正。

[QSVEnc.auo]
- 設定画面の挙動不審を修正。

2022.09.21 (7.20)
[QSVEncC]
- AV1用に--gop-ref-distパラメータを追加。従来は "1" 固定であったが、これを "4" などに大きくすることで、圧縮率を大きく向上させる。
- AV1の--gop-ref-distのデフォルトを4に変更。(従来は "1" 固定)

2022.09.19 (7.19)
[QSVEncC]
- --hyper-modeのデフォルトをoffに変更する。
  非対応環境でエラーが発生し、特にHEVCエンコードにおいて現時点では回避が難しい場合があるため。

[QSVEnc.auo]
- 誤字の修正。
- 中国語翻訳を更新。

2022.09.18 (7.18)
[QSVEncC]
- QSVEnc 7.17で、--profileを指定すると異常終了するケースがあったのを修正。

[QSVEnc.auo]
- Aviutl中国語対応をされているNsyw様に提供いただいた中国語対応を追加。
  翻訳の対応、ありがとうございました！

2022.09.17 (7.17)
[QSVEncC]
・一部環境で、7.08以降VP9エンコードができなくなっていたのを修正。
・aud/pic-structが使用可能かのチェックを追加。

[QSVEnc.auo]
・設定画面上にツールチップを追加。
・英語表示に対応。

2022.09.07 (7.16)
[QSVEncC]
- QSVEnc 7.15で、Icelakeより前の環境でHEVCエンコードができなくなっていた問題を修正。

2022.09.05 (7.15)
[QSVEncC]
- HEVCでHyperModeが使用できるよう調整。
- デバイスを自動選択する際、意図しないデバイスを掴んでしまうことがあったのを修正。
- OpenCLフィルタの安定性改善。

2022.09.01 (7.14)
[QSVEncC]
- --ssim, --psnrの安定性を改善。
- GPU使用率等の情報収集を改善。

[QSVEnc.auo]
- AuoLink機能を廃止。

2022.08.25 (7.13)
[QSVEncC]
- --audio-streamの処理で、途中で音声のチャンネルが変化した場合にも対応。

2022.08.24 (7.12)
[QSVEncC]
- --vpp-yadifが最終フレームでエラー終了してしまうことがある問題を修正。

2022.08.23 (7.11)
[QSVEncC]
- インタレ解除フィルタをさらに追加。(--vpp-yadif)
- 環境によって、プライマリ以外のGPUがつかめない場合があったのを修正。

[QSVEnc.auo]
- AVX2使用時にFAWの1/2モードが正常に処理できなかったのを修正。

2022.08.17 (7.10)
[QSVEncC]
- Linux環境で、libmfx-gen1.2を導入すると正常に動作しなくなってしまう問題を修正。

2022.08.14 (7.09)
[QSVEncC]
- --disable-opencl使用時にエラー終了してしまう問題を修正。
- Linuxで--check-deviceでGPU名が取得できない問題を修正。
- Linuxで標準入力から読み込ませたときに、意図せず処理が中断してしまう問題を修正。
  なお、これに伴いLinuxではコンソールからの'q'あるいは'Q'でのプログラム終了はサポートしないよう変更した。(Ctrl+Cで代用のこと)

[QSVEnc.auo]
- qaac/fdkaacのコマンドラインに --gapless-mode 2 を追加。

2022.08.08 (7.08)
[QSVEncC]
- HyperModeの検出を修正。
- --hyper-mode on の時になるべくHyperModeが使用できるようパラメータ調整を行うように。
- Bフレームのチェックを追加。
- --check-featuresを拡張し、--fixed-func有効時(FF)と無効時(PG)の時の双方を別々にチェックするように。
- --check-features, --check-environmentが--deviceで指定するデバイスで情報を取得できるように。
- --async-depthの制限を解放。

[QSVEnc.auo]
- HyperModeの設定欄を追加。
- デバイス選択欄を追加。

2022.08.01 (7.07)
- デバイス選択時、デフォルトデバイス以外では異常終了してしまっていた問題を修正。
- 使用可能なデバイスの一覧を表示するオプションを追加。(--check-device)
- AV1のlevelとprofileが入れ替わってしまっていたのを修正。
- AV1ではBフレーム指定を無効に。
  Bフレーム設定を指定しまうと、正常に再生できないファイルが作成されていた。

2022.07.03 (7.06)
- API 2.2で追加されたAdaptiveCQMを有効にするオプションを追加。(--adapt-cqm)
- API 2.4で追加されたAdaptiveRefを有効にするオプションを追加。(--adapt-ref)
- API 2.4で追加されたAdaptiveLTRを有効にするオプションを追加。 (--adapt-ltr)
- QSVEnc 6.10からAV1 hwデコードができないケースがあったのを回避。

2022.07.02 (7.05)
[QSVEncC]
- QSVEnc 7.02から、--vpp-afsと--fixed-func使用時に画面の左端に色の異常がみられる問題を修正。
- oneVPLで非推奨となった--adapt-ltrを廃止。
- その他非推奨関数の使用を削減。

2022.06.28 (7.04)
[QSVEncC]
- oneVPLを更新し、API 2.6に対応。
- HyperMode(DeepLink)を使用可能に。(--hyper-mode)
- AVPacket関連の非推奨関数の使用を削減。
- yuv422読み込み時にcropを併用すると、黒い横線が生じてしまう問題を修正。

[QSVEnc.auo]
- デフォルトの音声ビットレートを変更。
- プリセットの音声ビットレートを変更。
- exe_filesから実行ファイルを検出できない場合、plugins\exe_filesを検索するように。

2022.06.17 (7.03)
[QSVEncC]
- --vpp-colorspace lut3dが正常に動作しない場合があったのを修正。
- --check-features-html使用時の文字化け対策。

[QSVEnc.auo]
- エンコード終了ログの文字化け対策。

2022.06.14 (7.02)
[QSVEncC]
- vpp-colorspaceで3D LUTを適用可能に。(--vpp-colorspace lut3d )
- 3次元ノイズ除去フィルタを追加。(--vpp-convolution3d)
- vpp-colorspaceのhdr2sdr使用時に、ldr_nits, source_peakに関するエラーチェックを追加。
- アスペクト比を維持しつつ、指定の解像度にリサイズするオプションを追加。
- vpp-smoothでfp16やsubgroupがサポートされないときに対応。
- hevc 10bitでavhw使用時に正常にデコードできない問題を修正。
- YUV420でvpp-afs使用時に、二重化するフレームで縞模様が発生してしまう問題を修正。
- Ubuntu 22.04向けパッケージを追加。

[QSVEnc.auo]
- 黒窓プラグイン使用時に設定画面の描画を調整。
- ffmpeg (AAC)で -aac_coder twoloop を使用するように。
- 簡易インストーラを直接実行した場合に、エラーメッセージを表示するように変更。
- ディスク容量が足りない時にどのドライブが足りないか表示するように。
- 外部muxer使用時に、なるべくremuxerで処理するよう変更。
- ScrollToCaret()を使用しないように。
- 音声の一時出力先が反映されなくなっていたのを修正。

2022.04.16 (7.01)
[QSVEncC]
- 環境によってエンコードが開始されず、フリーズしてしまう問題を修正。
- 音声の開始時刻が0でなく、かつ映像と音声のtimebaseが異なる場合の音ズレを修正。

2022.04.07 (7.00)
[QSVEncC]
- Visual Studio 2022に移行。
- AV1エンコード対応準備。(-c av1)
- 使用デバイスを選択するオプションを追加。(--device)

[QSVEnc.auo]
- .NET Framework 4.8に移行。
- パッケージのフォルダ構成を変更。
- 簡易インストーラによるインストールを廃止。
- パスが指定されていない場合、exe_files内の実行ファイルを検索して使用するように。
- ログに使用した実行ファイルのパスを出力するように。
- 相対パスでのパスの保存をデフォルトに。
- 拡張編集使用時の映像と音声の長さが異なる場合の動作の改善。
  拡張編集で音声を読み込ませたあと、異なるサンプリングレートの音声をAviutl本体に読み込ませると、
  音声のサンプル数はそのままに、サンプリングレートだけが変わってしまい、音声の時間が変わってしまうことがある。
  拡張編集使用時に、映像と音声の長さにずれがある場合、これを疑ってサンプリングレートのずれの可能性がある場合は
  音声のサンプル数を修正する。
- エンコードするフレーム数が0の場合のエラーメッセージを追加。
- ログの保存に失敗すると、例外が発生していたのを修正。
- ログの保存に失敗した場合にその原因を表示するように。
- muxエラーの一部原因を詳しく表示するように。
  mp4出力で対応していない音声エンコーダを選択した場合のエラーメッセージを追加。
- エラーメッセージ
  「x264が予期せず途中終了しました。x264に不正なパラメータ（オプション）が渡された可能性があります。」
    の一部原因を詳しく表示するように。
  1. ディスク容量不足でエンコードに失敗した場合のエラーメッセージを追加。
  2. 環境依存文字を含むファイル名- フォルダ名で出力しようとした場合のエラーメッセージを追加。
  3. Windowsに保護されたフォルダ等、アクセス権のないフォルダに出力しようとした場合のエラーメッセージを追加。

2022.03.06 (6.10)
[QSVEncC]
・ffmpeg関連のdllを更新。(Windows版)
  ffmpeg     4.x    -> 5.0
  expat      2.2.5  -> 2.4.4
  fribidi    1.0.1  -> 1.0.11
  libogg     1.3.4  -> 1.3.5
  libvorbis  1.3.6  -> 1.3.7
  libsndfile 1.0.28 -> 1.0.31
  libxml2    2.9.10 -> 2.9.12
  libbluray  1.1.2  -> 1.3.0
  dav1d      0.6.0  -> 0.9.2

[QSVEnc.auo]
・出力の際、Aviutlが開いているファイルに上書きしないように。
・H.264のfeaturesが設定画面で認識されないのを修正。

2022.02.26 (6.09)
[QSVEncC]
・6.08でb-pyramidが使用できなくなっている問題を修正。

[QSVEnc.auo]
・出力の際、Aviutlが開いているファイルに上書きしないように。
・設定が行われていない場合に、前回出力した設定を読み込むように。
・複数のAviutlプロセスで出力していても正常にチャプターを扱えるように。
・ログ出力のモードを変更すると正常に出力できないことがあったのを修正。

2022.02.08 (6.08v3)
2022.02.08 (6.08v2)
・SetThreadInformationが使用できない環境での問題を回避。

2022.02.06 (6.08)
・VP9エンコードに対応。(-c vp9)
・Dolby Visionのrpuを読み込み反映させるオプションを追加。(--dolby-vision-rpu)
・Dolby Visionのプロファイルを指定するオプションを追加。(--dolby-vision-profile)

2022.01.27 (6.07)
・VBRモード等のビットレート指定モード使用時に、長時間エンコードするとビットレートが著しく低下してしまう問題を解消。

2021.12.1 (6.06)
・スレッドの優先度とPower Throttolingモードを指定するオプションを追加。(--thread-priority, --thread-throttling)
・d3d11メモリの使用できない環境での問題に対処。
・一部の環境で、QSVEncC64.featureCache.txtが存在するとQSVEnc.auoの設定画面を表示する際にクラッシュするのを修正。

2021.11.01 (6.05)
・--dar指定時に負の解像度を使用すると、sar扱いで計算され意図しない解像度となるのを修正。
・API v2.05の--vpp-denoiseのモード指定に対応。
  なお、9955ドライバでも使用できない模様。

2021.10.14 (6.04)
・想定動作環境にWindows11を追加。
・Windows11の検出を追加。
・スレッドアフィニティを指定するオプションを追加。(--thread-affinity)
・ログの各行に時刻を表示するオプションを追加(デバッグ用)。(--log-opt addtime)
・dynamic hdr10plusのmetadataをコピーするオプションを追加。(--dhdr10-info)
・bitstreamのヘッダ探索をAVX2/AVX512を用いて高速化。
・12bit深度を10bit深度に変換するときなどに、画面の左上に緑色の線が入ることがあったのを修正。

2021.09.25 (6.03)
・起動速度をわずかに高速化。
・--caption2assが使用できなかったのを修正。
・OpenCLの情報を表示するオプションを追加。(--check-clinfo)
・--vpp-smoothでquality=0のときにはprec=fp16を使用できないので、自動的にprec=fp32に切り替え。
・ログの各行に時刻を表示するオプションを追加。(--log-opt addtime)

2021.09.19 (6.02)
・VPLを2021.6に更新。
・VPLの実装を使用して、利用可能なモードを列挙する--check-implを追加。
・--vpp-resize lanczosxの最適化。11700Kで50%高速化。
・--vpp-smoothの最適化。11700Kで25%高速化。
・--vpp-knnの最適化。11700Kで約2倍高速化。
・OpenCLフィルタのパフォーマンス測定用のオプションを追加。(--vpp-perf-monitor)
・音声にbitstream filterを適用する--audio-bsfを追加。

2021.09.06 (6.01)
・6.00で--d3d9, --disable-d3dが効かなかったのを修正。
・--vpp-colorspace使用時に、解像度によっては最終行に緑の線が入る問題を修正。
  6.00で修正しきれていなかった。

2021.09.05 (6.00)
・MediaSDKからoneAPI Video Processing Library(VPL)に移行し、API 2.04に対応。
  API 2.xx は、Rocketlake/Tigerlake(Winodws版)から使用可能(のはず)。
・yuv444→nv12で指定すべき横解像度が誤っていたのを修正。これにより最終行に緑の線が入る問題を解消。
  (5.06で修正しきれていなかった)
・audio-delayが効いていなかったのを修正。

2021.08.12 (5.07)
・vpp-subburnで使用できるフォントのタイプを更新。
・audio-delayが効いていなかったのを修正。
・--vpp-colorspace使用時に最終行に緑の線が入る問題を修正。

2021.07.26 (5.06)
・ssim/psnrを計算できるように。(--ssim/--psnr)
・yuv444→nv12で指定すべき横解像度が誤っていたのを修正。これにより最終行に緑の線が入る問題を解消する。
・yuv444→p010のavx2版の色ずれを修正。
・rgb読み込みとOpenCLフィルタが組み合わせられなかったのを修正。
・Linuxで--disable-vaで動作しなかった問題を修正。
・--vpp-colorspaceがLinuxで動作しなかった問題を修正。

2021.06.12 (5.05)
・avhw以外の読み込みとOpenCLフィルタがつながっている場合の処理を効率化。
・入力ファイルと出力ファイルが同じである場合にエラー終了するように。
・--vpp-decimateで異常終了が発生することがあったのを修正。
・y4m読み込みの際、指定したインタレ設定が反映されないことがあったのを修正。
・一部のAvisynth環境で生じるエラー終了を回避。

2021.05.29 (5.04)
・使用可能な環境では、常にd3d11を優先して使用するよう変更。
  MediaSDKのサンプルでもこのような変更が行われていたので対応。
  https://github.com/Intel-Media-SDK/MediaSDK/commit/c4fbaedd8a827ec36ee312e978e993d3f938201c
・5.02から、Failed to find d3d9 deviceと出てしまうのを修正。
・5.01から、avhw以外の読み込みから直接OpenCLフィルタに渡すと、フレーム順序が入れ替わったりしてしまっていた問題を修正。
・5.01から、OpenCLでcropすると色成分がずれるのを修正。
・Broadwell以前の環境でvpp-mpdecimate/decimateがフリーズしてしまう問題を回避。

2021.05.23 (5.03)
・raw出力、log出力の際にカレントディレクトリに出力しようとすると異常終了が発生する問題を修正。
・Win8.1のSandybridgeでのデコードエラーを修正。
・cropとOpenCLフィルタを併用すると、色成分がずれてしまうのを修正。

2021.05.16 (5.02)
[QSVEncC]
・5.01で、必要ない場面でもd3d11が優先して使用されていたのをd3d9を使用するようもとに戻す。
・5.01で、--avsync forcecfr使用時に連続16フレーム以上挿入ができなかったのを18000フレーム(実際は無制限)に挿入可能とする。
  連続16フレーム以上挿入しようとすると異常終了が発生していた。
・5.01で、--vpp-mpdecimate, --vpp-decimateを使用すると誤ってリサイズ行われる状態になっていたのを修正。
・5.01で、--vpp-pad, --cropで正しくない解像度操作・変更がなされていたのを修正。
・5.01で、--vpp-afs, --vpp-mpdecimate, --vpp-decimateで異常終了(Failed to acquire OpenCL interop)が発生していたのを修正。
・5.01で、--vpp-deinterlace bobで異常終了(Application provided invalid, non monotonically increasing dts to muxer)が発生していたのを修正。
・5.01で、Win7のSandybridge環境ではデコードが正常に行われなくなってしまう(緑の絵が出る)問題を回避する。
・5.01で、-c raw使用時にOpenCLフィルタを使用するとエラーが発生していたのを修正。
・デバッグ用のログメッセージの改善。

[QSVEnc.auo]
・設定画面からリサイズを指定しても効果がなかったのを修正。

2021.05.08 (5.01)
・5.00 beta1から動かなかったLinuxビルドを修正。
・avsw/avhwでのファイル読み込み時にファイル解析サイズの上限を設定するオプションを追加。(--input-probesize)
・--input-analyzeを小数点で指定可能なよう拡張。
・読み込んだパケットの情報を出力するオプションを追加。( --log-packets )
・data streamに限り、タイムスタンプの得られていないパケットをそのまま転送するようにする。
・オプションを記載したファイルを読み込む機能を追加。( --option-file )
・動画情報を取得できない場合のエラーメッセージを追加。
・コピーするtrackをコーデック名で選択可能に。
・字幕の変換が必要な場合の処理が有効化されていなかったのを修正。
・5.00betaで-c rawを指定してもraw出力されないのを修正。
・--vpp-subburnでサイズが0の字幕がくると異常終了が発生したのを修正。
・OpenCLフィルタを使用時、またはAV1デコード時は、d3d11モードを優先するように。

・--videoformatに関しては入力から容易に取得できないので、"auto"を削除。
・--audio-source/--sub-sourceを複数指定した場合の挙動を改善。
・字幕のmetadataが二重に出力されてしまっていた問題を修正。
・--sub-metadata, --audio-metadataを指定した場合にも入力ファイルからのmetadataをコピーするように。

・下記OpenCLによるvppフィルタを追加。
  - --vpp-afs
  - --vpp-colorspace
  - --vpp-deband
  - --vpp-decimate
  - --vpp-edgelevel
  - --vpp-mpdecimate
  - --vpp-nnedi
  - --vpp-pad
  - --vpp-pmd
  - --vpp-smooth
  - --vpp-tweak
  - --vpp-unsharp
  - --vpp-warpsharp

・yuv444→nv12/p010/ayuv/y410変換のAVX2/SSE2版を追加。

既知の問題
・YUV422/YUV444では、vppフィルタが動作しない場合がある。

2021.04.07 (5.00 beta2)
・SandyBridgeなどOpenCLのない環境で動作するように。
・Broadwell以前の環境で、Failed to find d3d9 device.で動作しなくなっていたのを修正。
・Broadwell以前の環境で、OpenCLでのコンパイルエラーが発生するのを修正。
・不安定だったCPU版の--vpp-delogoを廃止し、OpenCL版の--vpp-delogoを実装。
・RocketlakeでのAV1をHWデコードに対応。ただし、--d3d11を併せて指定する必要がある。
・--fixed-funcを指定するとYUV444エンコードができないのを修正。
・--async-depthのデフォルト値を3に変更。不必要にメモリを多く使用していた。
・-c raw使用時に、OpenCLフィルタを使用すると異常終了が発生することがあったのを修正。

既知の問題
・Linux環境ではビルドできない。
・YUV422/YUV444では、vppフィルタが動作しない場合がある。

2021.03.30 (5.00 beta1)
新機能
・MediaSDKの更新、API 1.35に対応。
・Icelakeへの対応を拡充。
・Rocketlake対応の初期実装。
・内部実装の刷新し、OpenCLフィルタを組み込み可能に。
  ・--vpp-knnの追加。
  ・--vpp-transposeの追加。
・HEVC YUV422/YUV444デコードに対応。(Icelake/Rocketlake)
・HEVC YUV444エンコードに対応。(--output-csp, Icelake/Rocketlake)
・VP9 YUV444デコードに対応。(Icelake/Rocketlake)
・--check-featuresで、HWデコードに対応している色空間の情報を追加。
・リサイザのアルゴリズムを指定するオプションを追加。(--vpp-resize/--vpp-resize-mode)
・H.264 Level 6, 6.1, 6.2を追加。

既知の問題
・Linux環境ではビルドできない。
・--vpp-delogoが動作しない。
・avhwリーダー以外を使用する際に動作が遅くなる場合がある。
・YUV422/YUV444では、vppフィルタが動作しない場合がある。

廃止
・vpp-half-turnを廃止。--vpp-transform等で代用できる。

2021.02.17 (4.13)
・AvisynthのUnicode対応を追加。
・Windows 10のlong path supportの追加。
・--audio-source / --sub-source でmetadataを指定可能なよう拡張。
・言語による音声や字幕の選択に対応。
・bit深度を下げるときの丸め方法を変更。
・chapterを読み込む際に、msの値を正しく取得できない場合があったのを修正。

2020.11.23 (4.12)
・extbrcが9025Betaドライバに更新するとVBRモードで使用できなくなっていた問題を回避。
・chromalocを設定すると出力が異常になる場合があったのを修正。

2020.11.22 (4.11)
・chapterを読み込む際に、msの値を正しく取得できない場合があったのを修正。
・AVX/AVX2が使用できない場合のLinuxビルドを修正。
・WinBRCの対象のレート制御モードかどうかをチェックするように。

2020.11.19 (4.10)
・4.09でB pyramidとPyram QP Offsetが使用できなくなっていたのを修正。
・extbrcオプションを追加。

2020.11.12 (4.09)
・Media SDK 1.34に対応。
・yuv422からCPUでyuv420に変換するように。
  dGPUがある環境でGPUでのyuv422からyuv420への変換が正常に動作しないことがあるようなので。
・パフォーマンスモニタが正常に取得できないことがあるのを改善。

2020.09.30 (4.08)
・Apple proresがデコードできないのを修正。
・raw読み込み時に色空間を指定するオプションを追加。( --input-csp )
  yuv420/422/444の8-16bitの読み込みに対応。
・--check-libの結果の成否によってプログラムの戻り値を変更するように。
・HEVCエンコード時に--output-depth 10指定時に自動的にmain10を使用するように。
  いままでは--profile main10と併せて使用しないと10bit深度でエンコードされなかった。
・Linuxビルドを更新、Broadwell以降のIntel iGPUでのQSVエンコードが容易に。
・Linuxでのビルド方法について追記。

2020.08.06 (4.07)
・ロードするAvisynth.dllを指定するオプションを追加。(--avsdll)

2020.08.02 (4.06)
[QSVEncC]
・場合により、異常終了が発生することがあったのを修正。

2020.07.29 (4.05)
[QSVEncC]
・Media SDKを2020 R1に更新。
・ffmpeg関連のdllを更新。
  これにより、ts/m2tsへのPGSのmuxを可能とする。
・--audio-stream stereoが動作しないのを修正。
・mkv出力時にdefault-durationが設定されるように。
・--chromalocが使用できないのについて回避策を実装。

2020.06.16 (4.04)
[QSVEncC]
・一部のHEVCファイルで、正常にデコードできないことがあるのに対し、可能であればswデコーダでデコードできるようにした。
・--audio-sourceでもdelayを指定できるように。
・avs読み込みで、より詳細なAvisynthのバージョンを取得するように。
・4.02からvpy読み込みがシングルスレッド動作になっていたのを
  マルチスレッド動作に戻した。

[QSVEnc.auo]
・QSVEnc.auoの設定画面でも、--output-resに特殊な値(負の値)を指定できるように。

2020.05.31 (4.03)
[QSVEncC]
・遅延を伴う一部の--audio-filterで音声の最後がエンコードされなくなってしまう問題を修正。
・lowlatencyが使用できないのを修正。
・--video-tagを指定すると異常終了してしまうのを修正。 
・出力するmetadata制御を行うオプション群を追加。
  --metadata
  --video-metadata
  --audio-metadata
  --sub-metadata
・streamのdispositionを指定するオプションを追加。 (--audio-disposition, --sub-disposition)
・--audio-source/--sub-sourceでうまくファイル名を取得できないことがあるのを修正。
・--helpに記載のなかった下記オプションを追記。
  --video-tag
  --keyfile
  --vpp-smooth
・オプションリストを表示するオプションを追加。 (--option-list)

2020.05.06 (4.02)
[QSVEncC]
・yuv444→yv12/p010変換のマルチスレッド時のメモリアクセスエラーを修正。
・遅延を最小化するモードを追加。 (--lowlatency)
  エンコードのスループット自体は下がってしまうので、あまり使い道はないかも?

[QSVEnc.auo]
・外部エンコーダ使用時に、音声エンコードを「同時」に行うと異常終了するのを修正。

2020.04.15 (4.01)
[QSVEncC]
・3.33からIvyBridgeの環境でvppを使用できない問題を回避。

[QSVEnc.auo]
・デフォルト音声エンコーダの設定が反映されないのを修正。

2020.04.05 (4.00)
[QSVEncC]
・音声デコーダやエンコーダへのオプション指定が誤っていた場合に、
  エラーで異常終了するのではなく、警告を出して継続するよう変更。
・3.33からSandyBridge/IvyBridgeの環境でvppを使用できない問題を回避。
・--chapterがavsw/avhw利用時にしか効かなかったのを修正。

[QSVEnc.auo]
・QSVEnc.auoで内部エンコーダを使用するモードを追加。
  こちらの動作をデフォルトにし、外部エンコーダを使うほうはオプションに。
・QSVのない環境で設定画面を開こうとすると異常終了してしまうのを修正。

2020.03.07 (3.33)
[QSVEncC]
・avsw/avhw読み込み時の入力オプションを指定するオプションを追加。(--input-option)
・Media SDKのcolorフィルタを使用するテストコードを追加。(--vpp-colorspace)
・trueHDなどの一部音声がうまくmuxできないのを改善。
・IceLake世代が正常に判定されないのを修正。
・QSVEnc.auoの修正に対応する変更を実施。

[QSVEnc.auo]
・QSVEnc.auoから出力するときに、Aviutlのウィンドウを最小化したり元に戻すなどするとフレームが化ける問題を修正。

2020.02.29 (3.32)
[QSVEncC]
・caption2assが正常に動作しないケースがあったのを修正。
・helpの見直し。
・3.31で--cqpが正常に動作しない問題を修正。

[QSVEnc.auo]
・簡易インストーラの安定動作を目指した改修。
  必要な実行ファイルをダウンロードしてインストールする形式から、
  あらかじめ同梱した実行ファイルを展開してインストールする方式に変更する。
・デフォルトの音声エンコーダをffmpegによるAACに変更。
・QSVEnc.auoの設定画面のタブによる遷移順を調整。

2020.02.20 (3.31)
[QSVEncC]
・コマンドラインの指定ミスの際のエラーメッセージを改善。
・mux処理を見直し、シークしづらくなるなどの症状を改善。

[QSVEnc.auo]
・ビットレート上限の解放。

2020.02.02 (3.30)
[QSVEncC]
・vpp-subは最近安定して動作しないため、無効化。
・colormatrix等の情報を入力ファイルからコピーする機能を追加。
  --colormtarix auto
  --colorprim auto
  --transfer auto
  --chromaloc auto
  --colorrange auto
・VUI情報、mastering display, maxcllの情報をログに表示するように。
・終了時にエラー終了してしまうことがあるのを修正。
・ログに常に出力ファイル名を表示するように。
・VUI情報、mastering dsiplay, maxcllの情報をログに表示するように。

[QSVEnc.auo]
・QSVEncCとの連携のための実装を変更。
  たまに緑のフレームが入ったりする(?)という問題に対処できているとよいが…。

2020.01.18 (3.29)
[共通]
・動作環境を変更。
・Media SDKを2019 R1に更新。
・プロセスのGPU使用率情報を使用するように。

[QSVEncC]
・HDR関連のmeta情報を入力ファイルからコピーできるように。
  (--master-display copy, --max-cll copy)
・ffmpeg関連のdllを更新。
  AV1のソフトウェアデコードを可能に。
  libogg-1.3.3 -> 1.3.4
  twolame-0.3.13 -> 0.4.0
  wavpack-5.1.0 -> 5.2.0
  libxml2-2.9.9 -> 2.9.10
  dav1d-0.5.2 !new!

2019.12.24 (3.28)
[QSVEncC]
・音声処理でのメモリリークを解消。
・音声エンコード時のエラーメッセージ強化。
・字幕のコピー等が動かなくなっていたのを修正。
・trueHD in mkvなどで、音声デコードに失敗する場合があるのを修正。
・音声に遅延を加えるオプションを追加。 ( --audio-delay )
・mkv入りのVC-1をカットした動画のエンコードに失敗する問題を修正。

[QSVEnc.auo]
・簡易インストーラを更新。
・AVX2版のyuy2→nv12i変換の誤りを修正。

2019.11.23 (3.27)
[QSVEnc.auo]
・プロファイルの保存ができなくなっていたのを修正。

2019.11.19 (3.26)
[QSVEnc.auo]
・リサイズが行えないのを修正。
・vpp-deinterlace bobが正常に動作しない問題を修正。

[QSVEncC]
・output-resに縦横のどちらかを負の値を指定できるように。
アスペクト比を維持したまま、片方に合わせてリサイズ。ただし、その負の値で割り切れる数にする。
--output-res -4x1080

2019.11.15 (3.25)
[QSVEnc.auo]
・QSVEnc.auo-QSVEncC間のプロセス間通信を高速化。
・QSVEnc.auoの出力をmp4/mkv出力に変更し、特に自動フィールドシフト使用時のmux工程数を削減する。
  また、QSVEncCのmuxerを使用することで、コンテナを作成したライブラリとしQSVEncCを記載するようにする。

[QSVEncC]
・VC-1をハードウェアデコードの対象から外す。
  3.04以降、VC-1のでコードができなくなっているが、復旧できなかった。
・高負荷時にデッドロックが発生しうる問題を修正。
・CPUの動作周波数が適切に取得できないことがあったのを修正。
・字幕ファイルを読み込むオプションを追加。 (--sub-source )
・--audio-sourceの指定方法を拡張。
・avsからの音声読み込みを可能に。
・音声エンコードが正常に動作しない場合があったのを修正。
・mux時にmaster-displayやmax-cllの情報が化けるのを回避。

2019.06.26 (3.24)
・--sub-copy asdataの挙動の見直し。
・3.21から-c rawや--disable-d3dなどを使用すると、"Failed to SynchronizeFirstTask : unknown error" で
  エラー終了してしまうようになっていたのを修正。

2019.06.26 (3.23)
[QSVEncC]
・データストリームをコピーするオプションを追加する。(--data-copy)

2019.06.23 (3.22)
[QSVEncC]
・--sub-copyで字幕をデータとしてコピーするモードを追加。
  --sub-copy asdata
・--audio-codecにデコーダオプションを指定できるように。
  --audio-codec aac#dual_mono_mode=main
・RGB読み込みができなくなっていたのを修正。

2019.06.15 (3.21)
[QSVEncC]
・--vpp-deinterlace noneでインタレ保持が有効になってしまう問題を修正。
・--chapterでmatroska形式に対応する。
・ffmpegと関連dllを追加/更新。
  - [追加] libxml2 2.9.9
  - [追加] libbluray 1.1.2
  - [追加] aribb24 rev85
  - [更新] libpng 1.6.34 -> 1.6.37
  - [更新] libvorbis 1.3.5 -> 1.3.6
  - [更新] opus 1.2.1 -> 1.3.1
  - [更新] soxr 0.1.2 -> 0.1.3

2019.04.26 (3.20)
[QSVEnc.auo]
・インタレ解除でbob/itなどを使用すると、適切にフレームレートが反映されず、
  音ずれしてしまうのを修正。

[QSVEncC]
・3.19で--mbbrcが効かないのを修正。

2019.04.19 (3.19)
[共通]
・VC++2019に移行。

[QSVEnc.auo]
・簡易インストーラを更新。(VC++2019対応)

[QSVEncC]
・一部のH.264ストリームなどで、デコードが停止してしまう問題を修正。
  スカパープレミアムなどで発生すると報告いただいた。
・TrueHDな一部の音声をコピーしようとしても正常にコピーされないのを修正。
・Adaptive LTR を有効にするオプションを追加。(--adapt-ltr)
  CBR, VBRなど一部のモードでのみ動作。

2019.03.24 (3.18)
[QSVEnc.auo/QSVEncC 共通]
・Haswell環境のd3d11モードでBフレームありのH.264インタレ保持エンコを行うと、映像が乱れることがあるので
  その場合にはBフレームを無効化するように。

[QSVEncC]
・映像のcodec tagを指定するオプションを追加。(--video-tag)
・音声エンコード時のtimestampを取り扱いを改良、VFR時の音ズレを抑制。

2018.12.17 (3.17)
[QSVEncC]
・--master-displayが正常に動作しない場合があったのを修正。

2018.12.11 (3.16)
[QSVEnc.auo]
・Aviutlからのフレーム取得時間がエンコードを中断した場合に正常に計算されないのを修正。

2018.12.10 (3.15)
[QSVEnc.auo]
・自動フィールドシフト使用時、widthが32で割り切れない場合に範囲外アクセスの例外で落ちる可能性があったのを修正。

2018.12.04 (3.14)
[QSVEncC]
・benchmarkモードが正常に動作しなかったのを修正。

[QSVEnc.auo]
・AuoLinkモードで、音声エンコーダが使用できなくなっていたのを修正。

2018.11.24 (3.13)
[QSVEncC]
・読み込みにudp等のプロトコルを使用する場合に、正常に処理できなくなっていたのを修正。
・--audio-fileが正常に動作しないことがあったのを修正。

2018.11.18 (3.12)
[QSVEncC]
・Caption.dllによる字幕抽出処理を実装。(--caption2ass)
・古いAvisynthを使うと正常に動作しなくなっていたのを修正。

[QSVEnc.auo]
・簡易インストーラを更新。
  - Apple dllがダウンロードできなくなっていたので対応。
  - システムのプロキシ設定を自動的に使用するように。

2018.10.19 (3.11)
[共通]
・SandyBridgeやIvyBridgeなどでvppを使用すると、
  エンコードが実行できない場合があったので対策を実施。

[QSVEnc.auo]
・QSVEnc.auoの設定画面からwav出力できなかったのを修正。
  指定された動画エンコーダは存在しません。[ ]とエラーが出てしまっていた。
・QSVEnc.iniにffmpegによるAACエンコードと、デュアルモノ分離の設定を追加。
・faw2aac使用時にも音声エンコ後バッチ処理を追加。
  (ただし、faw2aac使用時の音声エンコ前バッチ処理は実施しない)

[QSVEncC]
・--check-featuresに--vpp-mctfのチェックを追加。

2018.10.12 (3.10)
[共通]
・Intel Media SDK 2018 R2 (API v1.27)に更新。
・Motion Compensate Temporal Filter (MCTF) を追加。 (--vpp-mctf)

[QSVEnc.auo]
・一時フォルダの相対パス指定に対応した。
・多重音声を扱う際、muxer.exeがエラー終了してしまうのを修正。

[QSVEncC]
・--vbv-bufsizeを追加。
・一部のmp4/mkv等のコンテナに入った10bit HEVCの入力ファイルが正常にデコードできない問題を解消。
・一部の動画ファイルで、音ズレの発生するケースに対処。

2018.08.01 (3.09)
[QSVEncC]
・進捗状況でtrimを考慮するように。
・OpenCLがまともに動作しない環境でのクラッシュを回避。
  まれによくあることらしい。
・3.00以降、パイプ出力できない場合があったのを修正。

2018.07.10 (3.08)
[QSVEncC]
・音声エンコーダにオプションを引き渡せるように。
  例: --audio-codec aac:aac_coder=twoloop
・音声エンコード時にプロファイルを指定できるように。(--audio-profile)
・高ビットレートでのメモリ使用量を少し削減。
・可変フレームレートなどの場合に、中途半端なフレームレートとなってしまうのを改善。
・音声のほうが先に始まる場合の同期を改善。
・HEVCのtierを指定するオプションを追加。(--tier)

2018.07.05 (3.07)
[QSVEncC]
・--audio-fileが正常に動作していなかったのを修正。
・--colorprimや--transferなどに不足していたオプションを追加。
・--input-analyzeの効果を改善。
・raw出力の際、--vpp-deinterlaceが効かないのを改善。

2018.06.10 (3.06)
[QSVEncC]
・--check-featuresを高速化。
・--avsync forcecfr/vfrが正常に動作しないことがあるのを修正。
・音声エンコード系のオプションが意図しない動作をすることがあったのを修正。

2018.06.03 (3.05)
[QSVEnc.auo]
・3.04でプラグインが認識されないことがあったのを修正。

[QSVEncC]
・avs/vpy/y4mリーダーを使用すると落ちていたのを修正。

2018.06.02 (3.04)
[QSVEncC]
・ffmpegと関連ライブラリのdllを更新。
・--audio-codec / --audio-bitrate / --audio-samplerate / --audio-filter等のコマンドを
  トラックを指定せずに指定した場合、入力ファイルのすべての音声トラックを処理対象に。
・vfrを保持したエンコードに対応。(--avsync vfr)
・--max-cll / --masterdisplay 使用時の互換性を改善。
・chroma locationのフラグを指定するオプションを追加。
・インタレ保持エンコードでmuxしながら出力する際、フィールド単位でmuxせず、フレーム単位でmuxするように。

2018.05.14 (3.03)
[QSVEncC]
・HDR関連のmetadataの取り扱いを改善。
・映像と音声の同期を改善。
・プロセスのロケールを明示的にシステムのロケールに合わせるように。

2018.04.23 (3.02)
[QSVEnc]
・設定画面のコマンド表示欄のダブルクリック時の挙動を修正。
・リサイズが反映されないのを修正。
・AuoLink使用時の不審な挙動を修正。

2018.04.21 (3.01)
[QSVEnc]
・設定画面が120dpiベースになっており、96dpiで表示するとレイアウトが崩れるのを修正。

2018.04.21 (3.00)
[共通]
・Intel Media SDK 2018 R1 (API v1.26)に対応。
・VQPモードを廃止。ICQなどの登場により役目を終えた。
・extbrcオプションを廃止。
・シーンチェンジ検出を廃止。あまり目立った効果はなかった。

[QSVEnc]
・エンコーダをプラグインに内蔵せず、QSVEncCにパイプ渡しするように。
  Aviutl本体プロセスのメモリ使用量を削減する。

[QSVEncC]
・API 1.26で追加されたHEVC関連のオプションを追加。(--tskip, --sao, --ctu)
  基本的には、今後登場するCPU用(Kabylake世代では使用できない)。
・HDR関連metadataを設定するオプションを追加。(--max-cll, --master-display)
・"%"を含む出力ファイル名で出力しようとすると落ちるのを修正。
・"%"を含む出力ファイル名で--logを指定すると落ちるのを修正。
・yv12(10bit)->p010[AVX2]では、AVX2が使用されていなかったのを修正。
・avswのデコーダのスレッド数を16までに制限。
・rotationのmetadataが入力ファイルに存在すればコピーするように。

2018.01.13 (2.74)
[共通]
・HEVCエンコードでweightbが使えるように。
・Kabylake以降では、HEVCの10bit depthを強制的に有効に。
  Kabylake以降では、HEVCの10bit depthに対応しているはずだが、これがQueryで正常に判定されないことがある。
・ログ出力を改善。

[QSVEncC]
・--audio-copy/--audio-codec/--sub-copy指定時に、入力ファイルに音声/字幕トラックがない場合でもエラー終了しないように。
・linuxでビルドできなくなっていたのを修正。
・avsからのyuv420/yuv422/yuv444の高ビット深度読み込みに対応。
  ただし、いわゆるhigh bitdepth hackには対応しない。

2017.08.22 (2.73)
[QSVEncC]
・9/12/14/16bit深度のyuv422をy4m読み込みを修正。

2017.08.16 (2.72)
[QSVEncC]
・2.63以降、raw出力が正常に動作しなかったのを修正。
・高ビット深度のyuv422/yuv444をy4mから読み込むと色成分がおかしくなるのを修正。 
・ヘルプの修正。

2017.07.01 (2.71)
[共通]
・2.70でdGPU付きの環境だと正常に動作しないことがあったのを修正。
・2.70で起動が遅くなっていたのを修正。
・la/la-hrdでビットレートが表示されていなかったのを修正。

2017.06.20 (2.70)
[共通]
・Braswellなど一部の環境で正常に動作しなかったのを修正。

2017.06.18 (2.69)
[QSVEncC]
・--audio-streamを使用した際に、条件によっては、再生できないファイルができてしまうのを修正。

2017.06.17 (2.68)
[共通]
・FadeDetectをKabylakeより前の世代では無効化。
  やっぱりKabylake以前では、不安定でエンコードが途中で終了あるいはフリーズしてしまうようだ…。

[QSVEnc.auo]
・2.67で、HEVC 10bitでエンコードすると絵が破綻する問題を修正。

2017.06.17 (2.67)
[共通]
・Intel Media SDK 2017 R1 (API v1.23)に対応。
・fade-detectを有効に。
・weightb/weightpが動作しなくなっていたのを修正。

[QSVEncC]
・--repartition-checkオプションを追加(H.264エンコード時のみ)。
・avsw/y4m/vpyでのyuv422読み込みに対応。
  ただし、d3d9/d3d11メモリモードは使用できず、swメモリモードに切り替わる。
・avswでのrgb読み込みに対応。
・--audio-streamによるデュアルモノラルの分離などが正常に動作しないのを修正。
・--check-featuresにデコーダの機能を表示するように。

2017.06.12 (2.66)
[QSVEnc.auo]
・16で割り切れない解像度などで色ズレが発生していたのを修正。

[QSVEncC]
・avs/aviからYUY2で読み込んだ際、16で割り切れない解像度の場合に色ズレが発生していたのを修正。

2017.06.11 (2.65)
[QSVEncC]
・高ビット深度をy4m渡しすると、絵が破綻するのを修正。

2017.06.10 (2.64)
[QSVEnc.auo]
・2.63でAuoLinkモードが0xc0000094例外で正常に動作しなかったのを修正。

[QSVEncC]
・2.63でavi読み込みしようとするとエラー終了してしまう場合があったのを修正。

2017.06.08 (2.63)
[共通]
・d3d11モードでも10bit深度のエンコードを可能に。
・Windowsのビルドバージョンをログに表示するように。
・32で割りきれない高さの動画をインタレ保持エンコードできない場合があったのを修正。

[QSVEnc.auo]
・簡易インストーラを更新。

[QSVEncC]
・ffmpegと関連ライブラリのdllを更新。
・HEVCのGPB無効化が使用できなくなっていたのを修正。
・QSVデコード時の安定性を改善。
・vpyリーダー使用時に、エンコードを中断しようとするとフリーズしてしまう問題を修正。
・avsw読みでYUV444のソースを読み込めるように。
・字幕のコピーが正常に行われない場合があったのを修正。
・Intel Media SDKの使用するスレッド数を指定するオプションを追加。(--mfx-thread <int>)
  "2"以上で指定できるが、0や1にはできない。デフォルトは自動( = 論理プロセッサ数)。
  なるべくCPU使用率を下げたい場合に、--mfx-thread 2 とすると、わずかにCPU使用率が下がるかもしれない。

2017.01.08 (2.62)
[QSVEncC]
・KabylakeのHEVC 10bitエンコードに対応。
・GPUの情報が適切にとれない場合があったのを修正。

2017.01.05 (2.61)
[QSVEncC]
・2.57以降、vpyリーダーが正常に動作しないのを修正。

2016.12.19 (2.60)
[QSVEncC]
・mkvを入力としたHEVCエンコードで、
  エンコード開始直後にデッドロックしてしまうのを解消。

[QSVEnc.auo]
・簡易インストーラを更新。

2016.12.05 (2.59)
[QSVEncC]
・chapter読み込み時にtrimを反映しない--chapter-no-trimオプションを追加。

[QSVEnc.auo]
・簡易インストーラを更新。

2016.11.06 (2.58)
[QSVEncC]
・2.55から、avsync forcecfr時が正常に動作しないことがあったのを修正。

[QSVEnc.auo]
・簡易インストーラを更新。

2016.09.29 (2.57)
[QSVEncC]
・avsw/vpyリーダーで10bit読み込みに対応。
・ロゴを付加するオプションを追加。(--vpp-delogo-add)
  SSE4.1バージョンのみ。
・--audio-sourceが期待どおりに動作しない場合があったのを修正。
・エンコードを不安定にするオプション"--fade-detect"を無効化。
・まれにエンコード完了時でフリーズしてしまうを回避。
・音声処理のエラー耐性を向上。
・インタレ解除指定時はデフォルトで--tffとして扱うように。

2016.09.11 (2.56)
[QSVEnc.auo]
・AuoLink時に、常に29.97fpsでエンコードされるようになっていたのを修正。

2016.09.03 (2.55)
[共通]
・aud / pic_structを付加するオプションを追加。

[QSVEncC]
・SkylakeのHW HEVC 10bitデコードに対応。
・ffmpegのdllを更新。

2016.07.09 (2.54)
[QSVEnc]
・使用されていないswの情報を取得・表示しないように。

[QSVEncC]
・avqsv/avswリーダーで読み込む際の入力ファイルのフォーマットを指定するオプションを追加。(--input-format)
・flv出力などを行う際に長時間(6時間37分以上)エンコードすると、timestampがオーバーフローし
  "Application provided invalid, non monotonically increas ing dts to muxer in stream"
  というエラーで正常にmuxできなくなる問題を修正。
・2.46以降、--avsync forcecfrを使用すると"Failed to get free surface for vpp."でエラー終了する問題を修正。
・ffmpegのswデコーダを使用するモードを追加。(--avsw)

2016.06.23 (2.53)
[共通]
・API 1.19対応のドライバでvpp-rotateが使用できなくなっていた問題を修正。

[QSVEncC]
・GPU使用率をより簡単に表示可能に。
  「GPU使用率を表示可能に.bat」を[右クリック→管理者として実行]するだけで表示可能に。
・Linux向けビルドを修正。

2016.06.18 (2.52)
・エラーメッセージの不備を修正。
・簡易インストーラを更新。

2016.06.12 (2.51)
[共通]
・API v1.19に対応。
・vppによるリサイズの品質を指定するオプションを追加。
  --vpp-scaling <string>  simple, fine
・vppによる鏡像反転のオプションを追加。
  --vpp-mirror <string>   h, v
  "v"なら縦方向、"h"なら横方向。
・HEVCエンコード時に、ピラミッド参照の階層ごとにQPのオフセット値を指定する--qp-offsetを追加。

2016.05.19 (2.50)
[共通]
・2.49でインタレ解除すると、例外で落ちてしまうのを修正。

2016.05.18 (2.49)
[共通]
・一部環境で、デフォルトでも"Failed to initialize encoder. : invalid video parameters."で死んでしまうのを修正。
・2.46以降、--scenechangeやVQPが正常に動作していなかったのを修正。

2016.05.04 (2.48)
[QSVEncC]
・2.46以降、Windows10のビデオなど、一部のプレイヤーで再生できないmp4が出力される問題を修正。
・複数の動画トラックがある際に、これを選択するオプションを追加。(--video-track, --video-streamid)
  --video-trackは最も解像度の高いトラックから1,2,3...、あるいは低い解像度から -1,-2,-3,...と選択する。
  デフォルトは--video-track 1、つまり最も高い解像度のものを対象とする。
  --video-streamidは動画ストリームののstream idで指定する。

2014.04.29 (2.47)
[QSVEncC]
・--vpp-subで画像タイプの字幕も焼き込めるように。
  AVX2 / AVX / SSE4.1 / SSE4.1 pshufb slow対応。
・--vpp-subがGPUメモリモードでも動作可能にして大幅に高速化。
・--vpp-subで焼き込む字幕がないときの処理を大幅に高速化。
・--vpp-subのデフォルトの並列数を2→3へ。そのほうが高速。
・--async-depthのデフォルト値をすこし減らした。メモリを喰う原因となっていた。
・--avsync forcecfr + trimに対応。
・SkylakeでサポートされたVP8/VP9デコードを追加。(hybridらしい)

2016.04.24 (2.46)
[QSVEncC]
・libassを使用して字幕を焼きこむ機能を追加。(--vpp-sub <int> or <string>)
  整数指定の場合は、入力動画ファイルの指定された字幕トラックを抽出して焼きこむ。
  文字列指定の場合は、入力動画ファイルとは別の字幕ファイルを読み込み、これを焼きこむ。
  AVX2 / AVX / SSE4.1 / SSE4.1 pshufb slowの4モードから最適なものが自動的に選択される。
  以下3点に注意。
  ・テキスト形式の字幕のみの対応。
  ・--sub-copyとは併用できない。
  ・systemメモリモードが必須のため、d3d11モードを要求する機能(--vpp-rotate等)とは併用できない。
・libassのshapingを設定するオプションを追加。(--vpp-sub-shaping)
  simple(デフォルト)かcomplex。
・字幕の文字コードを指定するオプションを追加。(--vpp-sub-charset)
  指定しない場合は自動。日本語についてはある程度自動でもよいかも。
  指定する場合には下記を参照。
  https://trac.ffmpeg.org/attachment/ticket/2431/sub_charenc_parameters.txt
・H.264入力でも--avsync forcecfrが使用可能に。
・--check-featuresでテキストで出力すべき時にもHTMLで出力されていた部分があったのを修正。

2016.04.20 (2.45v2)
・簡易インストーラを更新。

2016.04.15 (2.45)
[QSVEncC]
・--audio-copyの際のエラー回避を追加。

2016.04.03 (2.44)
[QSVEncC]
・Haswell以降でvpp-detail-enhanceの効きが50で固定になっていたのを改善。
・x64版で、MFX/GPU使用率を取得できるように。リモートデスクトップ中は取得できない。
・Linuxでビルドできなくなっていたのを修正。
・コマンドラインのエラー時のメッセージを改善。

2016.03.31 (2.43)
[QSVEncC]
・音声関連ログの体裁改善とフィルタ情報の追加。
・音声フィルタをトラック別に指定可能なように。
・音声フィルタを適用すると不必要なログが表示される問題を修正。
・QSVが使用可能かのチェックと、使用できない場合の対策を提示するバッチファイルを追加。

2016.03.27 (2.42)
[QSVEncC]
・音声フィルタリングを可能に。 (--audio-filter)
  dllを含めて更新してください。
  音量変更の場合は、"--audio-filter volume=0.2"など。
  書式はffmpegの-afと同じ。いわゆるsimple filter (1 stream in 1 stream out) なら使用可能なはず。
・avsync forcecfr + trimは併用できないので、エラー終了するように。
・HEVCエンコード時にもmux可能に。
・HEVCエンコード時に連続Bフレーム数のデフォルトを2にしていたのを3に戻した。
  また、連続Bフレーム数を3以上にした場合の警告を表示しないようにした。
  最近のドライバでは問題ない模様。

2016.03.19 (2.41)
[QSVEncC]
・2.40で修正しきれていなかった音ズレを修正。
・出力バッファサイズ(--output-buf)のデフォルトを8MBに減らす。
  かえってパフォーマンスが低下するという報告があったため。

2016.03.17 (2.40)
[QSVEncC]
・一部入力ファイルで音ズレが発生するのを修正。
・フレームレート推定を改善。
・ベンチマークの結果ファイルに入力ファイル名を追記。
・--perf-monitorで得られる情報を追加。

[QSVEnc]
・簡易インストーラを更新。

2016.03.13 (2.39)
[共通]
・2.37以降、エンコード開始時にフリーズしてエンコードが進まなくなってしまう問題を修正。
・音声トラックが見つからない場合のエラーメッセージを修正。動作には影響なし。
・エンコード終了時のログが不正確であることがあったのを修正。

[QSVEnc]
・簡易インストーラ更新。

[QSVEncC]
・2.38以降、エンコード開始時にフリーズしてエンコードが進まなくなってしまう問題を修正。
・コマンドラインパースエラーが分かりにくい場合があったのを改善。
・ベンチマークモードが動作しないのを修正。

2016.03.08 (2.38)
[共通]
・API 1.4以下のマシンで機能情報取得(--check-features)が不正確だったのを修正。

[QSVEncC]
・音声トラックがない場合に、エラー終了させず、エンコードを続行するオプションを追加。(--audio-ignore-notrack-error)
・使用できないレート制御モードが指定された場合に、エラー終了するのではなく、
  自動的により一般的にサポートされるレート制御モードにフォールバックするオプションを追加。(--fallback-rc)
  ビットレート指定系なら最終的にvbrを、品質指定系なら最終的にcqpを使用する。
・--avsync forcecfrは--vpp-deinterlace it/bobとは併用できないことへのエラーメッセージを追加。
・パラメータの値を自動的に丸めた場合は、警告を表示するように。
・レート制御モードがサポートされていないのか、コーデックがサポートされていないかをわかりやすく表示。
・読み込み用スレッドを追加。--input-thread <int>でオンオフ可能。
・初期化処理を大きく変更し、簡略化。初期化を大幅に高速化。

2016.02.29 (2.37)
[QSVEncC]
・メモリ解放漏れを修正。
・2.27以降、
  "Failed to SynchronizeFirstTask"
  "Failed to get free surface for vpp pre."
  などのエラーで終了してしまうことがある問題を修正。

2016.02.24 (2.36)
[QSVEncC]
・--seekの効かないケースがあったのを修正。
・--avsync forcecfrはMPEGデコード時しか動作しないが、それ以外の場合はエラー終了ではなく、
  警告を表示して無効化の上続行するように。

2016.02.21 (2.35)
[QSVEncC]
・--avsync forcecfrで、最初から少し音ずれしている場合があったのを修正。
・--avsync forcecfrはH.264デコード時には使用できないようにした。(当面MPEG2デコード専用)
・H.264デコード中に落ちる場合があったのを修正。

2016.02.20 (2.34)
[共通]
・2.32以降、HEVCエンコードができなかったのを修正。

[QSVEncC]
・QSVの処理速度に上限を設けたいときの設定を追加。(--max-procfps)
  デフォルトは0 (制限なし)。複数本QSVエンコードをしていて、ひとつのストリームにCPU/GPUの全力を奪われたくないというときのためのオプション。
・映像と音声の同期を保つためのオプションを追加。(--avsync)
  現在は、through, forcecfrモードのみ実装。
  through(デフォルト)はこれまで同様、入力はCFRを仮定し、入力ptsをチェックしない。
  forcecfrでは、入力ptsを見ながら、CFRに合うようフレームの水増し・間引きを行い、音声との同期が維持できるようにする。
  主に、入力がvfrやRFFなどのときに音ズレしてしまう問題への対策。
  vfrに対して使用する際には、合わせてエンコードしたいfpsを明示的に--fpsで与えてください。
・シークしてからエンコードを開始するオプションを追加。(--seek)
  高速だが不正確なシークをしてからエンコードを開始する。正確な範囲指定を行いたい場合は従来通り--trimで行う。
  書式は、hh:mm:ss.ms。"hh"や"mm"は省略可。
・コマンドラインでエラーとなった時の情報表示を強化。
・入力fps判定をさらに改善。RFFでも誤判定しないように。
・主に低解像度向け処理を高速化。

2016.02.15 (2.33)
[QSVEnc]
・ゼロ除算例外(0xc0000094)を修正。

[QSVEncC]
・音声のデコードエラーを無視して処理を継続するようにした。エラーの箇所は無音に置き換える。
  具体的には、連続するデコードエラーの数をカウントし、閾値以内ならエラーを無視して処理を継続する。
  閾値を--audio-ignore-decode-error <int>で設定する。デフォルトは10。
  0とすれば、1回でもデコードエラーが起これば処理を中断してエラー終了する。

2016.02.13 (2.32)
[共通]
・エンコード開始時の機能チェックの際に、初期化済みsessionを使うことでチェックを高速化。
・Sandybridgeでd3dメモリでVPPを使用する際に、正常に動作するにもかかわらず、エラーメッセージが表示されるのを修正。

[QSVEnc]
・bob化の際、進捗表示が200%になっていたのを修正。
・bob化の際、fpsが倍にならなくなっていた問題を修正。

[QSVEncC]
・音声関連オプションを複数指定する際に、--audio-streamを使用すると同一trackへの指定として扱われないのを修正。
・ffmpegのdllが存在しないときに--audio-streamを使用すると例外で落ちてしまうのを修正。
・出力ファイルのフォルダが存在しないとエラー終了するのを修正。
・--fade-detectが効いていなかったのを修正。
・helpを整理。
・helpにvpp-denoise, vpp-detail-enhanceの値の範囲を明記。
・Linux向けがコンパイルできなかったのを修正。

2016.02.10 (2.31)
[共通]
・2.29以降、ffmpegのdllがない場所で実行するとエンコード終了時に落ちていたのを修正。
  AviutlのQSVEnc.auoでは0xc06d007e例外(不明なアプリケーション例外)[kernelbase.dll]で落ちていた。

2016.02.09 (2.30)
[QSVEncC]
・hls出力で、m3u8ファイルが正常に出力されないのを修正。
・ffmpeg_dllを更新。udp読み込み時に問題があったのを修正する。--avqsv-analyzeも特に指定する必要がなくなった。
  下記のようにすれば問題なく動作する。
  -i udp://127.0.0.1:1234?pkt_size=262144^&fifo_size=8000000 -o test.mp4
・チャプターファイルを読み込むオプションを追加。(--chapter <string>)
  nero形式とapple形式に対応する。
  --chapter-copyとは併用できない。

2016.02.05 (2.29)
[QSVEncC]
・ffmpeg_dllを更新。速度最適化(-O3)に切り替え、音声エンコード時などに高速化。
  また、networkやprotocolsを有効にしてビルドした。これにより、udpのリアルタイムエンコードなどが可能。
  -i udp://127.0.0.1:1234?pkt_size=262144^&fifo_size=8000000 -o test.mp4 --output-thread 0 -a 1 --avqsv-analyze 10
・dllのバージョンを表示するオプションを追加。(--check-avversion)
・サポートされているプロトコルを表示するオプションを追加。(--check-protocols)
・mux時にオプションパラメータを渡すオプションを追加。(-m <string1>:<string2>)
  主にHttp Live Streaming出力時に必要なパラメータを渡すために使用する。
  例として、HLS用の出力を行いたい場合には、以下のように設定する。
  -i <input> -o test.m3u8 -f hls -m hls_time:5 -m hls_segment_filename:test_%03d.ts --gop-len 30
・libavcodec/libavformatからのエラーメッセージをログファイルに書き出せるようにした。
  これまではコンソールにしか表示されていなかった。
・音声のサンプリング周波数を変換する機能を追加。(--audio-samplerate [<int>?]<int>)
・音声のサンプリング周波数変換時に使用するエンジンを切り替えるオプションを追加。(--audio-resampler <string>)
  選択肢は"swr"(デフォルト=swresampler)と"soxr"(libsoxr)。
・トラックを指定して、音声チャンネルの分離・統合などを行うオプションを追加。(--audio-stream [<int>?][<string>])
  典型的にはデュアルモノ音声などに対し、--audio-stream FR,FLなどとして分離する。
  また同時に、音声のチャンネル数を指定するのにも使用することができ、--audio-stream stereoなどとすることで常に音声を2chに変換したりできる。
  
  音声チャンネルの分離・統合などを行う。
  --audio-streamが指定された音声トラックは常にエンコードされる。
  ,(カンマ)で区切ることで、入力の同じトラックから複数のトラックを生成できる。

  書式:
  <int>に処理対象のトラックを指定する。
  <string1>に入力として使用するチャンネルを指定する。省略された場合は入力の全チャンネルを使用する。
  <string2>に出力チャンネル形式を指定する。省略された場合は、<string1>のチャンネルをすべて使用する。

  例1: --audio-stream FR,FL
  最も必要だと思われる機能。デュアルモノから左右のチャンネルを2つのモノラル音声に分離する。

  例2: --audio-stream :stereo
  どんな音声もステレオに変換する。

  例3: --audio-stream 2?5.1,5.1:stereo
  入力ファイルの第２トラックを、5.1chの音声を5.1chとしてエンコードしつつ、ステレオにダウンミックスしたトラックを生成する。
  実際に使うことがあるかは微妙だが、書式の紹介例としてはわかりやすいかと。

  使用できる記号
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

2016.01.25 (2.28)
[共通]
・音声エンコード速度より動画のエンコード速度が速くなる場合に、メモリ使用量が大きく膨れ上がっていたのを修正。
・mux時の同期に問題がある場合があったのを修正。

[QSVEncC]
・perf-monitor-plotをpyqtgraphベースに変更。
  以前のmatplotlibベースより高速。
  python3.4以降 + pyqtgraph (+ numpy, PySide)が必要に。

2016.01.17 (2.27)
[共通]
・GPUデバイスの取得とGPUメモリ取得のあたりのログ情報・エラー情報を細かく取得できるようにした。

[QSVEnc]
・AuoLinkが使用不可である場合にも、AuoLink関連のタブが見えていたのを修正。
・簡易インストーラでQuickTimeがダウンロードできなくなっていたのを修正。

[QSVEncC]
・出力バッファサイズを指定するオプションを追加。(--output-buf)
  出力バッファサイズをMB単位で指定する。デフォルトは64、最大値は128。0で使用しない。
  これまで常に64MB確保していたのを変更できるようにする。
・出力スレッドを使用しないオプションを追加。(--no-output-thread)
  出力スレッドはエンコードを高速化する一方、それなりにメモリを消費する。
  そこでメモリ節約のため、出力スレッドを使用しないオプションを追加。
  デフォルトでは出力スレッドを使用する。
・メモリ使用量を最小化するオプションを追加(--min-memory)。
  現時点では、"-a 1 --no-output-thread --output-buf 0 --input-buf 1"と同じ。

2015.12.30 (2.26)
[QSVEncC]
・mkv出力時にSAR比が反映されないのを修正。
・GPU-Zが起動していれば、GPU使用率を取得できるように。
・--audio-source使用時にも--trimを使用できるように。
・avqsv以外のリーダーでも--trimを使用できるように。

2015.12.24 (2.25)
[QSVEncC]
・avqsvリーダーで指定されたfpsで読み込むように。
・ts終端などの中途半端なパケットにより音声のヘッダparser部分でエラーが発生した際に、
  処理が中断され、正常なファイルが出力できない問題を修正。

2015.12.20 (2.24)
[QSVEnc]
・AuoLinkと組み合わせて、avqsvを使用できるように。

[QSVEncC]
・エンコードパイプライン内で発生したエラーのエラーコードが正常に回収されないことがあるのを修正。

2015.11.24 (2.23)
[QSVEncC]
・HaswellでHEVCがデコードできないのを修正。

2015.11.20 (2.22)
[共通]
・API v1.17に対応。
・フェード検出に対応。(--fade-detect)
・画像回転に対応。90°, 180°, 270°に対応。(--vpp-rotate)
・Skylake Pentiumで、Broadwellと判定されてしまうのを修正できている…かもしれない。

[QSVEnc]
・設定画面でCBRが選択できなくなっていたのを修正。

[QSVEncC]
・mkvなどで時間解像度が粗かったり、割り切れなかったりする場合に
  シークがうまくいかない(シーク時に音がしばらく再生されない)問題を修正する。
・ヘルプの誤字を修正。

2015.11.15 (2.21)
[共通]
・qpmin/qpmaxを指定すると落ちてしまう問題を修正。

[QSVEncC]
・x64版で--check-environmentを使用すると、落ちてしまう問題を修正。
・ベンチマーク時に測定する品質設定の対象を変更できるように。またデフォルトを「すべて」でなく「1,4,7」に変更。
・パフォーマンス分析を行うオプションを追加。(--perf-monitor)
・音声エンコード時の速度を大幅に改善。(Windowsのみ)

2015.11.02 (2.20)
[共通]
・rdseedのフラグ位置を1bit間違っていたので修正。
  BroadwellがHaswellと検出されていた。
・100%で進捗が停止してしまう問題を修正。

[QSVEnc]
・fdk-aac (ffmpeg)にもaudio delay cut用のパラメータをQSVEnc.iniに追加。
・libmfxsw32.dllのチェックがまだ残っていたので削除。

[QSVEncC]
・tsなどで音声をエンコードする際に、最終パケットが中途半端だとフリーズしてしまう問題を修正。

2015.10.30 (2.19)
[QSVEnc]
・libmfxsw32.dllの読み込み時エラーが問題になっているので、ひとまずswでのエンコードを無効化した。
・ログ表示をより詳細に。

[QSVEncC]
・いつかのコマンドラインのパースがうまくいかない問題を修正。 
・m2ts/tsのVC-1がデコードできないのを改善。
・m2tsなどにおける字幕の取り扱いを改善。
・音声をmuxしない場合、字幕もmuxされなかった問題を修正。
・デバッグ用ログ出力を強化。 
・内部的な様々な修正。

2015.10.16 (2.18)
[QSVEnc]
・更新なし

[QSVEncC]
・音声を抽出・コピー・エンコードした場合に映像にブロックノイズが乗ることがあるのを修正。

2015.10.11 (2.17)
[QSVEnc]
・重み付きBフレームの設定画面での挙動が怪しいのを修正。
・設定画面を一度閉じてから開けると例外が発生することがあったのを修正。

[QSVEncC]
・更新なし

2015.10.10 (2.16)
[共通]
・API 1.16に対応。
・重み付きBフレーム、重み付きPフレームのオプションを追加。
  --weightb, --weightp。

[QSVEncC]
・コピーする字幕を選択できるように。
  --sub-copyで字幕の番号を1,2,3,...で指定する。
  何も指定しない場合、今まで通りすべての字幕をコピーする。
・VC-1 hw decodeを追加。
・--disable-d3dが効かなくなっていた問題を修正。
・メモリの指定が自動の場合、出力コーデックがrawなら、systemメモリを自動的に使用するように。
  そのほうが圧倒的に高速。

2015.10.03 (2.15)
[QSVEncC]
・UTF-8 plain textな字幕にも対応してみた。

2015.10.03 (2.14)
[QSVEncC]
・古いdllをチェックしていて、動作しなかったのを修正。
・ヘルプのlevel, profileの表記でH.264が2回表示されていたのを修正。

2015.10.03 (2.13)
[共通]
・H.264 Level 5.2を追加。

[QSVEncC]
・チャプターをコピーする機能を追加。--chapter-copy
・字幕をコピーする機能を追加。--sub-copy
・動画メタ情報をコピーするように。
・ヘルプに指定可能なlevel, profileを表示可能に。
・その他、ヘルプの修正。

2015.09.02 (2.12)
[共通]
・VC++2015に移行。
・HEVCエンコード時は「連続Bフレーム数」のデフォルトを2に。
  3以上にすると、ブロック状に崩れる現象が一部のデコーダで発生するため。
・一部ログ表示を改善・修正。

[QSVEnc]
・VC++2015移行に合わせ、簡易インストーラを更新。
  
[QSVEncC]
・2.11のx64版で--check-featuresが正常に出力されない問題を修正。

2015.08.26 (2.11)
[共通]
・Skylake HEVCに対応。(HWエンコード)
  ただし、まだ不安定な可能性がある。
  また、いまのところmuxしながらの出力には非対応。

[QSVEncC]
・VP8に対応しようとしたが、pluginがないと言われ動かなかった。
・--check-featuresを改良。
  --check-features <出力ファイル名>とすることで、指定したファイルに出力し、
  出力後、既定のアプリケーションでそれを開く。
  出力ファイル名が".html"ならhtml形式、".csv"ならcsv形式、それ以外は従来通りのtxt形式で出力する。
・--audio-sourceを追加。--audio-copyとの併用で、外部音声ファイルをmuxできる。

2015.08.20 (2.10)
[共通]
・SkylakeのFixed Funcを使用した完全HWエンコを行うオプションを追加した。
  QSVEnc: FixedFunc, QSVEncC: --fixed-func

2015.08.13 (2.09)
[共通]
・Win10で実行した場合に、例外0xc0000005で落ちる問題を修正。
・Skylake世代の判定を追加。
・GPU情報をOpenCL経由で取得できなかった場合の表記を改善。

[QSVEncC]
・d3d11モードで、vpp-delogoを使用すると異常終了する問題を修正。
・--level指定で、一部正しく指定できなかったのを修正。
・--darオプションを追加。
・--format rawを指定してもraw出力できなくなっていた問題を修正。

2015.08.03 (2.08)
[共通]
・OSバージョンの確認方法を変更。

[QSVEnc]
・ログウィンドウが裏に隠れてしまうことがある問題を修正。

[QSVEncC]
・--avsync-depthオプションを追加。
・オプション名を--lookahead-dsから--la-qualityに改名。
  lookahead関係はほかはla-xxxなのに、これだけlookaheadで筋が悪かった。
・エンコードせずに出力するモードを追加。-c rawを指定する。
  QSVデコード、QSV VPP、QSVデコード+VPPなどのみを稼働させてy4mで出力することができる。これにより
    QSVEncC.exe -i <avsファイル> -o - -c raw --tff --vpp-deinterlace bob | x264 --demuxer y4m -o <出力mp4ファイル> -
    QSVEncC.exe --avqsv -i <mp4ファイル> -o - -c raw --tff --vpp-deinterlace bob | x264 --demuxer y4m -o <出力mp4ファイル> -
  などの実行が可能になる。
・--profileを指定した場合のエラーを修正。

2015.07.21 (2.07)
[QSVEnc]
・音声をエンコードしないとmuxされないのを修正。

2015.07.20 (2.06)
[共通]
・API v1.15に対応。
・Skylake HEVC エンコードに仮対応。
・IvyBridgeではピラミッド参照をデフォルトでオフにし、
  オンが指定された場合にも警告を表示するように。

[QSVEncC]
・ffmpeg_lgpl_dllのビルドをgcc 5.1.0に。
  swresample-1.dllを追加。
・MPEG2 hwエンコードに対応。
・--check-featuresでコーデック別の対応表を作成。
・音声のエンコードに対応。これに伴い、--audio-codec, --audio-bitrateを追加。
・--mv-scalingのヘルプを修正。
・使用可能なフォーマット、コーデックを確認可能に。
  --check-codecs, --check-encoders, --check-decoders, --check-formats
・大きなtimebaseの場合に、avgDurationの計算で32bit整数が桁あふれしておかしなfps値になるのを修正。
・y4m reader使用時にログ表示がおかしかったのを修正。

2015.07.11 (2.05)
[共通]
・デバッグ用出力を追加。

[QSVEncC]
・--audio-fileで正常に出力できない問題を修正。

2015.07.06 (2.04)
[QSVEnc]
・プロファイルを選択すると落ちるのを解消。

2015.07.05 (2.03)
[QSVEncC]
・ファイル名が出力側で文字化けする問題を修正。

2015.07.05 (2.02)
[QSVEncC]
・avqsvリーダーもunicodeファイル名に対応。
・--vpp-delogo, --vpp-half-turnがavqsvリーダー使用時以外に使用できなかった問題を修正。

2015.07.05 (2.01)
[QSVEnc]
・.NET Framework 4.5に移行。
・Windows10を正しく検出できるように。
・qaacでのALACモードがmux出来なかったのを修正。
・設定画面のサイズが大きくなることがあったのを修正。

[QSVEncC]
・avqsvリーダーの標準入力からの読み込みに対応。
・--mux-videoオプションを廃止。
・--format <string>で、muxしながら出力する際のフォーマットを指定可能に。
  指定しない場合は、拡張子から自動的に判断する。
  "raw"を指定することで従来通りH.264/ESで出力する。
・muxしながら出力する際に、標準出力に出力できるように。
・vppとして透過性ロゴフィルタを追加。--vpp-delogo-file等でロゴファイルを指定する。".lgd",".ldp",".ldp2"に対応。
  ロゴパックの場合は、--vpp-delogo-selectでロゴ名を指定するか、自動選択用iniファイルを指定する。
  その他のオプションとして、
   > --vpp-delogo-posで1/4画素精度のロゴ位置の調整
   > --vpp-delogo-depthで透明度の補正
   > --vpp-delogo-y, --vpp-delogo-cb, --vpp-delogo-crで各色成分の補正
  処理はCPUでSSE4.1, AVX, AVX2により行われる。
  Aviutl用同様、YC48-12bitで処理されるが、色差成分は4:2:0で処理する。
  また、最終的に8bitに丸めるため、高い計算精度は必要ないので、いくつかの処理を端折って高速化している。

2015.06.28 (v2.00β12)
[QSVEncC]
・ヘルプの--lookahead-dsの引数が誤って記述されていたのを修正。
・ヘルプの--trellisの引数が誤って記述されていたのを修正。
・--lookahead-dsの値が、trellisに反映されてしまうバグを修正。
・Windows10を正しく検出できるように。
・y4m読み込みができなくなっていた問題を修正。
・一部のmpegファイルで進捗が正しく表示されていなかった問題を修正。

2015.06.22 (v2.00β11)
[QSVEncC]
・2.00β10でmuxしながらのインタレ保持エンコができなくなっていたのを修正。
・2.00β10でもまだ音ズレが発生することがあったのへの対策。

2015.06.21 (v2.00β10)
[QSVEncC]
・mkv/flvエンコード時に音ズレが発生することがあったのへの対策。
・一部のMPEG2ファイルで、正しくエンコードできない場合があったのを修正。

2015.06.14 (v2.00β9)
[QSVEncC]
・リサイズ時には、SAR比の自動反映を行わないように。
・音声が途中から始まる場合に、音ズレする可能性が高かったのを修正。
・音声が映像より長い場合にtrimを指定していなくても動画に合わせて短くなってしまう問題を修正。
・mkvなどで、音ズレする問題を修正。
・--copy-audioで音声ファイルが数字のみの場合、ファイル名を正常に読み取れないのを修正。
・入力がflvなどの場合にデコードできない問題を修正。

2015.06.07 (v2.00β8)
[QSVEncC]
・インタレ保持出力をmuxした際の再生互換性を向上。
・--audio-fileでコピーする音声トラックの指定、複数指定を可能に。
・avqsv + VQPには非対応なことをメッセージに明示。
・いくつかのエラーメッセージを追加。
・m2tsなどのac3音声が適切に抽出できないのを改善。
・m2tsなどのPAFFの取り扱いを改善。

2015.05.31 (v2.00β7)
[QSVEncC]
・動画をmuxした際の再生互換性を向上。
・--copy-audioで複数音声トラックがあってもコピーできるように。
  また--copy-audioでコピーする音声トラックを選択できるように。
・フレームレートの推定を改善。
・QSVでデコードできないコーデックを入力した際のエラー処理を改善。
・進捗表示の精度を向上。

2015.05.24 (v2.00β6)
[QSVEncC]
・2.00β5でavqsvリーダー以外動かなくなっていたのを修正。

2015.05.24 (v2.00β5)
[QSVEncC]
・mp4/mkv/movへのmuxを追加。出力ファイルの拡張子で自動的に有効に。強制する場合は--mux-video。
・--copy-audioで音声もmux。
・ts系での音ズレを改善。(RFF以外、RFFは非対応です!)
・2.00β3でもH.264 PAFFを入れるとフレームレートが倍になることがあったのを修正。

2015.05.21 (v2.00β4)
[QSVEncC]
・フレームレート推定の精度を改善(主にts)。
・vppの10bit→8bit色変換テスト用に、vapoursynthからhigh bit depthで読み込めるように。
・Broadwellで、HEVC 10bitのデコードが正常に動作しない問題を修正。

2015.05.18 (v2.00β3)
[QSVEncC]
・avqsvリーダーにH.264 PAFFを入れるとフレームレートが倍になる問題を解決。
・avqsvリーダーを使用時に--cropを使用すると、"undefined behavior"で落ちる問題を修正。
・不適切なコマンドラインが指定された際に、きちんとエラーとして処理されていなかった問題を修正。
・GPU情報の取得を改善。
・フレームレート推定の精度を改善。
・PCM音声のwav出力が正常に行われないことがあったのを改善。

2015.05.16 (v2.00β2)
[QSVEncC]
・HEVC in mp4のデコードに対応。
・DVD-Video/BlurayなどのPCM音声でもwav出力できるように。
・avqsvリーダーでも進捗を表示できるように。

2015.05.12 (v1.34)
[QSVEnc]
・音声エンコ前後にバッチ処理を行う機能を追加。
・1.31以降、bob化した映像をmuxするとフレームレートが半分になっていた問題を修正。

2015.05.10 (v2.00β)
[QSVEncC]
・QSVでデコードからエンコードまでを一貫して行う事ができるようにした。
  MPEG2, H.264, HEVCのデコードをサポート(ただしHEVCはraw formatのみ対応)。
  --avqsv, --audio-file, --trimを追加。
  まだ実験段階でいろいろ不安定なので、テストするだけにしてください。

2015.04.05 (v1.33)
[共通]
・4156ドライバで1.31以降インタレ解除ができない問題で、
  1.31の更新を取り消して対処。

2015.03.21 (v1.32)
[QSVEncC]
・エンコード結果が出力されないことがあるのを修正。
・ログのb-pyramidの表示を改善。
・--vpp-deinterlaceが効かなくなっていたのを修正。

2015.03.07 (v1.31)
[共通]
・API v1.13に対応。
  - ダイレクトモード最適化(--direct-bias-adjust)とMVコスト調整(--mv-scaling)を追加。
  - 新たなインタレ解除モードを追加。(自動、自動(bob)、24fps化(固定))
・ログのb-pyramidの表示を改善。
・ログのQP上限/下限の表示を改善。

2015.03.04 (v1.30)
[共通]
・QSVEncのビットレートの上限(65535kbps)を撤廃。

2015.02.19 (v1.29v2)
[QSVEnc]
・簡易インストーラで、QuickTimeの抽出が正常に実行されないのを修正。
  setupフォルダに7z.exe/7z.dllがなかったため。

2015.02.16 (v1.29)
[QSVEnc]
・簡易インストーラによるインストールで、
  VC++2005 Runtimeがインストールされていないと、qaacが動かない問題を修正。
[QSVEncC]
・vpyリーダー使用時にフリーズする可能性があったのを修正。

2015.02.08 (v1.28)
[QSVEnc]
・自動フィールドシフト使用時以外で、muxを一工程削減。

2014.11.20 (1.27)
[共通]
API v1.11に対応。
・LA_HRD (先行探索レート制御 (HRD互換)) モードに対応。(--la-hrd)
・QVBR (品質ベース可変レート制御) モードに対応。(--qvbr)
・先行探索レート制御に使用可能な「ウィンドウベースレート制御」を追加。 (--la-window-size)
・VppのImage Stablizerを追加。効果の程は謎。(--vpp-image-stab)
・機能情報表示を拡充。
・ログ表示を改善。

2014.11.10 (1.26)
[共通]
・OSのバージョン情報をログに表示するように。
・エンコード中のCPU使用率を表示するように。
[QSVEnc]
・x264guiEx 2.23までの機能追加に追従
  - デフォルトの音声エンコーダを変更する機能を追加。
  - タスクバーへの進捗表示で一時停止が反映されないことがあるのを解消。
  - qaacとfdk-aacについて、edtsにより音声ディレイのカットをする機能を追加。
  - muxerのコマンドに--file-formatを追加。
    FAWを使用した際に、apple形式のチャプターが反映されない問題を解決。
  - 音声やmuxerのログも出力できるように。
  - 0秒時点にチャプターがないときは、ダミーのチャプターを追加するように。
    Apple形式のチャプター埋め込み時に最初のチャプターが
    時間指定を無視して0秒時点に振られてしまうのを回避。
  - flacの圧縮率を変更できるように。
  - ログにmuxer/音声エンコーダのバージョンを表示するように。
  - 音声エンコーダにopusencを追加。
[QSVEncC]
・ベンチマーク時にメモリの実速度を測定し、表示するように。
・高精度タイマーをQSVEncCからも利用可能に。デフォルトでオン。
  どうしてもオフにしたい場合は--no-timer-period-tuningを使ってください。

2014.07.01 (1.25)
[共通]
・不安定な以下の条件でのエンコードを自動的に回避するようにした。
  - API v1.8以降、Lookahead系 + scenechangeは不安定(フリーズ)
  - Lookahead系でのbframes >= 10 + b-pyramidは不安定(フリーズ)
  - b-pyramid + scenechangeは不安定(画像崩壊)

2014.06.27 (1.24)
[共通]
・SandybridgeなどAPI v1.6未満の環境で、インタレ保持エンコができなくなっていた問題を修正。

2014.06.25 (1.23)
[共通]
・QSVの機能チェックを改善
  - ICQが使用可能か、正しく判定されるように
  - 各モードがインタレ対応か、チェックするように
・エンコード情報表示を改良
  - ICQ時にビットレートが表示されていたのを修正
  - GPU情報とドライバのバージョンも表示するように
[QSVEnc]
・機能情報にGPU名とドライバ番号を表示
・CBR,VBR以外での最大ビットレートの指定は無効に
[QSVEncC]
・色変換部分でAVX2に対応
・avsリーダーもYUY2/RGB24/RGB32読みに対応

2014.06.21 (1.22)
[共通]
・Lookaheadモードとシーンチェンジ検出を併用すると固まってしまうことがあるので、
  Lookaheadモード使用時にはシーンチェンジ検出を強制的に無効化するようにした。
[QSVEnc]
・nero形式のチャプターをUTF-8に変換する機能を追加。その他の設定から。

2014.04.01 (1.21)
[QSVEnc]
・1.19以降で「Aviutlのプロファイル」に保存した設定が
  1.18以前と互換性がなくなっていた問題を修正。

2014.03.28 (1.20)
[QSVEnc]
・faw2aac使用時など、muxerのみで一発でmuxが可能な場合に、
  チャプターがmuxされない問題を修正
  また、これに合わせてmux情報表示を改善

2014.03.25 (1.19)
[QSVEnc]
・音声エンコードディレイをカットする機能を追加 (音声カットのみ)
[QSVEncC]
・ベンチマークログの改善
  - 可能ならCPUのBoostクロックを表示
  - GPUの情報取得を改良
  - GPUのドライババージョンを表示

2014.03.07 (1.18)
[共通]
・1.16以降、ファイルが正常に出力されないことがあったのを修正。
[QSVEnc]
・ログ自動保存の保存場所を変更する機能を追加。

2014.03.06 (1.17)
[QSVEncC]
・ベンチマークモードの出力ファイルを調整。
  - GPU情報も表示するように。
  - レイアウト調整。
  - 使用メモリ量が、残りメモリ量になっていたのを修正。
・Vapoursynth Readerをx64対応に。
・1.08以降、vpy読みでエンコードを中断するとクラッシュするのを修正。

2014.03.04 (1.16)
[共通]
・1.12以降、Win8 + dGPUでiGPUから出力していなくても
  QSVエンコ出来る機能が使えなくなっていたのを修正。
  Intel Media SDK 2014でビデオメモリモード周りのフラグの意味が
  変わっていたのに対応できていなかった。
[QSVEncC]
・CQPモードのベンチマークモードを追加(--benchmark)。
・1.12以降、エクスプローラから見えるバージョン情報が
  x86版でもx64と表示されていたのを修正。

2014.03.01 (1.15)
[共通]
・1.12以降、LookaheadDepthが正しく反映されない問題を修正。
[QSVEnc]
・1.12以降、Sandy環境など、API v1.6をサポートしない環境で、
  設定画面を開く際にフリーズする問題を解決…したつもり。
[QSVEncC]
・リダイレクトした場合でも進捗情報がすぐに読み取れるようflushするように。

2014.02.22 (1.14)
[QSVEnc]
・1.12以降、設定画面を開くのに時間がかかっていたのを少し改善。
・指定したログ保存場所が存在しないとエラーで終了してしまっていたのを修正。

2014.02.20 (1.13)
[共通]
・Sandy環境など、API v1.6をサポートしない環境で
  正常に動作しない問題を解決…したつもり。

2014.02.18 (1.12)
[共通]
・Intel Media SDK 2014 ベースに移行、API v1.8に対応。
・libmfxsw32.dll / libmfxsw64.dll を更新。
・開発環境をVC++ 2013 Express for Desktopに移行。
・環境がサポートしている機能チェックを強化
  - QSVEncでは機能表示タブに表示
  - QSVEncCでは--check-featuresにより確認できる。
・ハードウェアエンコでも色設定が可能に
  - colormatrix / colorprim / transfer
・OpenGOPオプションを追加。
・API v1.6の機能を追加。
  - マクロブロック単位のレート制御
  - 拡張レート制御 (ビットレート指定モードの時のみ)
・API v1.8の機能を追加。
  - レート制御モード追加
    > 固定品質モード
    > 先行探索付き固定品質モード
    > ビデオ会議モード
  - 適応的Iフレーム挿入
  - 適応的Bフレーム挿入
  - Bフレームピラミッド参照
  - 先行探索品質設定 (3段階)
・その他いろいろやった気もするが忘れた
[QSVEnc]
・経過時間を表示
・ログウィンドウ右クリックから一時停止できるように
・バッチ出力時の安定性をすこし改善。

2013.12.07 (1.11v3)
・簡易インストーラを更新
  - 簡易インストーラをインストール先のAviutlフォルダに展開すると
    一部ファイルのコピーに失敗する問題を修正
    
2013.11.24 (1.11v2)
[QSVEnc]
・簡易インストーラを更新
  - L-SMASHがダウンロードできなくなっていたのを修正。
  - インストール先が管理者権限を必要とする際は、
    これを取得するダイアログを表示するようにした。
    
2013.10.19 (v1.11)
[QSVEnc]
・変更したフォントの(標準⇔斜体)が保存されない問題を修正。
・設定ファイルのメモが保存されない問題を修正。
・簡易インストーラを更新
  - Windows 8.1に対応したつもり
  - アップデートの際にプリセットを上書き更新するかを選択できるようにした。
[QSVEncC]
・更新なし

2013.09.12 (v1.10)
[共通]
・入力がインターレースとして設定されていない場合にVPPのインタレ解除を設定すると、おかしな事になるのを修正。
[QSVEncC]
・Unicodeに対応(基本的にワイド文字列で処理)。
・ログをファイルに保存できるようにした。追記型。(--log <ファイル名>)

2013.08.25 (v1.09)
[共通]
・1.08で、「d3d11を試さないように」というのが徹底されていなかった問題を修正。
・d3d11は基本的にはd3d9より遅いようなので、必要な時以外はd3d9を使うようにした。
[QSVEncC]
・--d3d9/--d3d11 オプションにより、(使用可能なら)強制的にモードを設定できるようにした。

2013.08.21 (v1.08)
[共通]
・Win7ではd3d11を試さないように。
[QSVEncC]
・avs/vpy readerのエラー処理を改善。
・vpyマルチスレッドモードを追加(--vpy-mt)。
  VapourSynthの示すスレッド数を使用する(ただし最大127まで)。

2013.08.05 (v1.07)
[共通]
・Win8 + dGPUでiGPUから出力していなくても、QSVを利用できるようにした。
  BIOS(UEFI)で[iGPU Multi-Monitor]をEnabledにすることで使用可能。
・d3d11モードに対応した。
[QSVEncC]
・1.05以降、y4m経由で正しく動作しなくなっていた問題を修正。
  ご指摘ありがとうございました。
・Ctrl + Cで中断した際にも途中までの情報を表示するようにした。

2013.08.02 (v1.06)
[共通]
・vppにより「通常」あるいは「Bob化」のインタレ解除を行う場合に、
  シーンチェンジ検出・VQPを使用可能にした。
・インタレ保持エンコではLookaheadモードを使用できないというメッセージを表示するようにした。
[QSVEnc]
・配布プロファイルを見直し
  いくつかの設定例を追加。
[QSVEncC]
・x86版にVapoursynth r19用readerを追加。拡張子vpyで自動でVapoursynth Readerを使用。
  vfw読みより若干(～5%)高速。
  yv12読み込み専用。
  yv12以外(yuy2,RGB等)の場合は自動的にavi(vfw)読みに変更。
  
2013.07.20 (v1.05v2)
[QSVEncC]
・ヘルプ表示を修正。

2013.07.20 (v1.05)
[QSVEncC]
・Avisynth Readerを追加。拡張子avsで自動でAvisynth Readerを使用。
  vfw読みより若干(～10%)高速。

2013.07.13 (v1.04)
[QSVEnc]
・Lookaheadモード時に最大ビットレートの指定ができなかった問題を修正。
[QSVEncC]
・Lookaheadモードの指定に対応。

2013.07.08 (v1.03)
・デバッグ用のログファイル出力が行われていたのを修正。

2013.07.07 (v1.02)
[QSVEnc]
・新しいプロファイルを保存しようとするとエラーが出る問題を修正。
  ご指摘感謝します。
・プロファイルを少し見直し。
[QSVEncC]
・特に変更なし。

2013.07.03 (v1.01)
[共通]
・Intel Media SDK 2013 ベースに移行、API v1.7に対応。
・libmfxsw32.dll / libmfxsw64.dll を更新。
・指定しても効果のないAPI v1.6の機能を削除。
  - マクロブロック単位のレート制御
  - 拡張レート制御 (ビットレート指定モードの時のみ)
・API v1.7の機能を追加。
  - 先行探索レート制御 (lookahead)
  - 歪みレート最適化 (trellis)
[QSVEncC]
・qualityオプションが正しく読めない問題を修正。

2013.07.01 (v1.00)
[共通]
・開発環境をVC++ 2012 Express for Desktopに移行。
・あわせて.NET Framework 4.0 Client に移行。
・動作環境からWin Vistaを外した。Intel Media SDK 及びドライバの対応が微妙なため。
・色空間変換・シーンチェンジ検出などでAVX/AVX2に対応。
  いつもどおり自動的に最速のものを選択。
・Intel Media SDK 2013 ベースに移行、これによりAPI v1.6に対応。
・libmfxsw32.dll / libmfxsw64.dll を更新。
・以下のAPI v1.6の機能を追加。
  - マクロブロック単位のレート制御
  - 拡張レート制御 (ビットレート指定モードの時のみ)
・品質設定を3段階から7段階に拡張。
・v1.00にあわせて簡易インストーラを更新。
・シーンチェンジ検出時に段階的なフェードシーンで
  フリッカのようになってしまう問題を対策。
[QSVEnc]
・ログウィンドウで出力ファイル名を右クリックから
  「動画を再生」「動画のあるフォルダを開く」機能を追加。
[QSVEncC]
・--qualityオプションの変更。
  best, higher, high, balanced(default), fast, faster, fastestの7段階。

2013.05.23 (v0.23v2)
[共通]
・Blurayモード強制を追加。
  QSVEnc.confの[QSVEnc]セクションに
  force_bluray=1
  と追記してください。

2013.05.12 (v0.23)
[共通]
・エンコード情報表示を改善。
[QSVEncC]
・AVI(vfw)読み込みに対応。
  - avi/avs/vpyなど、vfw経由で読み込めるものについて、直接読めるようにした。
  - 拡張子avi/avs/vpyで自動的にavi読み込みに切り替え。明示的な指定は--avi。
  - YV12/YUY2/RGB24/RGB32読み込み対応、YV12推奨。
  - YV12/YUY2読み込みについてはインタレ対応。
  - RGB24/RGB32読み込みはvppにてRGB32->NV12変換を行うため、インタレ非対応。
  - 進捗状況と残り時間を表示。
・最後にエンコードにかかった時間を表示。

※AvisynthのAPIを直接叩く方法は、とある理由により実装する予定はありません。

2013.05.05 (v0.22)
[共通]
・vppによるインタレ解除にbob化(60fps化)を追加。
・最大GOP長を"0"とすることでfps×10を自動的に設定するようにした。

2013.05.05 (v0.21)
[QSVEnc]
・映像一時ファイルが残ってしまう問題を修正。
[QSVEncC]
・y4m入力時にアスペクト比を読み取るようにした。

2013.05.04 (v0.20)
[共通]
・HWエンコ時の入力フレームバッファのデフォルトを4→3。
・内部のパイプラインバッファ数を調整し、少し高速化。
[QSVEnc]
・yuy2→nv12変換を調整してわずかに高速化。
・x264guiEx 1.75までの更新を反映。
  - mux時にディスクの空き容量の取得に失敗した場合でも、警告を出して続行するようにした。
  - 設定画面で「デフォルト」をクリックした時の挙動を修正。
  - 音声設定のパイプ - 2passのチェックボックスの挙動を修正。
  - エンコ前後バッチ処理を最小化で実行する設定を追加。
[QSVEncC]
・フレーム読み込み時のyv12→nv12変換を調整して高速化。
・cropオプションを追加。
・エンコード後の結果表示が崩れていたのを修正。

2013.03.08 (v0.19v2)
[共通][QSVEnc]
 なし。
[QSVEncC]
・y4mのカラーフォーマットの指定形式を追加。

2013.02.14 (v0.19)
問題の報告ありがとうございました。
[共通]
・0.15以降、Baseline Profileでのエンコードが失敗する問題を修正。
[QSVEncC]
・ヘルプでsarが二重に表示されていたのを修正。

2013.01.31 (v0.18)
[共通]
・0.17で、一部の環境で "undefined behavior" と出てエンコードが始まらない問題を修正。
  報告ありがとうございます。
・x86のシーンチェンジ検出・可変QP調整計算をさらに高速化。

2013.01.26 (v0.17)
[共通]
・インタレ保持エンコード時に、
  0.15以降、あるいは0.14以前で固定長GOPにチェックを入れていた場合に、
  周期的に破綻するフレームが出ていたのを修正。
  エラー報告に感謝致します。
・シーンチェンジ検出・可変QP調整計算を高速化。

2013.01.23 (v0.16)
[共通]
・シーンチェンジ検出・可変QPが意図と違った動作を指定していたのを修正。
  画面の下半分しか見ていなかった。

2013.01.22 (v0.15)
[共通]
・シーンチェンジ検出による強制Iフレーム挿入機能を追加。
・可変QPモード追加。
  両方とも入力がプログレッシブ(非インタレ)の時のみ有効。
・エンコード後、フレームタイプごとの総サイズを表示。
[QSVEnc]
・自動フィールドシフト使用時に不安定だった問題を修正。
・その他の設定にタイマー精度を向上させる設定を追加。
・エンコ前後バッチ処理を最小化で実行する設定を追加。
  その他の設定から。

2012.12.26 (v0.14)
[QSVEnc]
・出力ファイルの種類のデフォルトを変更できるようにした。
  その他の設定から。反映には設定後Aviutlの再起動が必要。

2012.12.22 (v0.13)
[QSVEnc]
・自動フィールドシフト対応。
  L-SMASH muxer用iniファイル(auoと一緒に入っているほう)を使用してください。
  mp4box用iniファイルでは動きません。
・x264guiEx 1.65までの更新を反映。
  - ログウィンドウの位置を保存するようにした。
  - 高Dpi設定時の表示の崩れを修正。
  - エンコード開始時にクラッシュする可能性があるのを修正。

2012.11.15 (v0.12)
[QSVEnc]
・x264guiEx 1.62までの更新を反映。
  - ログウィンドウの色の指定。
  - 高DPI時の表示崩れ修正。

2012.11.02 (v0.11)
[QSVEnc]
・muxerコマンドにfps指定を追加。
・x264guiEx 1.61までの更新を反映。
  - 音声エンコ / muxerのメッセージ取り込みとエラー表示。
  - 映像・音声同時エンコード時の表示の改善。
  - ログウィンドウ透過率の指定。

2012.10.20 (v0.10)
[共通]
・v0.08以降、Intel iGPUがプライマリGPU(メインモニタに出力しているGPU)でない場合に、
  ビデオメモリモードが使用できず、error: null pointer というエラーを発生させていた問題を修正。
・colormatrix, coloprim, transferの設定は、HWエンコ(QSV)では効かないので、
  設定画面やヘルプの表示をそのように変更。
[QSVEnc]
・Bluray互換出力のチェックボックスが変更チェックから外れていたのを修正。
・x264guiEx 1.59までの更新を反映。
  - エンコ"前"バッチ処理を追加。
  - ブログへのリンクを追加。

2012.10.13 (v0.09)
[共通]
・Bluray向け出力設定を追加。
[QSVEnc]
・Bluray出力用プリセットを追加。

2012.10.06 (v0.08)
[共通]
・Intel Media SDK 2012 R3 (API v1.4)に対応。
  - Windows8 + DirectX 11.1対応、ということのようだ。
  - Win7では特に意味は無い?
[QSVEnc]
・簡易インストーラを追加。
・x264guiEx 1.57までの更新を反映。
  - 映像と音声の同時処理モードを追加。
    音声処理順の選択肢として"後","前","同時"。
  - 実行ファイルを指定するボタンを右クリックすると、
    現在指定中の実行ファイルのhelpを表示できたり。
  - QSVEnc.iniで、音声/muxファイル名を記述するところで、2つ以上書けるように
    具体的には、
    filename="ffmpeg.exe;avconv.exe"
    みたく、どちらでも利用できるように。
  - インタレi420変換で、これまでの単純平均でなく、3,1-加重平均を使用。
  - 音声の長さと動画の長さが大きく異なる場合に警告を出すようにした。
  - 1/2サイズのFAWCheckで、384kbpsを超えるAACの場合にnon-FAWと誤判定するのを修正。
  - 音声設定にflac / fdk-aac (ffmpeg/avconv)の設定を追加。
[QSVEncC]
・特になし。

2012.07.08 (v0.07)
[共通]
・v0.06で追加したオプション類をソフトエンコの時のみ有効とした。
  - ハードウェアエンコ(QSV)ではどうも反映されないので。
・MVC関連の使用していないコードを省略
・その他細かい調整
[QSVEnc]
要QSVEnc.ini更新
・L-SMASH対応
  - muken氏にL-SMASH muxerを更新していただき、PAFF H.264/ESをインポートできるようにしていただきました。
    ありがとうございます!
  - QSVEncでは、L-SMASH用iniファイルを基本に変更します。
    mp4boxに代わり、L-SMASH muxer / remuxerを指定してください。
  - L-SMASH rev600以降を使用してください。
  - 一応mp4box用iniも入れておきますが…。
・x264guiEx 1.46～1.49までの更新を反映。
  - faw2aac.auo対応
    > faw2aacの進捗をログウィンドウにも表示。
    > faw2aac.auoがあれば、fawclがなくてもFAW処理できるように。
  - L-SMASH対応 (x264guiExと同じコードを使用)
  - 細かい問題の修正
    > 実行ファイルを開く... 画面で、デフォルトのフォルダがおかしくなることがあるのを修正。
    > 保存していた"stg設定ファイルの場所"のルートが存在しない場合に、エラーを吐くのを修正。
  - 表示の修正
    > mux時のエラーメッセージで、音声ファイルがないのでmuxできない場合に、
      映像ファイルがないという表示になっていたのを修正。
  - %{chpath}の置換が一時ファイル名を元に作られていたのを、出力ファイル名からに修正。
[QSVEncC]
・とくになし。

2012.05.11 (v0.06)
[QSVEnc]
・x264guiEx 1.42の更新の一部を反映。
 - WinXPで相対パスが正常に取得されない問題への対策。
   WinXPだと、相対パスを取得するときに使うPathRelativePathTo関数が、".\dir"を返すべきときに"\dir"を返すことがあるようで、
   QSVEncはXPは対象外だが、念のため、回避のためのコードを追加した。
 - QSVEnc.iniのqtaaacenc/qaac用設定にCVBRモードの設定を追加した。
   順番的に自然な、ABRとTVBRの間に挿入したので、
   設定ファイルのインデックスがずれることがあるかもしれない。
[QSVEncC]
・v0.05でy4m + パイプが動かない問題を修正。

2012.05.05 (v0.05)
[共通]
・Intel Media SDK 2012 R2 (API v1.4)に対応。
  - libmfxsw**.libの更新(API v1.4)
  - API v1.4に特に重要な更新はない。
・vppの動作を修正。
・情報表示の改善。また表示するエンコード情報の追加。
・これまで「品質」での自動設定しかできなかった設定項目の追加。
  ただし、ある程度「品質」の設定に左右されるので、そのとおりになるとは限らない。
  - MV探索範囲、MV探索精度の設定。
  - CAVLC/CABACの設定。
  - 歪みレート最適化(RDO, 要CABAC)の設定。
  - フレーム間予測/フレーム内予測時のブロックサイズの設定。
・プロファイル指定時の動作を改善。
・その他多くの細かい改善・変更。もう忘れた。
[QSVEnc]
※要QSVEnc.ini更新
※設定ファイルの互換性が一部ありません。(SAR比等)
  もう一度再設定をお願いします。
・SARをmux時でなく、エンコ時に指定するようにした。
・SAR比をmp4boxで再指定するオプションを追加した。
・x264guiEx 1.34～1.41の更新のうち、以下のものを反映
  - 設定ファイル(stgファイル)の表示を、フォルダ構造を反映して表示できるようにした。
    あわせて、設定ファイルの新規保存もフォルダを指定できるようにした。
  - 設定保存処理を改良。
  - 簡易ビットレート計算機で、動画の長さをフレーム数 + フレームレートでも計算できるようにした。
    時分秒 <-> フレーム数は下のボタンで切り替えられる。
  - chapterファイルをmuxすることを選択し、しかしchapterファイルが存在しない時に、
    L-SMASH remuxerやmkvmergeを使用したmuxでも、とりあえずmuxは通るようにした。
  - 「ファイルを開く...」などから得られるパスを、相対パスにする設定を追加。
    その他の設定から。
[QSVEncC]
・オプションの追加とhelpの追加・修正。
[現状の問題点] (Intel 2696ドライバ from Windows Update, API v1.3)
・インタレ保持エンコが事実上できない。(倍フレーム数問題)
・hwエンコ(QSV)でcolormatrix,colorprim,transferの指定が効かない。
  (colour_description_present_flag = 0 (h264_parse))
・hwエンコ(QSV)でシーンチェンジ検出できない。
  実装してみて、ソフトウェアモードでは動くのだが、hwエンコでうまく動かないので、無効化してある。
  参考: http://software.intel.com/en-us/forums/showthread.php?t=103385

2012.02.27 (0.04v2)
[QSVEncC]
・一部のコマンドラインが正常に動かないのを修正。
・x86のバイナリなのかx64のバイナリなのか、
  右クリック > プロパティから確認できるようにした。

2012.02.23 (v0.04)
[共通]
・Intel Media SDK 2012に仮対応 (API v1.3に仮対応)
  - 色設定の追加
  - AVBRモードの追加
  - libmfxsw**.libの更新(API v1.3)
  - APIバージョンの検出と表示 (QSVEncでは設定画面に、QSVEncCでは--lib-checkで)
[QSVEnc]
・x264guiEx 1.27～1.33の更新を反映
  - 安定性の向上。
  - ログウィンドウの大きさを保存できるようにした。
  - STAThreadAttributeを指定。
  - muxを行っていない場合に「エンコ後バッチ処理」に失敗するのを修正。
  - 録画後バッチ処理のバッチファイル指定欄のドラッグドロップ対応。
  - 音声エンコで単純なWAV出力に対応した。
  - 設定ファイル保存処理の改良。
  - ファイルサイズ取得の改良。
  - 設定画面とログウィンドウでフォントを変更できるようにした。
  - QSVEnc.iniのoggenc2コマンドラインを修正。強制的に44.1kHzになってしまっていた。
  - QSVEnc.iniにqaac/refalac用設定を追加。
  - QSVEnc.iniにAnonEncoder用設定を追加。
[QSVEncC]
・x86版も静的リンク。(QSVEncCだけなら、VC++2008 再頒布可能パッケージ不要)
・-o - でstdout(標準出力)に出せるようにした。

2012.01.22 (v0.03v2)
・設定画面が出ない状態を解消。

2012.01.22 (v0.03)
要iniファイル更新(iniファイルバージョン1→2)
・別スレッドでフレーム読み込み。ちょっぴり高速化するはず。
  フレーム読み込みバッファ機能を追加。バッファサイズは1～16フレーム。部分並列化効率を向上させる。
  QSVEncでは読み込みバッファサイズ(タブ3枚目)で、QSVEncCでは--input-bufで大きさを指定できる。
  あまり多すぎると逆に遅くなるので注意。(キャッシュサイズとの関連だと思う)
  エンコ速度  適切なバッファサイズ
  ～50fps         1～2
  ～100fps        2～3
  ～200fps        3～6
  それ以上        4～8
  エンコ速度が速ければ速いほど効果があるはず。
  逆にfullHDのエンコードなどではたいして効果はない。
  QSV使用時にバッファサイズをあんまり大きくするとGPUメモリを確保できなくなってこけるので注意。
・x264guiEx v1.11～v1.26の追加機能の取り込みとバグ修正の反映
  ・相対パスを使用できるようにした。
  ・ツールチップヘルプの抑制(その他の設定から)
  ・「このウィンドウを最小化で開始」が一回しか効力を発揮せず、解除されてしまう問題を修正。
  ・視覚効果をオフにできるようにした。
  ・プロファイルにメモを残せるようにした。プロファイルの右側に表示。ダブルクリックで変更できる。
  ・QSVEnc.iniが存在しない、あるいは古い時にエラーメッセージの前に例外が発生する問題を解決。
  ・Apple系に対応したmp4/chapterをmp4boxを用いても出力できるようにした。
  ・その他の設定にログウィンドウ関連の設定(「透過」と「最小化で開始」)を追加。
  ・設定ファイルのサイズが異なっても、Aviutl側のプロファイルに保存された設定も読めるようにした。
  ・チャプターファイルの自動削除をオンオフできるようにした。その他の設定から。
  ・チャプターファイルが存在しない場合でもとりあえずmuxを成功させるようにした。
  ・エンコ中でないとき、Escキーでログウィンドウを閉じるようにした。
  ・エンコ後バッチファイル実行を追加。
  ・設定画面でEscキーのオンオフ設定を追加。
  ・muxエラー対策。mux時チェックを根本的に改善。
・QSVEncCにx64版を追加。わずかに速い…かもしれない。基本たいして変わらない。
  まあx64ビルドもできるよ、ということ。
・不要なメモリ解放->再確保を防止。

2012.01.16 (QSVEncC_20120116)
・コンソールへの出力をきちんとstderrとstdoutに整理。基本stderr。

2011.10.02 (v0.02)
・Intel GPU がプライマリGPUでない場合でもQSVが使用できるようにした。
  またその時d3dメモリモードが使用出来ない問題を解決。
  (d3dメモリモードでないとvppのパフォーマンスが低下する)
・QSVEncCがわけのわからんエラーメッセージを吐くのを改善。

2011.09.27 (v0.01)
・fpsの表示がおかしい問題を修正。エンコに影響なし。

2011.09.26 (v0.00)
・公開版

2011.09.26
・自動ログ保存の場所指定追加。

2011.09.25
・たくさんバグ修正。
・エンコードしたフレームタイプの内訳をログに表示。

2011.09.24
・いろいろメッセージ追加。

2011.09.23
・slices設定項目追加。
・コマンドライン版を作ってみた。QSVencC。
  まあBonTSDemuxに対するBonTSDemuxCみたいなもん。
  パイプ入力やってみたかっただけ。まあでもこれでAvisynthでも使えるはず。

2011.09.22
vpp SceneChangeDetectionがよくわからん。よって対応せず。

2011.09.20
vppインタレ解除対応(ITの使い方がよくわからん)

2011.09.19
ぶっちゃけ、色変換(YUY2->NV12)だけをhw vppでやっても遅くなるだけということが0.02でわかった。
しょうがないからその他のvpp(Resize,Denoise,DetailEnhancer)も追加(どういう方向性だ…)

2011.09.19
hw vpp (YUY2 -> NV12) 対応。
GPU EU処理で早くなるか、あるいはメモリコピー増加で遅くなるか
→遅くなったorz

2011.09.18
Bフレーム設定、GOP長、d3d mem modeなどを追加。
シークできなくなる条件が解明。
YUY2->NV12を直接変換に改良。

2011.09.18
動くよ。

2011.09.17 (on sample_encode.exe)
インタレ保持 + hw encode は変。