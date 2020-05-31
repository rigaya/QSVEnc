
# QSVEncC option list

**[日本語版はこちら＞＞](./QSVEncC_Options.ja.md)**


## Command line example


### Basic commands
```Batchfile
QSVEncC.exe [Options] -i <filename> -o <filename>
```

### More practical commands
#### example of using hw (qsv) decoder
```Batchfile
QSVEncC --avhw -i "<mp4(H.264/AVC) file>" -o "<outfilename.264>"
```

#### example of using hw (qsv) decoder (interlaced)
```Batchfile
QSVEncC --avhw --interlace tff -i "<mp4(H.264/AVC) file>" -o "<outfilename.264>"
```

#### avs (Avisynth) example (avs and vpy can also be read via vfw)
```Batchfile
QSVEncC -i "<avsfile>" -o "<outfilename.264>"
```

#### example of pipe usage
```Batchfile
avs2pipemod -y4mp "<avsfile>" | QSVEncC - y4m - i - - o "<outfilename.264>"
```

#### pipe usage from ffmpeg

```Batchfile
ffmpeg -y -i "<inputfile>" -an -pix_fmt yuv420p -f yuv4mpegpipe - | QSVEncC --y4m -i - -o "<outfilename.264>"
```

#### passing video & audio from ffmpeg
--> use "nut" to pass both video & audio thorough pipe.
```Batchfile
ffmpeg -y -i "<input>" <options for ffmpeg> -codec:a copy -codec:v rawvideo -pix_fmt yuv420p -f nut - | NVEncC --avsw -i - --audio-codec aac -o "<outfilename.mp4>"
```

## Option format

```
-<short option name>, --<option name> <argument>

The argument type is
- none
- <int>    ... use integer
- <float>  ... use decimal point
- <string> ... use character string

The argument with [] brackets are optional.

--(no-)xxx
If it is attached with --no-xxx, you get the opposite effect of --xxx.
Example 1: --xxx: enable xxx → --no-xxx: disable xxx
Example 2: --xxx: disable xxx → --no-xxx: enable xxx
```

## Display options

### -h, -? --help
Show help

### -v, --version
Show version of QSVEncC

### --option-list
Show option list.

### --check-hw
Check whether the specified device is able to run QSVEnc.

### --check-lib
Show the API version of Media SDK installed on the system.

### --check-features
Show the information of features supported.

### --check-features-html [&lt;string&gt;]
Output the information of features supported to the specified path in html format.
If path is not specified, the output will be "qsv_check.html".

### --check-environment
Show environment information recognized by QSVEncC.

### --check-codecs, --check-decoders, --check-encoders
Show available audio codec names

### --check-profiles &lt;string&gt;
Show profile names available for specified codec

### --check-formats
Show available output format

### --check-protocols
Show available protocols

### --check-filters
Show available audio filters

### --check-avversion
Show version of ffmpeg dll

## Basic encoding options

### -c, --codec &lt;string&gt;
Specify the output codec
 - h264 (default)
 - hevc
 - mpeg2
 - raw

### -o, --output &lt;string&gt;
Set output file name, pipe output with "-"

### -i, --input &lt;string&gt;
Set input file name, pipe input with "-"

Table below shows the supported readers of QSVEnc. When input format is not set,
reader used will be selected depending on the extension of input file.

**Auto selection of reader**  

| reader |  target extension |
|:---|:---|          
| Avisynth reader    | avs |
| VapourSynth reader | vpy |
| avi reader         | avi |
| y4m reader         | y4m |
| raw reader         | yuv |
| avhw/avsw reader | others |

**color format supported by reader**  

| reader | yuv420 | yuy2 | yuv422 | yuv444 | rgb24 | rgb32 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| raw | ○ |  |  |  |  |  |
| y4m | ◎ |  | ◎ | ◎ |  |  |
| avi | ○ | ○ |  |  | ○ | ○ |
| avs | ◎ | ○ | ◎ | ◎ |  |  |
| vpy | ◎ |  | ◎ | ◎ |  |  |
| avhw | ◎ |  |  |  |  |  |
| avsw | ◎ |  | ◎ | ◎ | ○ | ○ |

◎ ... 8bit / 9bit / 10bit / 12bit / 14bit / 16bit supported  
○ ... support only 8 bits

### --raw
Set the input to raw format.
input resolution & input fps must also be set.

### --y4m
Read input as y4m (YUV4MPEG2) format.

### --avi
Read avi file using avi reader.

### --avs
Read Avisynth script file using avs reader.

### --vpy
### --vpy-mt
Read VapourSynth script file using vpy reader.

### --avsw
Read input file using avformat + ffmpeg's sw decoder.

### --avhw [&lt;string&gt;]
Read input file using avformat + QSV hw decoder. Using this mode will provide maximum performance,
since entire transcode process will be run on the GPU.

**Codecs supported by avhw reader**  

| Codecs | Status |
|:---|:---:|
| MPEG1      | ○ |
| MPEG2      | ○ |
| H.264/AVC  | ○ |
| H.265/HEVC | ○ |
| VP8        | × |
| VP9        | ○ |
| VC-1       | × |
| WMV3/WMV9  | × |

○ ... supported  
× ... no support

### --interlace &lt;string&gt;
Set interlace flag of **input** frame.

Deinterlace is available through [--vpp-deinterlace](#--vpp-deinterlace-string). If deinterlacer is not activated for interlaced input, then interlaced encoding is performed.

- none ... progressive
- tff ... top field first
- bff ... Bottom Field First

### --crop &lt;int&gt;,&lt;int&gt;,&lt;int&gt;,&lt;int&gt;
Number of pixels to be cropped from left, top, right, bottom.

### --fps &lt;int&gt;/&lt;int&gt; or &lt;float&gt;
Set the input frame rate. Required for raw format.

### --input-res &lt;int&gt;x&lt;int&gt;
Set input resolution. Required for raw format.

### --output-res &lt;int&gt;x&lt;int&gt;
Set output resolution. When it is different from the input resolution, HW/GPU resizer will be activated automatically.

If not specified, it will be same as the input resolution. (no resize)

_Special Values_
- 0 ... Will be same as input.
- One of width or height as negative value    
  Will be resized keeping aspect ratio, and a value which could be divided by the negative value will be chosen.

```
Example: input 1280x720
--output-res 1024x576 -> normal
--output-res 960x0    -> resize to 960x720 (0 will be replaced to 720, same as input)
--output-res 1920x-2  -> resize to 1920x1080 (calculated to keep aspect ratio)
```

## Encode Mode Options

The default is CQP (Constant quantization).

### --cqp &lt;int&gt; or &lt;int&gt;:&lt;int&gt;:&lt;int&gt;
Set the QP value of &lt;I frame&gt;:&lt;P frame&gt;:&lt;B frame&gt;

Generally, it is recommended to specify the QP value to be I &lt; P &lt; B.

### --cbr &lt;int&gt;  (CBR, Constant Bitrate mode)
### --vbr &lt;int&gt;  (VBR, Variable Bitrate mode)
### --avbr &lt;int&gt; (AVBR, Adaptive Variable Bitrate mode)
### --la &lt;int&gt;   (LA, LookAhead mode)
### --la-hrd &lt;int&gt; (LA-HRD, HRD-compliant LookAhead mode)
### --vcm &lt;int&gt; (VCM, Video Conference Mode)
Encode in bitrate(kbps) specified.

### --qvbr &lt;int&gt;, --qvbr-q &lt;int&gt; (QVBR, Quality based VBR mode)
Encode in bitrate specified with "--qvbr", based on quality specified by "--qvbr-quality" (default: 23, lower value => high quality).

### --icq &lt;int&gt; (ICQ, Intelligent Const. Quality mode, default: 23)
### --la-icq &lt;int&gt; (LA-ICQ, Lookahead based ICQ mode: default: 23)
Constant Quality encoding modes. (lower value => high quality)

### --fallback-rc
Enable fallback of ratecontrol mode, when platform does not support new ratecontrol modes.

**Selecting Encode modes**  
CBR, VBR, AVBR are rather basic encoding modes, and although they are fast, the quality tends to be poor.
Using more complex encoding modes, such as ICQ, LA-ICQ, QVBR, LA modes, will result in higher quality.
CQP will be the fastest and will provide stable quality, but with rather large output.

Special encoding, such as encoding for Bluray, requires max bitrate to be set.
In those cases VBR or AVBR must be used, as max bitrate can be set only with those modes. 

## Options for Frame Buffer

**Types of Frame Buffer**  

| OS  | system memory | graphics memory |
|:---|:---:|:---:|
| Windows | system | d3d9 / d3d11 |
| Linux   | system | va           |

Types of Frame Buffer will be set automatically by default as below.

**Windows**  
<u>When using QSV encode:</u>  
As d3d9 memory mode is faster than d3d11 memory mode, QSVEnc will use d3d9 mode whenever possible.
However, in some cases (such as systems with dGPU), d3d9 memory mode is not available. In this case d3d11 memory will be used.

<u>When not using QSV encode (QSV decode only):</u>  
When graphic memory is used, QSV decode will be fast, but sending back frame data from graphics memory to system memory is **very** slow.
Therefore, when you only use QSV decode and pass frame data to other apps, graphics memory will not be used (system memory is used).

**Linux**  
To enhance stability, system memory is used.


### --disable-d3d (Win)
### --disable-va (Linux)
Disable use of graphics memory. (Use system memory.)

### --d3d
Use d3d9 or d3d11 memory mode. (Windows only)

### --d3d9
Use d3d9 memory mode. (Windows only)

### --d3d11
Use d3d11 memory mode. (Windows only)

### --va
Use va memory mode. (Linux only)


## Other Options for Encoder

### --fixed-func
Use only fixed function (fully hw encoding) and not use GPU EU.
In this mode, encoding will be done in very low GPU utilization in low power,
but the quality will be poor compared to ordinary mode.

### --max-bitrate &lt;int&gt;
Maximum bitrate (in kbps).

### --vbv-bufsize &lt;int&gt;
VBV buffersize (in kbps).

### --qvbr-quality &lt;int&gt;
Set quality used in qvbr mode, should be used with --qvbr. (0 - 51, default = 23)

### --avbr-unitsize &lt;int&gt;
Set AVBR calculation period in unit of 100 frames. Default 90 (means the unit is 9000 frames).

### --qp-min &lt;int&gt; or &lt;int&gt;:&lt;int&gt;:&lt;int&gt;
Set the minimum QP value with &lt;I frame&gt;:&lt;P frame&gt;:&lt;B frame&gt;. This option will be ignored in CQP mode. 

It could be used to suppress bitrate being used unnecessarily to a portion of movie with still image.

### --qp-max &lt;int&gt; or &lt;int&gt;:&lt;int&gt;:&lt;int&gt;
Set the maximum QP value to &lt;I frame&gt;:&lt;P frame&gt;:&lt;B frame&gt;. This option will be ignored in CQP mode.

It could be used to maintain certain degree of image quality in any part of the video, even if doing so may exceed the specified bitrate.

### --qp-offset &lt;int&gt;[:&lt;int&gt;][:&lt;int&gt;]...
Set qp offset of each pyramid reference layers. (default = 0)

### -u, --quality &lt;string&gt;
Set encoding quality preset.
```
best, higher, high, balanced(default), fast, faster, fastest
```

### --la-depth &lt;int&gt;
Specify lookahead depth in frames. (10 - 100)  
When encoding in interlace mode, the upper limit will be halved to 50.

### --la-window-size &lt;int&gt; 0(auto)
Set bitrate calculation window length in frames.

### --la-quality &lt;string&gt;
Specify quality of lookahead.
- auto (default)
- fast
- medium
- slow

### --mbbrc
Enable per macro block rate control.

### --i-adapt
Enable adaptive I frame insertion.

### --b-adapt
Enable adaptive B frame insertion.

### --strict-gop
Force fixed GOP length.

### --gop-len &lt;int&gt;
Set maximum GOP length. 

### -b, --bframes &lt;int&gt;
Set the number of consecutive B frames.

### --ref &lt;int&gt;
Set the reference distance. In hw encoding, increasing ref frames will have minor effect on image quality or compression rate.

### --b-pyramid
Enable B frame pyramid reference.

### --weightb
Enable weighted B frames.

### --weightp
Enable weighted P frames.

### --direct-bias-adjust
Lower usage of B frame Direct/Skip type.

### --adapt-ltr
Enable adaptive LTR frames.

### --mv-scaling &lt;string&gt;
Set mv cost scaling.
- 0  set MV cost to be 0
- 1  set MV cost 1/2 of default
- 2  set MV cost 1/4 of default
- 3  set MV cost 1/8 of default

### --slices &lt;int&gt;
Set number of slices.

### --level &lt;string&gt;
Specify the Level of the codec to be encoded. If not specified, it will be automatically set.
```
h264: auto, 1, 1 b, 1.1, 1.2, 1.3, 2, 2.1, 2.2, 3, 3.1, 3.2, 4, 4.1, 4.2, 5, 5.1, 5.2
hevc: auto, 1, 2, 2.1, 3, 3.1, 4, 4.1, 5, 5.1, 5.2, 6, 6.1, 6.2
mpeg2: auto, low, main, high, high1440
```

### --profile &lt;string&gt;
Specify the profile of the codec to be encoded. If not specified, it will be automatically set.
```
h264:  auto, baseline, main, high, high444
hevc:  auto, main, main10, main444
mpeg2: auto, Simple, Main, High
```

### --tier &lt;string&gt;  [HEVC only]
Specify the tier of the codec.
```
hevc:  main, high
```

### --sar &lt;int&gt;:&lt;int&gt;
Set SAR ratio (pixel aspect ratio).

### --dar &lt;int&gt;:&lt;int&gt;
Set DAR ratio (screen aspect ratio).

### --colorrange &lt;string&gt;
"auto" will copy characteristic from input file (available when using [avhw](#--avhw)/[avsw](#--avsw) reader).
```
  limited, full, auto
```

### --videoformat &lt;string&gt;
"auto" will copy characteristic from input file (available when using [avhw](#--avhw)/[avsw](#--avsw) reader).
```
  undef, auto, ntsc, component, pal, secam, mac
```
### --colormatrix &lt;string&gt;
"auto" will copy characteristic from input file (available when using [avhw](#--avhw)/[avsw](#--avsw) reader).
```
  undef, auto, bt709, smpte170m, bt470bg, smpte240m, YCgCo, fcc, GBR, bt2020nc, bt2020c
```
### --colorprim &lt;string&gt;
"auto" will copy characteristic from input file (available when using [avhw](#--avhw)/[avsw](#--avsw) reader).
```
  undef, auto, bt709, smpte170m, bt470m, bt470bg, smpte240m, film, bt2020
```
### --transfer &lt;string&gt;
"auto" will copy characteristic from input file (available when using [avhw](#--avhw)/[avsw](#--avsw) reader).
```
  undef, auto, bt709, smpte170m, bt470m, bt470bg, smpte240m, linear,
  log100, log316, iec61966-2-4, bt1361e, iec61966-2-1,
  bt2020-10, bt2020-12, smpte2084, smpte428, arib-std-b67
```

### --chromaloc &lt;int&gt; or "auto"
Set chroma location flag of the output bitstream from values 0 ... 5.  
"auto" will copy from input file (available when using [avhw](#--avhw)/[avsw](#--avsw) reader)
default: 0 = unspecified

### --max-cll &lt;int&gt;,&lt;int&gt; [HEVC only]
Set MaxCLL and MaxFall in nits.  "copy" will copy values from the input file.
```
Example1: --max-cll 1000,300
Example2: --max-cll copy  # copy values from source
```

### --master-display &lt;string&gt; [HEVC only]
Set Mastering display data. "copy" will copy values from the input file.
```
Example1: --master-display G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)
Example2: --master-display copy  # copy values from source
```

### --aud
Insert Access Unit Delimiter NAL.

### --pic-struct
Insert picture timing SEI.

### --bluray [H.264 only]
Perform output for Bluray. (Default: off)

### --repartition-check
Enable prediction from small partitions. [H.264]

### --trellis &lt;string&gt; [H.264]
Set H.264 trellis mode.
- auto(default)
- off
- i
- ip
- all

### --no-deblock
Disable deblock filter. [H.264]

### --tskip
Enable transform skip. [HEVC]

### --sao &lt;string&gt;
Set modes for SAO. [HEVC]
- auto    ... default
- none    ... disable sao
- luma    ... enable sao for luma
- chroma  ... enable sao for chroma
- all     ... enable sao for luma & chroma

### --ctu &lt;int&gt;
Set max ctu size, from 16, 32 or 64. [HEVC]

## IO / Audio / Subtitle Options

### --input-analyze &lt;int&gt;
Specify the length in seconds that libav parses for file analysis. The default is 5 (sec).
If audio / subtitle tracks etc. are not detected properly, try increasing this value (eg 60).

### --trim &lt;int&gt;:&lt;int&gt;[,&lt;int&gt;:&lt;int&gt;][,&lt;int&gt;:&lt;int&gt;]...
Encode only frames in the specified range.

```
Example 1: --trim 0:1000,2000:3000    (encode frame #0 - #1000 and frame #2000 - #3000)
Example 2: --trim 2000:0              (encode frame #2000 to the end)
```

### --seek [&lt;int&gt;:][&lt;int&gt;:]&lt;int&gt;[.&lt;int&gt;]
The format is hh:mm:ss.ms. "hh" or "mm" could be omitted. The transcode will start from the time specified.

Seeking by this option is not exact but fast, compared to [--trim](#--trim-intintintintintint). If you require exact seek, use [--trim](#--trim-intintintintintint).
```
Example 1: --seek 0:01:15.400
Example 2: --seek 1:15.4
Example 3: --seek 75.4
```

### --input-format &lt;string&gt;
Specify input format for avhw / avsw reader.

### -f, --output-format &lt;string&gt;
Specify output format for muxer.

Since the output format is automatically determined by the output extension, it is usually not necessary to specify it, but you can force the output format with this option.

Available formats can be checked with [--check-formats](#--check-formats). To output H.264 / HEVC as an Elementary Stream, specify "raw".

### --video-track &lt;int&gt;
Set video track to encode by resolution. Will be active when used with avhw/avsw reader.
 - 1 (default)  highest resolution video track
 - 2            next high resolution video track
    ...
 - -1           lowest resolution video track
 - -2           next low resolution video track
    ...
    
### --video-streamid &lt;int&gt;
Set video track to encode in stream id.

### --video-tag &lt;string&gt;
Specify video tag.
```
 -o test.mp4 -c hevc --video-tag hvc1
```

### --video-metadata &lt;string&gt; or &lt;string&gt;=&lt;string&gt;
Set metadata for video track.
  - copy  ... copy metadata from input if possible
  - clear ... do not copy metadata (default)

```
Example1: copy metadata from input file
--video-metadata 1?copy

Example2: clear metadata from input file
--video-metadata 1?clear

Example3: set metadata
--video-metadata 1?title="video title" --video-metadata 1?language=jpn
```

### --audio-copy [&lt;int&gt;[,&lt;int&gt;]...]
Copy audio track into output file. Available only when avhw / avsw reader is used.

If it does not work well, try encoding with [--audio-codec](#--audio-codec-intstring), which is more stable.

You can also specify the audio track (1, 2, ...) to extract.

```
Example: Copy all audio tracks
--audio-copy

Example: Extract track numbers #1 and #2
--audio-copy 1,2
```

### --audio-codec [[&lt;int&gt;?]&lt;string&gt;[:&lt;string&gt;=&lt;string&gt;][,&lt;string&gt;=&lt;string&gt;][#&lt;string&gt;=&lt;string&gt;][,&lt;string&gt;=&lt;string&gt;],...]
Encode audio track with the codec specified. If codec is not set, most suitable codec will be selected automatically. Codecs available could be checked with [--check-encoders](#--check-codecs---check-decoders---check-encoders).

You can also specify the audio track (1, 2, ...) to extract.

Also, after ":" you can specify params for audio encoder,  after "#" you can specify params for audio decoder.
```
Example 1: encode all audio tracks to mp3
--audio-codec libmp3lame

Example 2: encode the 2nd track of audio to aac
--audio-codec 2?aac

Example 3: set param "aac_coder" to "twoloop" which will improve quality at low bitrate for aac encoder
--audio-codec aac:aac_coder=twoloop

Example 4: set param "dual_mono_mode" to "main" for audio decoder
--audio-codec aac#dual_mono_mode=main
```

### --audio-bitrate [&lt;int&gt;?]&lt;int&gt;
Specify the bitrate in kbps when encoding audio.

You can also specify the audio track (1, 2, ...) to extract.
```
Example 1: --audio-bitrate 192 (set bitrate of audio track to 192 kbps)
Example 2: --audio-bitrate 2?256 (set bitrate of 2nd audio track to to 256 kbps)
```

### --audio-profile [&lt;int&gt;?]&lt;string&gt;
Specify audio codec profile when encoding audio.

### --audio-stream [&lt;int&gt;?][&lt;string1&gt;][:&lt;string2&gt;]
Separate or merge audio channels.
Audio tracks specified with this option will always be encoded. (no copying available)

By comma(",") separation, you can generate multiple tracks from the same input track.

**format of the option**

Specify the track to be processed by &lt;int&gt;.

Specify the channel to be used as input by &lt;string1&gt;. If omitted, input will be all the input channels.

Specify the output channel format by &lt;string2&gt;. If omitted, all the channels of &lt;string1&gt; will be used.

```
Example 1: --audio-stream FR,FL
Separate left and right channels of "dual mono" audio track, into two mono audio tracks.

Example 2: --audio-stream :stereo
Convert any audio track to stereo.

Example 3: --audio-stream 2?5.1,5.1:stereo
While encoding the 2nd 5.1 ch audio track of the input file as 5.1 ch,
another stereo downmixed audio track will be generated
from the same source audio track.
```

**Available symbols**
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
Specify the sampling frequency of the sound in Hz.
You can also specify the audio track (1, 2, ...) to extract.
```
Example 1: --audio-bitrate 44100 (converting sound to 44100 Hz)
Example 2: --audio-bitrate 2?22050 (Convert the second track of voice to 22050 Hz)
```

### --audio-resampler &lt;string&gt;
Specify the engine used for mixing audio channels and sampling frequency conversion.
- swr ... swresampler (default)
- soxr ... sox resampler (libsoxr)

### --audio-delay [&lt;int&gt;?]&lt;int&gt;
Specify audio delay in milli seconds.

### --audio-file [&lt;int&gt;?][&lt;string&gt;]&lt;string&gt;
Extract audio track to the specified path. The output format is determined automatically from the output extension. Available only when avhw / avsw reader is used.

You can also specify the audio track (1, 2, ...) to extract.
```
Example: extract audio track number #2 to test_out2.aac
--audio-file 2?"test_out2.aac"
```

[&lt;string&gt;] allows you to specify the output format.
```
Example: Output in adts format without extension
--audio-file 2?adts:"test_out2"  
```

### --audio-filter [&lt;int&gt;?]&lt;string&gt;
Apply filters to audio track. Filters could be slected from [link](https://ffmpeg.org/ffmpeg-filters.html#Audio-Filters).

You can also specify the audio track (1, 2, ...) to filter.

```
Example 1: --audio-filter volume=0.2  (lowering the volume)
Example 2: --audio-filter 2?volume=-4db (lowering the volume of the 2nd track)
```

### --audio-disposition [&lt;int&gt;?]&lt;string&gt;[,&lt;string&gt;][]...
音声のdispositionを指定する。

```
 default
 dub
 original
 comment
 lyrics
 karaoke
 forced
 hearing_impaired
 visual_impaired
 clean_effects
 attached_pic
 captions
 descriptions
 dependent
 metadata
 copy

例:
--audio-disposition 2?default,forced
```

### --audio-metadata &lt;string&gt; or &lt;string&gt;=&lt;string&gt;
Set metadata for audio track.
  - copy  ... copy metadata from input if possible (default)
  - clear ... do not copy metadata

```
Example1: copy metadata from input file
--audio-metadata 1?copy

Example2: clear metadata from input file
--audio-metadata 1?clear

Example3: set metadata
--audio-metadata 1?title="audio title" --audio-metadata 1?language=jpn
```


### --audio-ignore-decode-error &lt;int&gt;
Ignore the consecutive audio decode error, and continue transcoding within the threshold specified. The portion of audio which could not be decoded properly will be replaced with silence.

The default is 10.

```
Example1: Quit transcoding for a 5 consecutive audio decode error.
--audio-ignore-decode-error 5

Example2: Quit transcoding for a single audio decode error.
--audio-ignore-decode-error 0
```

### --audio-source &lt;string&gt;[:[&lt;int&gt;?][;&lt;param1&gt;=&lt;value1&gt;][;&lt;param2&gt;=&lt;value2&gt;]...][:...]
Mux an external audio file specified.

**params** 
- copy  
  Copy audio track.

- codec=&lt;string&gt;  
  Encode audio to specified audio codec.

- profile=&lt;string&gt;  
  Specify audio codec profile when encoding audio.

- bitrate=&lt;int&gt;  
  Specify audio bitrate in kbps.
  
- samplerate=&lt;int&gt;  
  Specify audio sampling rate.

- enc_prm=&lt;string&gt;  
  Specify params for audio encoder.

- filter=&lt;string&gt;  
  Specify filters for audio.

```
Example1: --audio-source "<audio_file>":copy
Example2: --audio-source "<audio_file>":codec=aac
Example3: --audio-source "<audio_file>":1?codec=aac;bitrate=256:2?codec=aac;bitrate=192
```

### --chapter &lt;string&gt;
Set chapter in the (separate) chapter file.
The chapter file could be in nero format, apple format or matroska format. Cannot be used with --chapter-copy.

nero format  
```
CHAPTER01=00:00:39.706
CHAPTER01NAME=chapter-1
CHAPTER02=00:01:09.703
CHAPTER02NAME=chapter-2
CHAPTER03=00:01:28.288
CHAPTER03NAME=chapter-3
```

apple format (should be in utf-8)  
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

matroska形式 (hould be in utf-8)  
[Other Samples&gt;&gt;](https://github.com/nmaier/mkvtoolnix/blob/master/examples/example-chapters-1.xml)
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
Copy chapters from input file.

### --sub-source &lt;string&gt;
Read subtitle from the specified file and mux into the output file.

### --sub-copy [&lt;int&gt;[,&lt;int&gt;]...]
Copy subtitle tracks from input file. Available only when avhw / avsw reader is used.
It is also possible to specify subtitle tracks (1, 2, ...) to extract with [&lt;int&gt;].

Supported subtitles are PGS / srt / txt / ttxt.

```
Example: Copy subtitle track #1 and #2
--sub-copy 1,2
```

### --sub-disposition [&lt;int&gt;?]&lt;string&gt;
set disposition for the specified subtitle track.

```
 default
 dub
 original
 comment
 lyrics
 karaoke
 forced
 hearing_impaired
 visual_impaired
 clean_effects
 attached_pic
 captions
 descriptions
 dependent
 metadata
 copy
```

### --sub-metadata &lt;string&gt; or &lt;string&gt;=&lt;string&gt;
Set metadata for subtitle track.
  - copy  ... copy metadata from input if possible (default)
  - clear ... do not copy metadata

```
Example1: copy metadata from input file
--sub-metadata 1?copy

Example2: clear metadata from input file
--sub-metadata 1?clear

Example3: set metadata
--sub-metadata 1?title="subtitle title" --sub-metadata 1?language=jpn
```

### --caption2ass [&lt;string&gt;]
Enable internal caption2ass process. This feature requires Caption.dll.  

**Note:** Pelase always select srt format when muxing to mp4.  

supported formats ... srt (default), ass

### --data-copy [&lt;int&gt;[,&lt;int&gt;]...]
Copy data stream from input file. Available only when avhw / avsw reader is used.

### --attachment-copy [&lt;int&gt;[,&lt;int&gt;]...]
Copy attachment stream from input file. Available only when avhw / avsw reader is used.

### --input-option &lt;string1&gt;:&lt;string2&gt;
Pass optional parameters for input for avhw/avsw reader. Specify the option name in &lt;string1&gt, and the option value in &lt;string2&gt;.

```
Example: Reading playlist 1 of bluray 
-i bluray:D:\ --input-option palylist:1
```

### -m, --mux-option &lt;string1&gt;:&lt;string2&gt;
Pass optional parameters to muxer. Specify the option name in &lt;string1&gt, and the option value in &lt;string2&gt;.

```
Example: Output for HLS
-i <input> -o test.m3u8 -f hls -m hls_time:5 -m hls_segment_filename:test_%03d.ts --gop-len 30
```

### --metadata &lt;string&gt; or &lt;string&gt;=&lt;string&gt;
Set global metadata for output file.
  - copy  ... copy metadata from input if possible (default)
  - clear ... do not copy metadata

```
Example1: copy metadata from input file
--metadata copy

Example2: clear metadata from input file
--metadata clear

Example3: set metadata
--metadata title="video title" --metadata language=jpn
```

### --avsync &lt;string&gt;
  - cfr (default)
    The input will be assumed as CFR and input pts will not be checked.

  - forcecfr
    Check pts from the input file, and duplicate or remove frames if required to keep CFR, so that synchronization with the audio could be maintained.

  - vfr  
    Honor source timestamp and enable vfr output. Only available for avsw/avhw reader, and could not be used with --trim.

## Vpp Options

### --vpp-deinterlace &lt;string&gt;
Activate GPU deinterlacer. 

- none ... no deinterlace (default)
- normal ... standard 60i → 30p interleave cancellation.
- it    ... inverse telecine
- bob ... 60i → 60p interleaved.

### --vpp-denoise &lt;int&gt;
Enable vpp denoise, strength 0 - 100.

### --vpp-mctf ["auto" or &lt;int&gt;]
Enable Motion Compensate Temporal Filter (MCTF), if no param specified, then strength will automatically adjusted by the filter. You can also force filter strength by setting value between 1 (week) - 20 (strong). (default: 0 as auto)

### --vpp-detail-enhance &lt;int&gt;
Enable vpp detail enhancer, strength 0 - 100.

### --vpp-image-stab &lt;string&gt;
Set image stabilizer mode.
- none
- upscale
- box

### --vpp-rotate &lt;int&gt;
Rotate image by specified degree. Degree could be selected from 90, 180, 270. Requires d3d11 mode.

### --vpp-mirror &lt;string&gt;
Mirror image.
- h ... mirror in horizontal direction.
- v ... mirror in vertical   direction.

### --vpp-half-turn &lt;string&gt;
Half turn video image. unoptimized and very slow.

### --vpp-resize &lt;string&gt;
Specify the resizing algorithm.

| option name | desciption |
|:---|:---|
| auto  | auto select |
| simple | use simple scaling |
| fine | use high quality scaling |



    
### --vpp-colorspace [&lt;param1&gt;=&lt;value1&gt;][,&lt;param2&gt;=&lt;value2&gt;],...  
Converts colorspace of the video. 

**parameters**
- matrix=&lt;from&gt;:&lt;to&gt;  
  
```
  bt709, smpte170m
```

- range=&lt;from&gt;:&lt;to&gt;  
```
  limited, full, auto
```

```
example1: convert from BT.601 -> BT.709
--vpp-colorspace matrix=smpte170m:bt709
```

### --vpp-delogo &lt;string&gt;
Specify a logo file. Corresponds to ".lgd", ".ldp", ".ldp2".

### --vpp-delogo-select &lt;string&gt;
For logo pack, specify the logo to use with one of the following.

- Logo name
- Index (1, 2, ...)
- Automatic selection ini file
```
 [LOGO_AUTO_SELECT]
 logo<num>=<pattern>,<logo name>
 ```

 Example:
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
Adjustment of logo position with 1/4 pixel accuracy in X:y direction.

### --vpp-delogo-depth &lt;int&gt;
Adjustment of logo transparency. Default 128.

### --vpp-delogo-y  &lt;int&gt;
### --vpp-delogo-cb &lt;int&gt;
### --vpp-delogo-cr &lt;int&gt;
Adjustment of each color component of the logo.
### --vpp-delogo-add
Add logo.

## Other Options

### --async-depth &lt;int&gt;
set async depth for QSV pipeline. default: 0 (=auto, 4+2*(extra pipeline step))

### --output-buf &lt;int&gt;
Specify the output buffer size in MB. The default is 8 and the maximum value is 128.

The output buffer will store output data until it reaches the buffer size, and then the data will be written at once. Higher performance and reduction of file fragmentation on the disk could be expected.

On the other hand, setting too much buffer size could decrease performance, since writing such a big data to the disk will take some time. Generally, leaving this to default should be fine.

If a protocol other than "file" is used, then this output buffer will not be used.

### --mfx-thread &lt;int&gt;
Set number of threads for QSV pipeline (must be more than 2). 

### --output-thread &lt;int&gt;
Specify whether to use a separate thread for output.
- -1 ... auto (default)
- 0 ... do not use output thread
- 1 ... use output thread  
Using output thread increases memory usage, but sometimes improves encoding speed.

### --min-memory
Minimize memory usage of QSVEncC, same as option set below.
```
--output-thread 0 --audio-thread 0 --mfx-thread 2 -a 1 --input-buf 1 --output-buf 
```

### --(no-)timer-period-tuning
Make finer the system timer period to enhance performance. Enabled by default. Windows only.

### --benchmark &lt;string&gt;
Run benchmark, and output results to file specified.

### --bench-quality "all" or <int>[,<int>][,<int>]...
List of target quality to check on benchmark. Default is "best,balanced,fastest".

### --log &lt;string&gt;
Output the log to the specified file.

### --log-level &lt;string&gt;
Select the level of log output.

- error ... Display only errors
- warn ... Show errors and warnings
- info ... Display general encoding information (default)
- debug ... Output additional information, mainly for debug
- trace ... Output information for each frame (slow)

### --log-framelist
FOR DEBUG ONLY! Output debug log for avsw/avhw reader.

### --max-procfps &lt;int&gt;
Set the upper limit of transcoding speed. The default is 0 (= unlimited).

This could be used when you want to encode multiple stream and you do not want one stream to use up all the power of CPU or GPU.
```
Example: Limit maximum speed to 90 fps
--max-procfps 90
```

### --lowlatency
Tune for lower transcoding latency, but will hurt transcoding throughput. Not recommended in most cases.

### --perf-monitor [&lt;string&gt;][,&lt;string&gt;]...
Outputs performance information. You can select the information name you want to output as a parameter from the following table. The default is all (all information).

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
Specify the time interval for performance monitoring with [--perf-monitor](#--perf-monitor-stringstring) in ms (should be 50 or more). The default is 500.