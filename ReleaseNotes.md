# QSVEnc Release Notes

## 8.04

- Add support for Vapoursynth API V4.
- Add option to encode only when input audio codec differs from codec specified by [--audio-codec](./QSVEncC_Options.en.md#--audio-codec-intstringstringstringstringstringstring). ([--audio-encode-other-codec-only](./QSVEncC_Options.en.md#--audio-encode-other-codec-only))

## 8.03

- Fix [--qp-offset](./QSVEncC_Options.en.md#--qp-offset-intintint) not working for AV1. ( #273 )
- Fix error when encoding H.264 to RTMP/FLV output.
- Fix mkv not created when encoding with [-c](./QSVEncC_Options.en.md#-c---codec-string) av_libsvtav1.
- Work around OpenCL driver issue on Linux Intel iGPU where read_imageui caused "undefined reference to __spirv_ImageSampleExplicitLod_Ruint4" error on runtime compile.
- Fix build error on Arch Linux. ( #276 )

## 8.02

- Add option to fallback to 8bit encoding when 10bit encoding is not supported by the hardware.([--fallback-bitdepth](./QSVEncC_Options.en.md#--fallback-bitdepth))

## 8.01

- Avoid unintended fps values when front of input file is corrupted.
- Improve handling when input files have negative pts.
- Improve quality of burned in subtitles in [--vpp-subburn](./QSVEncC_Options.en.md#--vpp-subburn-param1value1param2value2) processing by changing libass initialization method.
- Improve progress indicator when using [--parallel](./QSVEncC_Options.en.md#--parallel-int-or-param1value1param2value2).
- Add support for using [--parallel](./QSVEncC_Options.en.md#--parallel-int-or-param1value1param2value2) with multiple pipes.
- Update libass.dll
  - libass 0.9.0 -> 0.17.4 (x64), 0.14.0 (x86)
  - harfbuzz 11.4.4 (new)
  - libunibreak 6.1 (new)
- Update rpm build environment to fedora41.

## 8.00

- Update libvpl to 2.15.
  - Add options for mfx_ai_superres in [--vpp-resize](./QSVEncC_Options.en.md#--vpp-resize-string). (superres-mode, superres-algo)
    Note: superres-mode does not seem to work at this time.
- Add feature to use filters with avcodec encoders.
  - Available with ```-c av_xxx```
    Example: [-c](./QSVEncC_Options.en.md#-c---codec-string) av_libsvtav1 [--avcodec-prms](./QSVEncC_Options.en.md#--avcodec-prms-string) "preset=6,crf=30,svtav1-params=enable-variance-boost=1:variance-boost-strength=2"
    Other usable options include av_libvvenc, av_libvpx-vp9, etc.
- Add QSVEnc.auo2 with official support for AviUtl2.
- Update ffmpeg libraries. (Windows)
  - ffmpeg 7.1+ (20240822) -> 8.0
  - libpng 1.6.44 -> 1.6.50
  - expat 2.6.2 -> 2.7.1
  - fribidi 1.0.11 -> 1.0.16
  - libogg 1.3.5 -> 1.3.6
  - libxml2 2.12.6 -> 2.14.5
  - libvpl 2.13.0 -> 2.15.0
  - libvpx 1.14.1 -> 1.15.2
  - dav1d 1.4.3 -> 1.5.1
  - libxxhash 0.8.2 -> 0.8.3
  - glslang 15.0.0 -> 15.4.0
  - dovi_tool 2.1.2 -> 2.3.1
  - libjpeg-turbo 2.1.0 -> 3.1.1
  - lcms2 2.16 -> 2.17
  - zimg 3.0.5 -> 3.0.6
  - libplacebo 7.349.0 -> 7.351.0
  - libsvtav1 3.1.0 (new!) x64 only
  - libvvenc 1.13.1 (new!) x64 only
  - Remove mmt/tlv patch

## 7.94

- Fix framerate not set properly with --avhw/--avsw when reading mpeg2 ts files. 

## 7.93

- Updates for QSVEnc.auo (AviUtl/AviUtl2 plugin).

## 7.92

- Adjust default maximum GOP length for AV1 to be a multiple of gop-ref-dist.
- Change several options to let the driver handle by default.

## 7.91

- Fix 10-bit processing in [--vpp-afs](./QSVEncC_Options.en.md#--vpp-afs-param1value1param2value2).
- Improve precision of [--vpp-afs](./QSVEncC_Options.en.md#--vpp-afs-param1value1param2value2).
- Add option to explicitly specify field pattern in [--vpp-deinterlace](./QSVEncC_Options.en.md#--vpp-deinterlace-string).
- Fix incorrect frame rate when outputting in y4m format with raw output.
- Fix processing sometimes stopping during raw output.
- Add support for [--option-file](./QSVEncC_Options.en.md#--option-file-string) on Linux.
- Fix handling when end is omitted in [--dynamic-rc](./QSVEncC_Options.en.md#--dynamic-rc-intintintintparam1value1param2value2).

## 7.90

- Add support for combining [--output-format](./QSVEncC_Options.en.md#--output-format-string) with ```-c raw```. ( #257 )
  Now supports cases like ```-c raw --output-format nut```.
- Fix black/white processing in 10-bit depth for [--vpp-edgelevel](./QSVEncC_Options.en.md#--vpp-edgelevel-param1value1param2value2).
- Improve interlace detection when using [--avsw](./QSVEncC_Options.en.md#--avsw-string).


## 7.89

- Fixed an issue with [--vpp-decimate](./QSVEncC_Options.en.md#--vpp-decimate-param1value1param2value2) where timestamp and duration of frames became incorrect due to improper handling of the final frame's timing.
- Improved handling of [--avoid-idle-clock](./QSVEncC_Options.en.md#--avoid-idle-clock-string) auto during parallel encoding ([--parallel](./QSVEncC_Options.en.md#--parallel-int-or-param1value1param2value2)).

## 7.88

- Fix ```--check-features``` not working on Linux systmes from QSVEnc 7.86. ( #253 )

## 7.87

- Add ```inverse_tone_mapping``` option to [--vpp-libplacebo-tonemapping](./QSVEncC_Options.en.md#--vpp-libplacebo-tonemapping-param1value1param2value2). ( #250 )
- Fix GPU selection defaulting to the first GPU when performance counter information is not available. ( #248 )
- Add AI-based frame interpolation filter to double the frame rate. ([--vpp-ai-frameinterp](./QSVEncC_Options.en.md#--vpp-ai-frameinterp-param1value1param2value2), #215, #237)
- Adjust log output format.

## 7.86

- Use thread pool to prevent unlimited OpenCL build threads.
- Improve VBV buffer size log display for AV1. ( #249 )
- Improve stability of [--parallel](./QSVEncC_Options.en.md#--parallel-int-or-param1value1param2value2). ( #248 )
- Add ```gpu_select``` to [--log-level](./QSVEncC_Options.en.md#--log-level-string) to show GPU auto selection status.
- Fix error when using ```st2094-10``` and ```st2094-40``` for ```tonemapping_function``` in [--vpp-libplacebo-tonemapping](./QSVEncC_Options.en.md#--vpp-libplacebo-tonemapping-param1value1param2value2). ( #252 )

## 7.85

- Improve audio and video synchronization to achieve more uniform mixing when muxing with subtitles or data tracks.
- Improve invalid input data hadling to avoid freeze when "failed to run h264_mp4toannexb bitstream filter" error occurs. ( #246 )
  Now properly exits with error.
- Add support for uyvy as input color format.
- Automatically disable --parallel when number of encoders is 1 when using ```--parallel auto```. ( #247 )

## 7.84

- Add parallel encoding feature with file splitting. ([--parallel](./QSVEncC_Options.en.md#--parallel-int-or-param1value1param2value2))
- Add support for ISO 639-2 T-codes in language code specification.
- Continue processing even when DirectX11/Vulkan initialization fails.
- Fix timestamps occasionally becoming incorrect when using --seek with certain input files.
- Fix [--qp-min](./QSVEncC_Options.en.md#--qp-min-int) and [--qp-max](./QSVEncC_Options.en.md#--qp-max-int) not being set properly when only one of them is specified.
- Avoid unnecessary Dolby Vision RPU conversion.
- Fix error when using [--vpp-deinterlace](./QSVEncC_Options.en.md#--vpp-deinterlace-string) bob, where two frames with pts=0 were generated when the first frame was progressive with RFF.
- Add ```libmfx1``` package as dependency in Ubuntu 24.04 deb package.

## 7.83

- Fix [--dolby-vision-rpu](./QSVEncC_Options.en.md#--dolby-vision-rpu-string) in AV1 encoding.

## 7.82

- Fix some codecs not being able to decode with [--avsw](./QSVEncC_Options.en.md#--avsw) since version 7.80.
- Add options 10.0, 10.1, 10.2, 10.4 to [--dolby-vision-profile](./QSVEncC_Options.en.md#--dolby-vision-profile-string-hevc-av1).
- Fix [--dolby-vision-profile](./QSVEncC_Options.en.md#--dolby-vision-profile-string-hevc-av1) not working with inputs other than avhw/avsw.
- Improve hw device detection on Linux environments with multiple Intel GPUs.

## 7.81

- - Fix --avsw not working in QSVEnc 7.80.

## 7.80

- Fix chromaloc not written properly when writing into container format.

## 7.79

- Fix some case that audio not being able to play when writing to mkv using --audio-copy.

## 7.78

- Avoid width field in mp4 Track Header Box getting 0 when SAR is undefined.

## 7.77

- Fix some of the parameters of [--vpp-libplacebo-tonemapping](./QSVEncC_Options.en.md#--vpp-libplacebo-tonemapping-param1value1param2value2) not working properly.
- Fix [--trim](./QSVEncC_Options.en.md#--trim-intintintintintint) being offset for a few frames when input file is a "cut" file (which does not start from key frame) and is coded using OpenGOP.

## 7.76

- Fix --dolby-vision-rpu from file (not copy) corrupted from QSVEnc 7.74. ( #228 )
- Improve auto GPU select of --device auto (=default), to select unused device more accurately in multi GPU environment. ( #225 )
- Slightly improve process startup speed by running file input and device initialization in parallel.

## 7.75

- Fix [--dolby-vision-rpu](./QSVEncC_Options.en.md#--dolby-vision-rpu-string-hevc-av1) causing error from QSVEnc 7.74 when reading rpu file.
- Fix wrong parsing of ```grain_y``` and ```grain_c``` for [--vpp-libplacebo-deband](./QSVEncC_Options.en.md#--vpp-libplacebo-deband-param1value1param2value2).
- Now [--dolby-vision-rpu](./QSVEncC_Options.en.md#--dolby-vision-rpu-string-hevc-av1) and [--dhdr10-info](./QSVEncC_Options.en.md#--dhdr10-info-string-hevc-av1) can be used together.

## 7.74

- Remove rate control mode limitation for [--dolby-vision-rpu](./QSVEncC_Options.en.md#--dolby-vision-rpu-string-hevc-av1). ( #222 )
- Fix invalid value not returning error when using [--dolby-vision-profile](./QSVEncC_Options.en.md#--dolby-vision-profile-string-hevc-av1). ( #222 )
- Add option to set active area offsets to 0 for dolby vision rpu metadata. ( [--dolby-vision-rpu-prm crop](./QSVEncC_Options.en.md#--dolby-vision-rpu-prm-param1value1param2value2), #222 )
- Change log behavior.

## 7.73

- Improve encode performance when using [--dolby-vision-rpu copy](./QSVEncC_Options.en.md#--dolby-vision-rpu-copy-hevc-av1) for a long input file. ( #216 )
  Previously, the encode speed kept on going slower when using [--dolby-vision-rpu](./QSVEncC_Options.en.md#--dolby-vision-rpu-copy-hevc-av1) copy, but now encode speed should be stable.
- Fix muxer error copying PGS subtitles (using [--sub-copy](./QSVEncC_Options.en.md#--sub-copy-intstringintstring)) when the input has unsorted subtitle packets.
  Now is able to avoid "Application provided invalid, non monotonically increasing dts to muxer" error.
- Improve AV1 output when using --dhdr10-info.

## 7.72

- Changed implementation of [--dhdr10-info](./QSVEncC_Options.en.md#--dhdr10-info-string-hevc-av1) to work on Linux. ( #216 )
  Changed to use [libhdr10plus](https://github.com/quietvoid/hdr10plus_tool) instead of hdr10plus_gen.exe.
- Fixed [--dhdr10-info](./QSVEncC_Options.en.md#--dhdr10-info-string-hevc-av1) not working on raw output. ( #216 )
- Fixed crush when [--dolby-vision-rpu](./QSVEncC_Options.en.md#--dolby-vision-rpu-string-hevc-av1) is specified when there is no dovi in the input file. ( #216 )
- Fix input SAR not passed to output in QSVEnc 7.71.

## 7.71

- Add custom shader filter using libplacebo. ([--vpp-libplaceo-shader](./QSVEncC_Options.en.md#--vpp-libplacebo-shader-param1value1param2value2))
- Add deband filter by libplacebo. ([--vpp-libplacebo-deband](./QSVEncC_Options.en.md#--vpp-libplacebo-deband-param1value1param2value2))
- Add tone mapping filter by libplacebo. ([--vpp-libplacebo-tonemapping](./QSVEncC_Options.en.md#--vpp-libplacebo-tonemapping-param1value1param2value2))
- Fix memory leak when using the resize filter by libplacebo ([--vpp-resize](./QSVEncC_Options.en.md#--vpp-resize-string) libplacebo-xxx).
- Now [--dolby-vision-rpu copy](./QSVEncC_Options.en.md#--dolby-vision-rpu-copy-hevc-av1) will automatically convert to dolby vision profile 8 when input files is dolby vision profile 7 using libdovi.
- Fix [--dhdr10-info](./QSVEncC_Options.en.md#--dhdr10-info-string-hevc-av1) not working. ( #216 )

## 7.70

- Update [libvpl](https://github.com/intel/libvpl) to support API 2.13.
- Add [libplacebo](https://code.videolan.org/videolan/libplacebo) resize filters for Windows x64 build ([--vpp-resize](./QSVEncC_Options.en.md#--vpp-resize-string)).
- Update ffmpeg libraries. (Windows)
  - ffmpeg 7.0 -> 20240822
  - dav1d 1.4.1 -> 1.4.3
  - libvpl 2.11.0 -> 2.12.0
  - libvpx 2.14.0
  - Add MMT/TLV demuxer patch to support mmts files.
- Fix help of --vpp-smooth showing wrong value range for qp option.
