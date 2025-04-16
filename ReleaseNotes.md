# QSVEnc Release Notes

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
