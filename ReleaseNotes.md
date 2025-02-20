# QSVEnc Release Notes

## 7.81

- - Fix --avsw not working in QSVEnc 7.80.

## 7.80

- Fix chromaloc not written properly when writing into container format.

## 7.79

- Fix some case that audio not being able to play when writing to mkv using --audio-copy.

## 7.78

- Avoid width field in mp4 Track Header Box getting 0 when SAR is undefined.

## 7.77

- Fix some of the parameters of [--vpp-libplacebo-tonemapping](https://github.com/rigaya/QSVEnc/blob/master/QSVEncC_Options.en.md#--vpp-libplacebo-tonemapping-param1value1param2value2) not working properly.
- Fix [--trim](https://github.com/rigaya/QSVEnc/blob/master/QSVEncC_Options.en.md#--trim-intintintintintint) being offset for a few frames when input file is a "cut" file (which does not start from key frame) and is coded using OpenGOP.

## 7.76

- Fix --dolby-vision-rpu from file (not copy) corrupted from QSVEnc 7.74. ( #228 )
- Improve auto GPU select of --device auto (=default), to select unused device more accurately in multi GPU environment. ( #225 )
- Slightly improve process startup speed by running file input and device initialization in parallel.

## 7.75

- Fix [--dolby-vision-rpu](https://github.com/rigaya/QSVEnc/blob/master/QSVEncC_Options.en.md#--dolby-vision-rpu-string-hevc-av1) causing error from QSVEnc 7.74 when reading rpu file.
- Fix wrong parsing of ```grain_y``` and ```grain_c``` for [--vpp-libplacebo-deband](https://github.com/rigaya/QSVEnc/blob/master/QSVEncC_Options.en.md#--vpp-libplacebo-deband-param1value1param2value2).
- Now [--dolby-vision-rpu](https://github.com/rigaya/QSVEnc/blob/master/QSVEncC_Options.en.md#--dolby-vision-rpu-string-hevc-av1) and [--dhdr10-info](https://github.com/rigaya/QSVEnc/blob/master/QSVEncC_Options.en.md#--dhdr10-info-string-hevc-av1) can be used together.

## 7.74

- Remove rate control mode limitation for [--dolby-vision-rpu](https://github.com/rigaya/QSVEnc/blob/master/QSVEncC_Options.en.md#--dolby-vision-rpu-string-hevc-av1). ( #222 )
- Fix invalid value not returning error when using [--dolby-vision-profile](https://github.com/rigaya/QSVEnc/blob/master/QSVEncC_Options.en.md#--dolby-vision-profile-string-hevc-av1). ( #222 )
- Add option to set active area offsets to 0 for dolby vision rpu metadata. ( [--dolby-vision-rpu-prm crop](https://github.com/rigaya/QSVEnc/blob/master/QSVEncC_Options.en.md#--dolby-vision-rpu-prm-param1value1param2value2), #222 )
- Change log behavior.

## 7.73

- Improve encode performance when using [--dolby-vision-rpu copy](https://github.com/rigaya/QSVEnc/blob/master/QSVEncC_Options.en.md#--dolby-vision-rpu-copy-hevc-av1) for a long input file. ( #216 )
  Previously, the encode speed kept on going slower when using [--dolby-vision-rpu](https://github.com/rigaya/QSVEnc/blob/master/QSVEncC_Options.en.md#--dolby-vision-rpu-copy-hevc-av1) copy, but now encode speed should be stable.
- Fix muxer error copying PGS subtitles (using [--sub-copy](https://github.com/rigaya/QSVEnc/blob/master/QSVEncC_Options.en.md#--sub-copy-intstringintstring)) when the input has unsorted subtitle packets.
  Now is able to avoid "Application provided invalid, non monotonically increasing dts to muxer" error.
- Improve AV1 output when using --dhdr10-info.

## 7.72

- Changed implementation of [--dhdr10-info](https://github.com/rigaya/QSVEnc/blob/master/QSVEncC_Options.en.md#--dhdr10-info-string-hevc-av1) to work on Linux. ( #216 )
  Changed to use [libhdr10plus](https://github.com/quietvoid/hdr10plus_tool) instead of hdr10plus_gen.exe.
- Fixed [--dhdr10-info](https://github.com/rigaya/QSVEnc/blob/master/QSVEncC_Options.en.md#--dhdr10-info-string-hevc-av1) not working on raw output. ( #216 )
- Fixed crush when [--dolby-vision-rpu](https://github.com/rigaya/QSVEnc/blob/master/QSVEncC_Options.en.md#--dolby-vision-rpu-string-hevc-av1) is specified when there is no dovi in the input file. ( #216 )
- Fix input SAR not passed to output in QSVEnc 7.71.

## 7.71

- Add custom shader filter using libplacebo. ([--vpp-libplaceo-shader](https://github.com/rigaya/QSVEnc/blob/master/QSVEncC_Options.en.md#--vpp-libplacebo-shader-param1value1param2value2))
- Add deband filter by libplacebo. ([--vpp-libplacebo-deband](https://github.com/rigaya/QSVEnc/blob/master/QSVEncC_Options.en.md#--vpp-libplacebo-deband-param1value1param2value2))
- Add tone mapping filter by libplacebo. ([--vpp-libplacebo-tonemapping](https://github.com/rigaya/QSVEnc/blob/master/QSVEncC_Options.en.md#--vpp-libplacebo-tonemapping-param1value1param2value2))
- Fix memory leak when using the resize filter by libplacebo ([--vpp-resize](https://github.com/rigaya/QSVEnc/blob/master/QSVEncC_Options.en.md#--vpp-resize-string) libplacebo-xxx).
- Now [--dolby-vision-rpu copy](https://github.com/rigaya/QSVEnc/blob/master/QSVEncC_Options.en.md#--dolby-vision-rpu-copy-hevc-av1) will automatically convert to dolby vision profile 8 when input files is dolby vision profile 7 using libdovi.
- Fix [--dhdr10-info](https://github.com/rigaya/QSVEnc/blob/master/QSVEncC_Options.en.md#--dhdr10-info-string-hevc-av1) not working. ( #216 )

## 7.70

- Update [libvpl](https://github.com/intel/libvpl) to support API 2.13.
- Add [libplacebo](https://code.videolan.org/videolan/libplacebo) resize filters for Windows x64 build ([--vpp-resize](https://github.com/rigaya/QSVEnc/blob/master/QSVEncC_Options.en.md#--vpp-resize-string)).
- Update ffmpeg libraries. (Windows)
  - ffmpeg 7.0 -> 20240822
  - dav1d 1.4.1 -> 1.4.3
  - libvpl 2.11.0 -> 2.12.0
  - libvpx 2.14.0
  - Add MMT/TLV demuxer patch to support mmts files.
- Fix help of --vpp-smooth showing wrong value range for qp option.
