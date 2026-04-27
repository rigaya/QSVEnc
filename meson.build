project('qsvencc', ['c', 'cpp'],
  version: run_command('git', 'describe', '--tags', '--abbrev=0', check: true).stdout().strip(),
  license: 'MIT',
  meson_version: '>=0.61.0',
  default_options: [
    'cpp_std=c++17',
    'buildtype=release',
    'warning_level=1',
  ]
)

cpp = meson.get_compiler('cpp')
fs = import('fs')

if host_machine.system() != 'linux'
  error('This meson.build is for Linux only')
endif

source_root = meson.project_source_root()
build_root = meson.project_build_root()

# Optional feature switches (configure互換)
enable_vapoursynth = get_option('enable_vapoursynth')
enable_avisynth = get_option('enable_avisynth')
enable_libass = get_option('enable_libass')
enable_libplacebo = get_option('enable_libplacebo')
enable_vulkan = get_option('enable_vulkan')
opencl_headers = get_option('opencl_headers')

threads_dep = dependency('threads')
dl_dep = cpp.find_library('dl', required: true)
m_dep = cpp.find_library('m', required: true)
stdcxxfs_dep = cpp.find_library('stdc++fs', required: false)

# libva
libva_dep = dependency('libva', required: true)
libva_drm_dep = dependency('libva-drm', required: false)
libva_x11_dep = dependency('libva-x11', required: false)
libdrm_dep = dependency('libdrm', required: false)
have_libva = libva_dep.found() or libva_drm_dep.found() or libva_x11_dep.found()

# Build bundled oneVPL (out-of-tree)
vpl_install_root = build_root / 'buildVPL'
vpl_build_root = build_root / 'buildVPL_build'
if not fs.is_file(vpl_install_root / 'lib/libvpl.a') and not fs.is_file(vpl_install_root / 'lib64/libvpl.a')
  vpl_cmd = [
    'cmake -S "' + source_root + '/libvpl" -B "' + vpl_build_root + '"',
    '-DCMAKE_INSTALL_PREFIX="' + vpl_install_root + '"',
    '-DBUILD_DEV=OFF',
    '-DBUILD_TOOLS=OFF',
    '-DBUILD_SHARED_LIBS=OFF',
    '-DBUILD_EXAMPLES=OFF',
    '-DBUILD_PYTHON_BINDING=OFF',
    '-DCMAKE_C_FLAGS_RELEASE="-Os -DONEVPL_EXPERIMENTAL"',
    '-DCMAKE_CXX_FLAGS_RELEASE="-Os -DONEVPL_EXPERIMENTAL"',
    '&& cmake --build "' + vpl_build_root + '" --config Release',
    '&& cmake --install "' + vpl_build_root + '" --config Release',
  ]
  vpl_build_result = run_command('bash', '-c', ' '.join(vpl_cmd), check: false)
  if vpl_build_result.returncode() != 0
    error('Failed to build bundled oneVPL')
  endif
endif

vpl_lib_dirs = []
if fs.is_file(vpl_install_root / 'lib/libvpl.a')
  vpl_lib_dirs += vpl_install_root / 'lib'
elif fs.is_file(vpl_install_root / 'lib64/libvpl.a')
  vpl_lib_dirs += vpl_install_root / 'lib64'
else
  error('libvpl.a not found after bundled oneVPL build')
endif
vpl_dep = cpp.find_library('vpl', dirs: vpl_lib_dirs, static: true, required: true)

# OpenCL headers
opencl_inc_args = []
if opencl_headers != ''
  opencl_inc_args += ['-I' + opencl_headers]
endif
if not cpp.has_header('CL/opencl.h', args: opencl_inc_args, required: false)
  error('OpenCL headers not found (CL/opencl.h)')
endif
if not cpp.has_header('clRNG/clRNG.h', args: ['-I' + source_root / 'clRNG/src/include'], required: false)
  error('clRNG headers not found (clRNG/clRNG.h)')
endif

# FFmpeg (required)
libavutil_dep = dependency('libavutil', required: true)
libavcodec_dep = dependency('libavcodec', required: true)
libavformat_dep = dependency('libavformat', required: true)
libavfilter_dep = dependency('libavfilter', required: true)
libavdevice_dep = dependency('libavdevice', required: true)
libswresample_dep = dependency('libswresample', required: true)

ffmpeg_deps = [
  libavutil_dep,
  libavcodec_dep,
  libavformat_dep,
  libavfilter_dep,
  libavdevice_dep,
  libswresample_dep,
]

# ヘッダ自動取得先（ビルドディレクトリ内）
_ext_headers_dir  = build_root / 'external_headers'
_vpy_fetched_inc  = _ext_headers_dir / 'vapoursynth-R72' / 'include'
_avs_fetched_inc  = _ext_headers_dir / 'AviSynthPlus-3.7.5' / 'avs_core' / 'include'

# VapourSynth ヘッダ自動ダウンロード用シェルスクリプト
_fetch_vpy_sh = '''
dest="$1"
[ -f "$dest/vapoursynth-R72/include/VapourSynth.h" ] && exit 0
mkdir -p "$dest"
url="https://github.com/vapoursynth/vapoursynth/archive/refs/tags/R72.tar.gz"
if command -v wget >/dev/null 2>&1; then
  wget -q -O "$dest/vpy.tar.gz" "$url" || exit 1
else
  curl -fsSL -o "$dest/vpy.tar.gz" "$url" || exit 1
fi
tar xf "$dest/vpy.tar.gz" -C "$dest" || exit 1
rm -f "$dest/vpy.tar.gz"
'''

# AviSynthPlus ヘッダ自動ダウンロード用シェルスクリプト
_fetch_avs_sh = '''
dest="$1"
[ -f "$dest/AviSynthPlus-3.7.5/avs_core/include/avisynth_c.h" ] && exit 0
mkdir -p "$dest"
url="https://github.com/AviSynth/AviSynthPlus/archive/refs/tags/v3.7.5.tar.gz"
if command -v wget >/dev/null 2>&1; then
  wget -q -O "$dest/avs.tar.gz" "$url" || exit 1
else
  curl -fsSL -o "$dest/avs.tar.gz" "$url" || exit 1
fi
tar xf "$dest/avs.tar.gz" -C "$dest" || exit 1
rm -f "$dest/avs.tar.gz"
'''

vapoursynth_compile_dep = disabler()
have_vapoursynth_headers = false
if enable_vapoursynth
  vapoursynth_pkg_dep = dependency('vapoursynth', required: false)
  vapoursynth_compile_dep = declare_dependency()
  if vapoursynth_pkg_dep.found()
    vapoursynth_compile_dep = vapoursynth_pkg_dep.partial_dependency(includes: true, compile_args: true)
  endif

  vapoursynth_v3_headers_code = '''
#include "VapourSynth.h"
#include "VSScript.h"
int main(void) { return 0; }
'''
  have_vapoursynth_v3_headers = cpp.compiles(
    vapoursynth_v3_headers_code,
    dependencies: vapoursynth_compile_dep,
    args: get_option('cpp_args'),
  )

  vapoursynth_v4_headers_code = '''
#include "VSScript4.h"
int main(void) { return 0; }
'''
  have_vapoursynth_v4_headers = cpp.compiles(
    vapoursynth_v4_headers_code,
    dependencies: vapoursynth_compile_dep,
    args: get_option('cpp_args'),
  )

  have_vapoursynth_headers = have_vapoursynth_v3_headers and have_vapoursynth_v4_headers
  if not have_vapoursynth_headers
    _fetch = run_command(
      'sh', '-c', _fetch_vpy_sh, 'fetch_vpy', _ext_headers_dir,
      check: false,
    )
    if _fetch.returncode() == 0
      vapoursynth_compile_dep = declare_dependency(
        compile_args: ['-I' + _vpy_fetched_inc],
      )
      have_vapoursynth_v3_headers = cpp.compiles(
        vapoursynth_v3_headers_code,
        dependencies: vapoursynth_compile_dep,
        args: get_option('cpp_args'),
      )
      have_vapoursynth_v4_headers = cpp.compiles(
        vapoursynth_v4_headers_code,
        dependencies: vapoursynth_compile_dep,
        args: get_option('cpp_args'),
      )
      have_vapoursynth_headers = have_vapoursynth_v3_headers and have_vapoursynth_v4_headers
    endif
    if not have_vapoursynth_headers
      error('VapourSynth headers not found (require V3: VapourSynth.h/VSScript.h and V4: VSScript4.h)')
    endif
  endif
endif

avisynth_compile_dep = disabler()
have_avisynth_headers = false
if enable_avisynth
  avisynth_pkg_dep = dependency('avisynth', required: false)
  avisynth_compile_dep = declare_dependency()
  if avisynth_pkg_dep.found()
    avisynth_compile_dep = avisynth_pkg_dep.partial_dependency(includes: true, compile_args: true)
  endif

  avisynth_headers_code = '''
#include "avisynth_c.h"
int main(void) { return 0; }
'''
  have_avisynth_headers = cpp.compiles(
    avisynth_headers_code,
    dependencies: avisynth_compile_dep,
    args: get_option('cpp_args'),
  )
  if not have_avisynth_headers
    _fetch = run_command(
      'sh', '-c', _fetch_avs_sh, 'fetch_avs', _ext_headers_dir,
      check: false,
    )
    if _fetch.returncode() == 0
      avisynth_compile_dep = declare_dependency(
        compile_args: ['-I' + _avs_fetched_inc],
      )
      have_avisynth_headers = cpp.compiles(
        avisynth_headers_code,
        dependencies: avisynth_compile_dep,
        args: get_option('cpp_args'),
      )
    endif
    if not have_avisynth_headers
      error('AviSynth headers not found (avisynth_c.h)')
    endif
  endif
endif
if enable_libass
  libass_dep = dependency('libass', required: true)
else
  libass_dep = disabler()
endif
if enable_vulkan
  vulkan_dep = dependency('vulkan', required: true)
else
  vulkan_dep = disabler()
endif
if enable_libplacebo
  libplacebo_dep = dependency('libplacebo', required: true)
else
  libplacebo_dep = disabler()
endif

if libplacebo_dep.found()
  placebo_api_check = '''
#include <libplacebo/config.h>
#if PL_API_VER < 338
#error "libplacebo API too old"
#endif
int main(void) { return 0; }
'''
  if not cpp.compiles(placebo_api_check, dependencies: libplacebo_dep)
    error('libplacebo API version >= 338 required')
  endif
endif

# libdovi / libhdr10plus: system優先、なければ同梱build (configure互換)
cargo_prog = find_program('cargo', required: false)
if cargo_prog.found()
  cargo_home_dir = source_root / '.cargo_home'
  cargo_c_check = run_command(
    'env',
    'CARGO_HOME=' + cargo_home_dir,
    cargo_prog.full_path(),
    'cinstall',
    '--version',
    check: false,
  )
  if cargo_c_check.returncode() != 0
    cargo_c_install = run_command(
      'env',
      'CARGO_HOME=' + cargo_home_dir,
      cargo_prog.full_path(),
      'install',
      'cargo-c',
      '--locked',
      check: false,
    )
    if cargo_c_install.returncode() != 0
      error('Failed to install cargo-c (exit=' + cargo_c_install.returncode().to_string() + ')\nstdout:\n' + cargo_c_install.stdout().strip() + '\nstderr:\n' + cargo_c_install.stderr().strip())
    endif
  endif
endif

libdovi_dep = dependency('dovi', required: false)
if not libdovi_dep.found()
  if not cargo_prog.found()
    error('cargo is required to build bundled libdovi when system dovi is not available')
  endif
  libdovi_install_dir = build_root / 'rust_libs' / 'libdovi'
  libdovi_build_result = run_command(
    'env',
    'CARGO_HOME=' + cargo_home_dir,
    'CARGO=' + cargo_prog.full_path(),
    'bash',
    source_root / 'build_libdovi.sh',
    libdovi_install_dir,
    check: false,
  )
  if libdovi_build_result.returncode() != 0
    error('Failed to build bundled libdovi (exit=' + libdovi_build_result.returncode().to_string() + ')\nstdout:\n' + libdovi_build_result.stdout().strip() + '\nstderr:\n' + libdovi_build_result.stderr().strip())
  endif
  libdovi_lib_dirs = []
  if fs.is_file(libdovi_install_dir / 'lib/x86_64-linux-gnu/libdovi.a')
    libdovi_lib_dirs += libdovi_install_dir / 'lib/x86_64-linux-gnu'
  elif fs.is_file(libdovi_install_dir / 'lib64/libdovi.a')
    libdovi_lib_dirs += libdovi_install_dir / 'lib64'
  elif fs.is_file(libdovi_install_dir / 'lib/libdovi.a')
    libdovi_lib_dirs += libdovi_install_dir / 'lib'
  endif
  if libdovi_lib_dirs.length() == 0
    error('Bundled libdovi build succeeded but libdovi.a was not found')
  endif
  libdovi_lib = cpp.find_library('dovi', dirs: libdovi_lib_dirs, static: true, required: true)
  libdovi_dep = declare_dependency(
    dependencies: libdovi_lib,
    compile_args: ['-I' + (libdovi_install_dir / 'include')],
  )
  if not cpp.compiles(
    '#include <libdovi/rpu_parser.h>\nint main(void){ dovi_parse_rpu(nullptr, 0); return 0; }',
    dependencies: libdovi_dep,
  )
    error('Bundled libdovi check failed (libdovi/rpu_parser.h)')
  endif
endif

libhdr10plus_dep = dependency('hdr10plus-rs', required: false)
if not libhdr10plus_dep.found()
  if not cargo_prog.found()
    error('cargo is required to build bundled libhdr10plus-rs when system hdr10plus-rs is not available')
  endif
  libhdr10plus_install_dir = build_root / 'rust_libs' / 'libhdr10plus'
  libhdr10plus_build_result = run_command(
    'env',
    'CARGO_HOME=' + cargo_home_dir,
    'CARGO=' + cargo_prog.full_path(),
    'bash',
    source_root / 'build_libhdr10plus.sh',
    libhdr10plus_install_dir,
    check: false,
  )
  if libhdr10plus_build_result.returncode() != 0
    error('Failed to build bundled libhdr10plus-rs (exit=' + libhdr10plus_build_result.returncode().to_string() + ')\nstdout:\n' + libhdr10plus_build_result.stdout().strip() + '\nstderr:\n' + libhdr10plus_build_result.stderr().strip())
  endif
  libhdr10plus_lib_dirs = []
  if fs.is_file(libhdr10plus_install_dir / 'lib/x86_64-linux-gnu/libhdr10plus-rs.a')
    libhdr10plus_lib_dirs += libhdr10plus_install_dir / 'lib/x86_64-linux-gnu'
  elif fs.is_file(libhdr10plus_install_dir / 'lib64/libhdr10plus-rs.a')
    libhdr10plus_lib_dirs += libhdr10plus_install_dir / 'lib64'
  elif fs.is_file(libhdr10plus_install_dir / 'lib/libhdr10plus-rs.a')
    libhdr10plus_lib_dirs += libhdr10plus_install_dir / 'lib'
  endif
  if libhdr10plus_lib_dirs.length() == 0
    error('Bundled libhdr10plus-rs build succeeded but libhdr10plus-rs.a was not found')
  endif
  libhdr10plus_lib = cpp.find_library('hdr10plus-rs', dirs: libhdr10plus_lib_dirs, static: true, required: true)
  libhdr10plus_dep = declare_dependency(
    dependencies: libhdr10plus_lib,
    compile_args: ['-I' + (libhdr10plus_install_dir / 'include')],
  )
  if not cpp.compiles(
    '#include <libhdr10plus-rs/hdr10plus.h>\nint main(void){ hdr10plus_rs_parse_json(nullptr); return 0; }',
    dependencies: libhdr10plus_dep,
  )
    error('Bundled libhdr10plus-rs check failed (libhdr10plus-rs/hdr10plus.h)')
  endif
endif

# FFmpeg API checks
conf_data = configuration_data()

av_channel_layout_code = '''
#include <libavutil/channel_layout.h>
int main(void) { AVChannelLayout ch_layout; return 0; }
'''
conf_data.set10('AV_CHANNEL_LAYOUT_STRUCT_AVAIL',
  cpp.compiles(av_channel_layout_code, dependencies: libavutil_dep))

av_frame_duration_code = '''
#include <libavutil/frame.h>
int main(void) { AVFrame *frame; frame->duration = 0; return 0; }
'''
conf_data.set10('AV_FRAME_DURATION_AVAIL',
  cpp.compiles(av_frame_duration_code, dependencies: libavutil_dep))

avcodec_par_side_data_code = '''
#include <libavcodec/codec_par.h>
int main(void) { AVCodecParameters *codecpar; codecpar->coded_side_data = 0; return 0; }
'''
conf_data.set10('AVCODEC_PAR_CODED_SIDE_DATA_AVAIL',
  cpp.compiles(avcodec_par_side_data_code, dependencies: libavcodec_dep))

conf_data.set10('ENABLE_RAW_READER', true)
conf_data.set10('ENABLE_AVI_READER', false)
conf_data.set10('ENABLE_AVISYNTH_READER', enable_avisynth and have_avisynth_headers)
conf_data.set10('ENABLE_VAPOURSYNTH_READER', enable_vapoursynth and have_vapoursynth_headers)
conf_data.set10('ENABLE_AVSW_READER', true)
conf_data.set10('ENABLE_SM_READER', false)
conf_data.set10('ENABLE_CUSTOM_VPP', true)
conf_data.set10('ENABLE_LIBASS_SUBBURN', libass_dep.found())
conf_data.set10('ENABLE_ADVANCED_DEINTERLACE', false)
conf_data.set10('ENABLE_MVC_ENCODING', false)
conf_data.set10('ENABLE_FPS_CONVERSION', false)
conf_data.set10('ENABLE_OPENCL', true)
conf_data.set10('ENABLE_SESSION_THREAD_CONFIG', false)
conf_data.set10('ENABLE_AVCODEC_OUT_THREAD', true)
conf_data.set10('ENABLE_CPP_REGEX', true)
conf_data.set10('ENABLE_DTL', true)
conf_data.set10('ENABLE_LIBDOVI', libdovi_dep.found())
conf_data.set10('ENABLE_LIBHDR10PLUS', libhdr10plus_dep.found())
conf_data.set10('ENABLE_VULKAN', vulkan_dep.found() and libplacebo_dep.found())
conf_data.set10('ENABLE_LIBPLACEBO', libplacebo_dep.found())

if enable_avisynth and have_avisynth_headers
  conf_data.set('AVS_INTERF_VER', 12)
else
  conf_data.set('AVS_INTERF_VER', 0)
endif

configure_file(
  output: 'rgy_config.h',
  configuration: conf_data,
)

rev_count = run_command('sh', '-c', 'cd "' + source_root + '" && git rev-list HEAD | wc -l', check: false).stdout().strip()
if rev_count == ''
  rev_count = '0'
endif

rev_data = configuration_data()
rev_data.set_quoted('ENCODER_REV', rev_count)
rev_data.set('ENABLE_METRIC_FRAMEWORK', 0)
configure_file(
  output: 'rgy_rev.h',
  configuration: rev_data,
)

build_dir_include = '-I' + meson.current_build_dir()

common_cpp_args = [
  build_dir_include,
  '-DLINUX',
  '-DUNIX',
  '-D_FILE_OFFSET_BITS=64',
  '-D__USE_LARGEFILE64',
  '-D__STDC_FORMAT_MACROS',
  '-D__STDC_CONSTANT_MACROS',
  '-Wno-unknown-pragmas',
  '-Wno-unused',
  '-Wno-missing-braces',
  '-msse4.1',
] + opencl_inc_args

if have_libva
  common_cpp_args += ['-DLIBVA_SUPPORT=1']
else
  common_cpp_args += ['-DLIBVA_SUPPORT=0']
endif
if libva_drm_dep.found()
  common_cpp_args += ['-DLIBVA_DRM_SUPPORT=1']
else
  common_cpp_args += ['-DLIBVA_DRM_SUPPORT=0']
endif
if libva_x11_dep.found()
  common_cpp_args += ['-DLIBVA_X11_SUPPORT=1']
else
  common_cpp_args += ['-DLIBVA_X11_SUPPORT=0']
endif

common_c_args = [
  '-DLINUX',
  '-DUNIX',
  '-D_FILE_OFFSET_BITS=64',
  '-D__USE_LARGEFILE64',
  '-D__STDC_FORMAT_MACROS',
  '-D__STDC_CONSTANT_MACROS',
  '-Wno-unknown-pragmas',
  '-Wno-unused',
  '-Wno-missing-braces',
  '-msse4.1',
]

common_include_dir_paths = [
  '.',
  'QSVPipeline',
  'tinyxml2',
  'ttmath',
  'dtl',
  'cppcodec',
  'clRNG/src/include',
  'libvpl/api/vpl',
]
common_include_dirs = include_directories(common_include_dir_paths)

qsvpipeline_sources = files(
  'QSVPipeline/DeviceId.cpp',
  'QSVPipeline/convert_csp.cpp',
  'QSVPipeline/cpu_info.cpp',
  'QSVPipeline/gpu_info.cpp',
  'QSVPipeline/gpuz_info.cpp',
  'QSVPipeline/logo.cpp',
  'QSVPipeline/qsv_allocator.cpp',
  'QSVPipeline/qsv_allocator_d3d11.cpp',
  'QSVPipeline/qsv_allocator_d3d9.cpp',
  'QSVPipeline/qsv_allocator_sys.cpp',
  'QSVPipeline/qsv_cmd.cpp',
  'QSVPipeline/qsv_device.cpp',
  'QSVPipeline/qsv_hw_d3d11.cpp',
  'QSVPipeline/qsv_hw_d3d9.cpp',
  'QSVPipeline/qsv_hw_device.cpp',
  'QSVPipeline/qsv_mfx_dec.cpp',
  'QSVPipeline/qsv_pipeline.cpp',
  'QSVPipeline/qsv_prm.cpp',
  'QSVPipeline/qsv_query.cpp',
  'QSVPipeline/qsv_session.cpp',
  'QSVPipeline/qsv_util.cpp',
  'QSVPipeline/qsv_vpp_mfx.cpp',
  'QSVPipeline/rgy_aspect_ratio.cpp',
  'QSVPipeline/rgy_avlog.cpp',
  'QSVPipeline/rgy_avutil.cpp',
  'QSVPipeline/rgy_bitstream.cpp',
  'QSVPipeline/rgy_bitstream_aac.cpp',
  'QSVPipeline/rgy_chapter.cpp',
  'QSVPipeline/rgy_cmd.cpp',
  'QSVPipeline/rgy_codepage.cpp',
  'QSVPipeline/rgy_d2v_parser.cpp',
  'QSVPipeline/rgy_def.cpp',
  'QSVPipeline/rgy_device_info_cache.cpp',
  'QSVPipeline/rgy_device_usage.cpp',
  'QSVPipeline/rgy_device_vulkan.cpp',
  'QSVPipeline/rgy_dummy_load.cpp',
  'QSVPipeline/rgy_env.cpp',
  'QSVPipeline/rgy_err.cpp',
  'QSVPipeline/rgy_event.cpp',
  'QSVPipeline/rgy_faw.cpp',
  'QSVPipeline/rgy_filesystem.cpp',
  'QSVPipeline/rgy_filter.cpp',
  'QSVPipeline/rgy_filter_afs.cpp',
  'QSVPipeline/rgy_filter_afs_analyze.cpp',
  'QSVPipeline/rgy_filter_afs_filter.cpp',
  'QSVPipeline/rgy_filter_afs_merge.cpp',
  'QSVPipeline/rgy_filter_afs_synthesize.cpp',
  'QSVPipeline/rgy_filter_bwdif.cpp',
  'QSVPipeline/rgy_filter_cl.cpp',
  'QSVPipeline/rgy_filter_colorspace.cpp',
  'QSVPipeline/rgy_filter_convolution3d.cpp',
  'QSVPipeline/rgy_filter_crop.cpp',
  'QSVPipeline/rgy_filter_curves.cpp',
  'QSVPipeline/rgy_filter_deband.cpp',
  'QSVPipeline/rgy_filter_decimate.cpp',
  'QSVPipeline/rgy_filter_decomb.cpp',
  'QSVPipeline/rgy_filter_delogo.cpp',
  'QSVPipeline/rgy_filter_denoise_dct.cpp',
  'QSVPipeline/rgy_filter_denoise_fft3d.cpp',
  'QSVPipeline/rgy_filter_denoise_knn.cpp',
  'QSVPipeline/rgy_filter_denoise_nlmeans.cpp',
  'QSVPipeline/rgy_filter_denoise_pmd.cpp',
  'QSVPipeline/rgy_filter_edgelevel.cpp',
  'QSVPipeline/rgy_filter_ivtc.cpp',
  'QSVPipeline/rgy_filter_msharpen.cpp',
  'QSVPipeline/rgy_filter_msmooth.cpp',
  'QSVPipeline/rgy_filter_libplacebo.cpp',
  'QSVPipeline/rgy_filter_mpdecimate.cpp',
  'QSVPipeline/rgy_filter_nnedi.cpp',
  'QSVPipeline/rgy_filter_overlay.cpp',
  'QSVPipeline/rgy_filter_pad.cpp',
  'QSVPipeline/rgy_filter_resize.cpp',
  'QSVPipeline/rgy_filter_rff.cpp',
  'QSVPipeline/rgy_filter_smooth.cpp',
  'QSVPipeline/rgy_filter_ssim.cpp',
  'QSVPipeline/rgy_filter_subburn.cpp',
  'QSVPipeline/rgy_filter_transform.cpp',
  'QSVPipeline/rgy_filter_tweak.cpp',
  'QSVPipeline/rgy_filter_unsharp.cpp',
  'QSVPipeline/rgy_filter_warpsharp.cpp',
  'QSVPipeline/rgy_filter_yadif.cpp',
  'QSVPipeline/rgy_frame.cpp',
  'QSVPipeline/rgy_frame_info.cpp',
  'QSVPipeline/rgy_hdr10plus.cpp',
  'QSVPipeline/rgy_ini.cpp',
  'QSVPipeline/rgy_input.cpp',
  'QSVPipeline/rgy_input_avcodec.cpp',
  'QSVPipeline/rgy_input_avi.cpp',
  'QSVPipeline/rgy_input_avs.cpp',
  'QSVPipeline/rgy_input_raw.cpp',
  'QSVPipeline/rgy_input_sm.cpp',
  'QSVPipeline/rgy_input_vpy.cpp',
  'QSVPipeline/rgy_language.cpp',
  'QSVPipeline/rgy_libdovi.cpp',
  'QSVPipeline/rgy_libplacebo.cpp',
  'QSVPipeline/rgy_log.cpp',
  'QSVPipeline/rgy_memmem.cpp',
  'QSVPipeline/rgy_opencl.cpp',
  'QSVPipeline/rgy_output.cpp',
  'QSVPipeline/rgy_output_avcodec.cpp',
  'QSVPipeline/rgy_parallel_enc.cpp',
  'QSVPipeline/rgy_perf_counter.cpp',
  'QSVPipeline/rgy_perf_monitor.cpp',
  'QSVPipeline/rgy_pipe.cpp',
  'QSVPipeline/rgy_pipe_linux.cpp',
  'QSVPipeline/rgy_prm.cpp',
  'QSVPipeline/rgy_resource.cpp',
  'QSVPipeline/rgy_simd.cpp',
  'QSVPipeline/rgy_status.cpp',
  'QSVPipeline/rgy_thread_affinity.cpp',
  'QSVPipeline/rgy_timecode.cpp',
  'QSVPipeline/rgy_util.cpp',
  'QSVPipeline/rgy_vapoursynth_wrapper.cpp',
  'QSVPipeline/rgy_vapoursynth_wrapper_v3.cpp',
  'QSVPipeline/rgy_vapoursynth_wrapper_v4.cpp',
  'QSVPipeline/rgy_version.cpp',
  'QSVPipeline/rgy_vulkan.cpp',
  'QSVPipeline/rgy_wav_parser.cpp',
)

if have_libva
  qsvpipeline_sources += files(
    'QSVPipeline/qsv_allocator_va.cpp',
    'QSVPipeline/qsv_hw_va.cpp',
    'QSVPipeline/qsv_hw_va_utils.cpp',
    'QSVPipeline/qsv_hw_va_utils_drm.cpp',
    'QSVPipeline/qsv_hw_va_utils_x11.cpp',
  )
endif

clrng_sources = files(
  'clRNG/src/library/clRNG.c',
  'clRNG/src/library/private.c',
)

tinyxml2_sources = files('tinyxml2/tinyxml2.cpp')
qsvencc_sources = files('QSVEncC/QSVEncC.cpp')

simd_sse2_lib = static_library('simd_sse2',
  'QSVPipeline/convert_csp_sse2.cpp',
  cpp_args: common_cpp_args + ['-msse2'],
  include_directories: common_include_dirs,
)

simd_ssse3_lib = static_library('simd_ssse3',
  'QSVPipeline/convert_csp_ssse3.cpp',
  cpp_args: common_cpp_args + ['-mssse3'],
  include_directories: common_include_dirs,
)

simd_sse41_lib = static_library('simd_sse41',
  'QSVPipeline/convert_csp_sse41.cpp',
  cpp_args: common_cpp_args + ['-msse4.1'],
  include_directories: common_include_dirs,
)

simd_avx_lib = static_library('simd_avx',
  'QSVPipeline/convert_csp_avx.cpp',
  cpp_args: common_cpp_args + ['-mavx', '-mpopcnt'],
  include_directories: common_include_dirs,
)

simd_avx2_lib = static_library('simd_avx2',
  [
    'QSVPipeline/convert_csp_avx2.cpp',
    'QSVPipeline/rgy_bitstream_avx2.cpp',
    'QSVPipeline/rgy_faw_avx2.cpp',
    'QSVPipeline/rgy_memmem_avx2.cpp',
  ],
  cpp_args: common_cpp_args + ['-mavx2', '-mfma', '-mpopcnt', '-mbmi', '-mbmi2'],
  include_directories: common_include_dirs,
)

simd_avx512_lib = static_library('simd_avx512',
  [
    'QSVPipeline/rgy_bitstream_avx512bw.cpp',
    'QSVPipeline/rgy_faw_avx512bw.cpp',
    'QSVPipeline/rgy_memmem_avx512bw.cpp',
  ],
  cpp_args: common_cpp_args + ['-mavx512f', '-mavx512bw', '-mpopcnt', '-mbmi', '-mbmi2'],
  include_directories: common_include_dirs,
)

simd_libs = [
  simd_sse2_lib,
  simd_ssse3_lib,
  simd_sse41_lib,
  simd_avx_lib,
  simd_avx2_lib,
  simd_avx512_lib,
]

all_deps = [
  threads_dep,
  dl_dep,
  m_dep,
  vpl_dep,
] + ffmpeg_deps

if stdcxxfs_dep.found()
  all_deps += stdcxxfs_dep
endif
if libva_dep.found()
  all_deps += libva_dep
endif
if libva_drm_dep.found()
  all_deps += libva_drm_dep
endif
if libva_x11_dep.found()
  all_deps += libva_x11_dep
endif
if libdrm_dep.found()
  all_deps += libdrm_dep
endif
if vapoursynth_compile_dep.found()
  all_deps += vapoursynth_compile_dep
endif
if avisynth_compile_dep.found()
  all_deps += avisynth_compile_dep
endif
if libass_dep.found()
  all_deps += libass_dep
endif
if vulkan_dep.found()
  all_deps += vulkan_dep
endif
if libplacebo_dep.found()
  all_deps += libplacebo_dep
endif
if libdovi_dep.found()
  all_deps += libdovi_dep
endif
if libhdr10plus_dep.found()
  all_deps += libhdr10plus_dep
endif

subdir('meson_embed')

executable('qsvencc',
  qsvpipeline_sources + clrng_sources + tinyxml2_sources + qsvencc_sources + resource_objects,
  c_args: common_c_args,
  cpp_args: common_cpp_args,
  include_directories: common_include_dirs,
  dependencies: all_deps,
  link_with: simd_libs,
  install: true,
)

summary({
  'libva': libva_dep.found(),
  'libva-drm': libva_drm_dep.found(),
  'libva-x11': libva_x11_dep.found(),
  'libdrm': libdrm_dep.found(),
  'VapourSynth': enable_vapoursynth and have_vapoursynth_headers,
  'AviSynth': enable_avisynth and have_avisynth_headers,
  'libass': libass_dep.found(),
  'libplacebo': libplacebo_dep.found(),
  'Vulkan': vulkan_dep.found(),
  'libdovi': libdovi_dep.found(),
  'libhdr10plus': libhdr10plus_dep.found(),
}, section: 'QSVEnc Features')
