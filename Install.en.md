
# How to install QSVEncC

- [Windows 10](./Install.en.md#windows)
- Linux
  - [Linux (Ubuntu 20.04 and later)](./Install.en.md#linux-ubuntu-2004-and-later)
  - [Linux (Fedora 32)](./Install.en.md#linux-fedora-32)
  - Other Linux OS  
    For Linux distributions that cannot use the distributed deb, building from source will be needed. Please check the [build instrcutions](./Build.en.md).


## Windows 10

### 1. Install Intel Graphics driver
### 2. Download Windows binary  
Windows binary can be found from [this link](https://github.com/rigaya/QSVEnc/releases). QSVEncC_x.xx_Win32.7z contains 32bit exe file, QSVEncC_x.xx_x64.7z contains 64bit exe file.

QSVEncC could be run directly from the extracted directory.
  
## Linux (Ubuntu 20.04 and later)

On Linux, Intel GPU encoding can use either QSV or VA-API.

| Backend | QSV | VA-API |
|:--|:--|:--|
| Required runtime | VPL (`libmfx-gen`) for Tiger Lake and later; Media SDK (`libmfx1`) for supported earlier GPUs | VA driver (`iHD` / `i965`) only; no VPL / Media SDK runtime |
| HW decoding (`--avhw`) | Supported | Unsupported; `--avsw` and automatic input selection use software decoding |
| Filters | MFX VPP / OpenCL | OpenCL only; encoding without filters if OpenCL is unavailable |
| Encoder settings | Detailed MFX settings | Basic rate control, GOP and quality settings; unsupported MFX-specific settings are ignored with a warning, while some produce an error |

See [--backend](./QSVEncC_Options.en.md#--backend-autoqsvvaapi) for detailed support. As a performance reference, encoding the same input with `--avsw` on Arc A310 / Ubuntu 26.04 / iHD 26.3.2 from the kobuk-team PPA reached approximately 96% of QSV speed for H.264 and 101% for HEVC. Results depend on the GPU, driver and settings.

Use VA-API when the GPU or distribution has no VPL runtime, or when an older GPU can only use i965. For example, Ubuntu 26.04 has no `libmfx1` package, leaving Kaby Lake (Gen9, including HD 630) without a VPL runtime. VA-API was verified in Ubuntu 24.04 Docker containers with the standard 24.1 driver and on Ubuntu 26.04 hardware with iHD 26.3.2 from the kobuk-team PPA. Operation with Ubuntu 26.04's standard iHD 26.1.2 driver has not been verified.

The default `--backend auto` tries QSV and switches to VA-API if no device is available. It skips QSV and selects VA-API directly in these cases:

- `QSVENC_VPL_DISABLE=1` is set.
- `LIBVA_DRIVER_NAME` is set to a value other than `iHD` (case-insensitive).
- `iHD_drv_video.so` is not found in the driver search paths, such as an i965-only installation.

Encoding errors after QSV has been selected do not trigger a switch. If the VPL runtime crashes at startup, use `QSVENC_VPL_DISABLE=1 qsvencc ...` to avoid loading it. Select VA-API explicitly with `--backend vaapi`, or QSV with `--backend qsv`.

### 1. Preparation

#### 1-1. Using QSV: Add the Intel Media driver repository

The Intel repository commands below are for Ubuntu 22.04 / 24.04. For VA-API-only operation, proceed to section 1-2.

:::note warn  
**If your system includes Gen11 or earlier GPUs, skip this section and proceed to section 2.**

Intel repository will provide the latest user mode driver, but intel-opencl-icd 24.35 supports only Gen12 or later, resulting failure when detecting Gen11 iGPU or before. Please use standard Ubuntu repo which provides intel-opencl-icd 23.43 (Ubuntu 24.04) or 22.14 (Ubuntu 22.04).

If Gen11 or earlier and Gen12 or later GPUs are mixed in the same PC, perform this section, and then add the OpenCL runtime for Gen11 or earlier following [3-1.](#3-1-when-mixing-gen11-or-earlier-and-gen12-or-later-gpus).

- Gen11 or before: Broadwell, Skylake, Kaby Lake, Coffee Lake, Apollo Lake, Gemini Lake, Ice Lake, Elkhart Lake
- Gen12 or later: Tiger Lake, Rocket Lake, Alder Lake, Raptor Lake, Arc dGPU など
::

Intel media driver can be installed following instruction on [this link](https://dgpu-docs.intel.com/driver/client/overview.html).

First, install required tools.

```Shell
sudo apt-get install -y gpg-agent wget
```

Next, add Intel package repository.

```Shell
# Ubuntu 24.04
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble unified" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu-noble.list

# Ubuntu 22.04
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy unified" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list
```

#### 1-2. Using VA-API

Install a VA driver from the standard Ubuntu 24.04 / 26.04 repositories. For iHD, use `intel-media-va-driver-non-free` (multiverse, recommended) or the free `intel-media-va-driver` package.

```Shell
sudo apt install --no-install-recommends libva2 libva-drm2 libva-x11-2 intel-media-va-driver-non-free vainfo
```

For the free version, replace the driver package above with `intel-media-va-driver`. Capabilities depend on the GPU and driver version. On i3-N305 with Ubuntu 24.04, the free 24.1 driver exposed only EncSliceLP for H.264 / HEVC, without ICQ / AVBR. The non-free version exposed EncSlice / EncSliceLP and ICQ, plus AVBR for H.264. Check the actual capabilities with `--check-features` in section 5.

For older GPUs using i965, install `i965-va-driver-shaders` (multiverse, with prebuilt shaders), which was verified for encoding on hardware. The universe package `i965-va-driver` does not include shaders and may not encode on some generations; it is not recommended here because operation has not been verified.

```Shell
sudo apt install --no-install-recommends libva2 libva-drm2 libva-x11-2 i965-va-driver-shaders vainfo
```

OpenCL filters and handling resolution changes during avsw input also require an OpenCL runtime that supports the GPU. It is optional for encoding without filters when input resolution remains constant.

```Shell
sudo apt install intel-opencl-icd clinfo
```

For Gen9 GPUs (including HD 630) on Ubuntu 26.04, use [intel-opencl-icd-legacy](https://packages.ubuntu.com/resolute/intel-opencl-icd-legacy) from the standard universe repository instead of the regular `intel-opencl-icd`.

```Shell
sudo apt install intel-opencl-icd-legacy clinfo
```

OpenCL hardware testing on 26.04 used `intel-opencl-icd` 26.31 from the kobuk-team PPA and the Intel-provided runtime in [3-1.](#3-1-when-mixing-gen11-or-earlier-and-gen12-or-later-gpus). Operation with the standard `intel-opencl-icd` 26.05 / `intel-opencl-icd-legacy` packages has not been verified.

Mesa rusticl (`sudo apt install mesa-opencl-icd clinfo`) is another option. `qsvencc` automatically sets `RUSTICL_ENABLE=iris` if the variable is unset. However, it was much slower than Intel OpenCL in the tested environment. If no OpenCL runtime supports the GPU, use encoding without filters.

Official deb packages list `libva-x11-2` as a dependency, so it is needed even for headless operation. They are built on an Ubuntu 20.04 base and use the system libva at runtime. When building from source, use libva compatible with the target environment and install any additional shared libraries listed by `ldd ./qsvencc`. See the [build instructions](./Build.en.md).

### 2. Add the user to GPU access groups

Set up the `video` and `render` groups for QSV / VA-API / OpenCL access. Log out and back in after changing group membership.

```Shell
# QSV
sudo gpasswd -a ${USER} video
# OpenCL
sudo gpasswd -a ${USER} render
```

### 3. Install qsvencc

The official distribution provides a single deb built on an Ubuntu 20.04 base. Use the same deb on Ubuntu 20.04 and later (including 26.04), and on apt-based distributions where the required dependencies are available. GPU-specific drivers and runtimes are still required.

Download deb package from [this link](https://github.com/rigaya/QSVEnc/releases), and install running the following command line. Please note "x.xx" should be replaced to the target version name.

```Shell
sudo apt install ./qsvencc_x.xx_amd64.deb
```

Official deb packages list QSV / OpenCL runtimes under Recommends. For VA-API-only operation, install the driver from section 1-2 and use `sudo apt install --no-install-recommends ./qsvencc_x.xx_amd64.deb` to skip recommended packages. Adjust the filename to the package you downloaded.

### 3-1. When mixing Gen11 or earlier and Gen12 or later GPUs

The following procedure manually installs Intel's `intel-opencl-icd-legacy1` package. This method was also verified for OpenCL on HD 630 with Ubuntu 26.04. Its package name differs from `intel-opencl-icd-legacy` in the standard repository.

On Ubuntu 22.04 / 24.04, when Gen11 or earlier GPUs (e.g. Kaby Lake iGPU) and Gen12 or later GPUs (e.g. Arc dGPU) are mixed in the same PC, intel-opencl-icd from the standard Ubuntu repo does not support newer Gen12 or later GPUs, while intel-opencl-icd from the Intel repository cannot detect Gen11 or earlier OpenCL devices.

In this case, install intel-opencl-icd for Gen12 or later from the Intel repository registered in section 1, and then add the OpenCL runtime for Gen11 or earlier (legacy1) provided by Intel. legacy1 is a separate package with a separate ICD (```/etc/OpenCL/vendors/intel_legacy1.icd```) from the normal intel-opencl-icd, so both can coexist. Note that QSVEncC 8.31 or later is required, which supports environments with multiple Intel OpenCL platforms.

```Shell
mkdir -p ~/neo-legacy1 && cd ~/neo-legacy1
wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17537.24/intel-igc-core_1.0.17537.24_amd64.deb
wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17537.24/intel-igc-opencl_1.0.17537.24_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/24.35.30872.36/intel-opencl-icd-legacy1_24.35.30872.36_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/24.35.30872.36/ww35.sum

# Verify checksum
grep intel-opencl-icd-legacy1_ ww35.sum | sha256sum -c -

sudo apt install ./intel-igc-core_1.0.17537.24_amd64.deb ./intel-igc-opencl_1.0.17537.24_amd64.deb ./intel-opencl-icd-legacy1_24.35.30872.36_amd64.deb

# intel-igc-* is installed to /usr/local/lib, so update the library cache
sudo ldconfig
```

After installation, check that both Gen11 or earlier and Gen12 or later GPUs are listed by ```clinfo -l```.

### 3-2. Check VA-API / OpenCL detection

Use `vainfo` to check that the target GPU lists `VAEntrypointEncSlice` or `VAEntrypointEncSliceLP` for the codec. With multiple GPUs, replace `renderD128` with the target node, such as `renderD129`.

```Shell
vainfo --display drm --device /dev/dri/renderD128
clinfo -l
```

Set `LIBVA_DRIVER_NAME=i965` to use i965; use this setting for `qsvencc` too. When checking rusticl with `clinfo`, explicitly set `RUSTICL_ENABLE=iris`, since it is not set automatically for this tool.

```Shell
LIBVA_DRIVER_NAME=i965 vainfo --display drm --device /dev/dri/renderD128
RUSTICL_ENABLE=iris clinfo -l
```

### 4. Addtional Tools

There are some features which require additional installations.  

| Feature | Requirements |
|:--      |:--           |
| avs reader       | [AvisynthPlus](https://github.com/AviSynth/AviSynthPlus) |
| vpy reader       | [VapourSynth](https://www.vapoursynth.com/)              |
| --vpp-onnx       | OpenVINO Runtime from the [Intel official APT repository](https://docs.openvino.ai/2026/get-started/install-openvino/install-openvino-apt.html) |

To install OpenVINO Runtime from the APT repository:

```Shell
sudo apt-get install -y gnupg wget
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo gpg --output /etc/apt/trusted.gpg.d/intel.gpg --dearmor GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

# Run only one of the following lines, matching your Ubuntu version.

# Ubuntu 24.04
echo "deb https://apt.repos.intel.com/openvino ubuntu24 main" | sudo tee /etc/apt/sources.list.d/intel-openvino.list

# Ubuntu 22.04
echo "deb https://apt.repos.intel.com/openvino ubuntu22 main" | sudo tee /etc/apt/sources.list.d/intel-openvino.list

sudo apt update
sudo apt install openvino
```

### 5. Check detection in qsvencc

Select VA-API explicitly to check encoding availability, device numbers and detailed capabilities.

```Shell
qsvencc --backend vaapi --check-hw
qsvencc --backend vaapi --check-device
qsvencc --backend vaapi --check-features
qsvencc --backend vaapi --avsw -i input.mp4 -c h264 --cqp 25 -o output.mp4
```

Use `qsvencc --backend qsv --check-hw` to check QSV. With VA-API, `-d` assigns QSV's numbers to GPUs available through VPL, and subsequent numbers to other GPUs. If VPL is unavailable (including an explicit disable, a non-iHD driver setting, or a missing iHD driver), Intel render node order is used. After switching backends or drivers, check the numbers with `--backend vaapi --check-device`.

```Shell
LIBVA_DRIVER_NAME=i965 qsvencc --backend vaapi --check-device
LIBVA_DRIVER_NAME=i965 qsvencc --backend vaapi -d 1 --avsw -i input.mp4 -c h264 --cqp 25 -o output.mp4
```

If opening the GPU fails with `Permission denied`, check the `render` / `video` group settings in section 2.

### 6. Others

- Error: "Failed to load OpenCL." when running qsvencc  
  Please check if /lib/x86_64-linux-gnu/libOpenCL.so exists. There are some cases that only libOpenCL.so.1 exists. In that case, please create a link using following command line.
  
  ```Shell
  sudo ln -s /lib/x86_64-linux-gnu/libOpenCL.so.1 /lib/x86_64-linux-gnu/libOpenCL.so
  ```
- Fixed Function(FF) mode not supported
- Unable to encode on Arc GPUs or JasperLake

  The problem might be caused by HuC firmware being not loaded. [See also](https://01.org/linuxgraphics/downloads/firmware)
  
  It is required to load HuC firmware to use FF mode (or Low Power mode).
  Therefore, it is essential to load HuC firmware in oreder to encode on such GPUs which support FF mode only, like Arc GPUs or JasperLake.
   
  Please check whether HuC firmware is loaded.
  ```
  sudo cat /sys/kernel/debug/dri/0/i915_huc_load_status
  ```

  Check also Huc Firmware module is available on your system.
  ```
  sudo modinfo i915 | grep -i "huc"
  ```

  If the module for the CPU gen you are using is available,
  you shall be able to use FF mode by loading HuC Firmware module.

  By adding option below to ```/etc/modprobe.d/i915.conf```, HuC Firmware will be loaded after reboot.
  ```
  options i915 enable_guc=2
  ```


## Linux (Fedora 32)

### 1. Install Intel Media and OpenCL driver  

```Shell
#Media
sudo dnf install intel-media-driver
#OpenCL
sudo dnf install -y 'dnf-command(config-manager)'
sudo dnf config-manager --add-repo https://repositories.intel.com/graphics/rhel/8.3/intel-graphics.repo
sudo dnf update --refresh
sudo dnf install intel-opencl intel-media intel-mediasdk level-zero intel-level-zero-gpu
```
### 2. Add user to proper group to use QSV and OpenCL
```Shell
# QSV
sudo gpasswd -a ${USER} video
# OpenCL
sudo gpasswd -a ${USER} render
```

### 3. Install qsvencc
Download rpm package from [this link](https://github.com/rigaya/QSVEnc/releases), and install running the following command line. Please note "x.xx" should be replaced to the target version name.

```Shell
sudo dnf install ./qsvencc_x.xx_1.x86_64.rpm
```

### 4. Addtional Tools

There are some features which require additional installations.  

| Feature | Requirements |
|:--      |:--           |
| avs reader       | [AvisynthPlus](https://github.com/AviSynth/AviSynthPlus) |
| vpy reader       | [VapourSynth](https://www.vapoursynth.com/)              |

### 5. Others

- Error: "Failed to load OpenCL." when running qsvencc  
  Please check if /lib/x86_64-linux-gnu/libOpenCL.so exists. There are some cases that only libOpenCL.so.1 exists. In that case, please create a link using following command line.
  
  ```Shell
  sudo ln -s /lib/x86_64-linux-gnu/libOpenCL.so.1 /lib/x86_64-linux-gnu/libOpenCL.so
  ```
