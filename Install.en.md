
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

On Linux, QSV and VA-API can be used to encode with Intel GPUs.

| Method | QSV | VA-API |
|:--|:--|:--|
| Required runtime | VPL (`libmfx-gen`) for Tiger Lake and later, Media SDK (`libmfx1`) for earlier supported GPUs | VA driver (`iHD` / `i965`) only. VPL / Media SDK are not required |
| HW decode (`--avhw`) | Supported | Not supported. `--avsw` and automatic reader selection use software decode |
| Filters | MFX VPP / OpenCL | OpenCL only. Without OpenCL, encode without filters |
| Encode settings | Detailed MFX settings | Basic settings such as rate control, GOP and quality only. Some options are not supported |

See [--backend](./QSVEncC_Options.en.md#--backend-autoqsvvaapi) for details.

Use VA-API for GPUs / distributions without a VPL runtime, and for older generations that can only use i965. For example, Ubuntu 26.04 does not provide `libmfx1`, so Gen11 and earlier iGPUs have no VPL runtime there.

The default `--backend auto` tries QSV, and switches to VA-API if the device is not available. VA-API is selected without trying QSV in the following cases.

- `QSVENC_VPL_DISABLE=1` is set.
- `LIBVA_DRIVER_NAME` is set to a value other than `iHD` (case-insensitive).
- `iHD_drv_video.so` is not found in the driver search path (e.g. environments with i965 only).

If the VPL runtime crashes at startup, use `QSVENC_VPL_DISABLE=1 qsvencc ...` to avoid loading VPL. Use `--backend vaapi` to select VA-API explicitly, or `--backend qsv` to select QSV explicitly.

### 1. Drivers

Available methods and required packages depend on the GPU generation.

| GPU | QSV | VA-API |
|:--|:--|:--|
| Gen12 and later<br>(Tiger Lake, Rocket Lake, Alder Lake, Raptor Lake, Arc dGPU, etc.) | iHD + VPL | iHD |
| Gen8 - Gen11<br>(Broadwell, Skylake, Kaby Lake, Coffee Lake, Apollo Lake, Gemini Lake, Ice Lake, Elkhart Lake, etc.) | iHD + Media SDK<br>(not available on Ubuntu 26.04, as `libmfx1` is not provided) | iHD |
| Haswell and earlier | Not available | i965 |

The package source depends on the Ubuntu version.

| Ubuntu | Source |
|:--|:--|
| 22.04 / 24.04 | Intel repository (register in 1-1.) |
| 26.04 | Ubuntu standard repository |

#### 1-1. Register the Intel repository (Ubuntu 22.04 / 24.04)

Register the repository following [this link](https://dgpu-docs.intel.com/driver/client/overview.html). It can be registered regardless of the GPU generation (for OpenCL on Gen11 and earlier, see the table in 2-1.).

```Shell
sudo apt-get install -y gpg-agent wget

# Ubuntu 24.04
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble unified" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu-noble.list

# Ubuntu 22.04
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy unified" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list

sudo apt update
```

#### 1-2. Install drivers

Install the packages required in the table of 1.

| Component | Package |
|:--|:--|
| iHD | `intel-media-va-driver-non-free` |
| VPL | `libmfxgen1` (Intel repository) / `libmfx-gen1.2` (Ubuntu standard) |
| Media SDK | `libmfx1` |
| i965 | `i965-va-driver-shaders` |

```Shell
# Common
sudo apt install libva2 libva-drm2 libva-x11-2 vainfo

# Gen12 and later (QSV / VA-API)
sudo apt install intel-media-va-driver-non-free libmfxgen1      # 22.04 / 24.04 (Intel repository)
sudo apt install intel-media-va-driver-non-free libmfx-gen1.2   # 26.04

# Gen8 - Gen11
sudo apt install intel-media-va-driver-non-free libmfx1         # 22.04 / 24.04 (QSV / VA-API)
sudo apt install intel-media-va-driver-non-free                 # 26.04 (VA-API only, as libmfx1 is not provided)

# Haswell and earlier (VA-API only)
sudo apt install i965-va-driver-shaders
```

For i965, use `i965-va-driver-shaders` (includes prebuilt shaders).

### 2. OpenCL

OpenCL filters require an OpenCL runtime that supports the GPU. If no OpenCL runtime supports the GPU, encode without filters.

#### 2-1. Intel (recommended)

The package depends on the GPU generation and the Ubuntu version.

| GPU | 22.04 / 24.04 (Intel repository) | 26.04 |
|:--|:--|:--|
| Gen12 and later | `intel-opencl-icd` | `intel-opencl-icd` |
| Gen9 - Gen11 | legacy1 (distributed by Intel, installed manually as below) | `intel-opencl-icd-legacy` |
| Mixed | Use both of the above | Use both of the above |
| Gen8 and earlier | None | None |

`intel-opencl-icd` in the Intel repository supports Gen12 and later only, so install the legacy runtime separately for Gen9 - Gen11. The legacy runtime is a separate package with a separate ICD from the one for Gen12 and later, so both can coexist. QSVEncC 8.31 or later is required for environments with multiple Intel OpenCL platforms.

```Shell
# Gen12 and later
sudo apt install intel-opencl-icd clinfo

# Gen9 - Gen11 (26.04)
sudo apt install intel-opencl-icd-legacy clinfo
```

Install legacy1 for Gen9 - Gen11 on 22.04 / 24.04 as below. Its package name differs from `intel-opencl-icd-legacy` in the Ubuntu standard repository.

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

After installation, check that the target GPUs are listed by `clinfo -l`.

#### 2-2. Mesa

Mesa OpenCL (rusticl) can also be used. However, it was much slower than Intel OpenCL in the tested environment.

```Shell
sudo apt install mesa-opencl-icd clinfo
```

Intel GPUs are disabled in rusticl by default, but `qsvencc` automatically sets `RUSTICL_ENABLE=iris` if the variable is unset.

### 3. Add the user to groups

`video` and `render` groups are required to use QSV / VA-API / OpenCL. Log in again after the change.

```Shell
sudo gpasswd -a ${USER} video
sudo gpasswd -a ${USER} render
```

### 4. Install qsvencc

A common deb is used for Ubuntu 20.04 and later (including 26.04) and apt-based distributions where the required dependencies can be installed.

Download the deb file from [this link](https://github.com/rigaya/QSVEnc/releases), and install as below. Replace "x.xx" with the version to install.

```Shell
sudo apt install ./qsvencc_x.xx_amd64.deb
```

### 5. Check QSV / VA-API / OpenCL detection

Check with `vainfo` that `VAEntrypointEncSlice` or `VAEntrypointEncSliceLP` is listed for the codecs of the target GPU. With multiple GPUs, replace `renderD128` with the target node (e.g. `renderD129`). Check OpenCL with `clinfo -l`.

```Shell
vainfo --display drm --device /dev/dri/renderD128
clinfo -l
```

Set `LIBVA_DRIVER_NAME=i965` to use i965; use this setting for `qsvencc` too. When checking rusticl with `clinfo`, set `RUSTICL_ENABLE=iris` explicitly, since it is not set automatically for this tool.

```Shell
LIBVA_DRIVER_NAME=i965 vainfo --display drm --device /dev/dri/renderD128
RUSTICL_ENABLE=iris clinfo -l
```

Check encoding availability, device numbers and detailed capabilities with qsvencc.

```Shell
# QSV
qsvencc --backend qsv --check-hw
qsvencc --backend qsv --check-device
qsvencc --backend qsv --check-features

# VA-API
qsvencc --backend vaapi --check-hw
qsvencc --backend vaapi --check-device
qsvencc --backend vaapi --check-features
```

With VA-API, `-d` assigns QSV's numbers to GPUs available through VPL, and subsequent numbers to other GPUs. If VPL is unavailable (including an explicit disable, a non-iHD driver setting, or a missing iHD driver), Intel render node order is used. After switching backends or drivers, check the numbers with `--backend vaapi --check-device`.

```Shell
LIBVA_DRIVER_NAME=i965 qsvencc --backend vaapi --check-device
LIBVA_DRIVER_NAME=i965 qsvencc --backend vaapi -d 1 --avsw -i input.mp4 -c h264 --cqp 25 -o output.mp4
```

If opening the GPU fails with `Permission denied`, check the `render` / `video` group settings in 3.

### 6. Additional Tools

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

### 7. Others

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
