
# How to install QSVEncC

- [Windows 10](./Install.en.md#windows)
- Linux
  - [Linux (Ubuntu 20.04 - 24.04)](./Install.en.md#linux-ubuntu-2004---2404)
  - [Linux (Fedora 32)](./Install.en.md#linux-fedora-32)
  - Other Linux OS  
    For other Linux OS, building from source will be needed. Please check the [build instrcutions](./Build.en.md).


## Windows 10

### 1. Install Intel Graphics driver
### 2. Download Windows binary  
Windows binary can be found from [this link](https://github.com/rigaya/QSVEnc/releases). QSVEncC_x.xx_Win32.7z contains 32bit exe file, QSVEncC_x.xx_x64.7z contains 64bit exe file.

QSVEncC could be run directly from the extracted directory.
  
## Linux (Ubuntu 22.04 - 24.04)

### 1. Add repository for Intel Media driver  

:::note warn  
**Please skip this section and proceed to section 2.**

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

### 2. Add user to proper group to use QSV and OpenCL
```Shell
# QSV
sudo gpasswd -a ${USER} video
# OpenCL
sudo gpasswd -a ${USER} render
```

### 3. Install qsvencc
Download deb package from [this link](https://github.com/rigaya/QSVEnc/releases), and install running the following command line. Please note "x.xx" should be replaced to the target version name.

```Shell
# Ubuntu 24.04
sudo apt install ./qsvencc_x.xx_Ubuntu24.04_amd64.deb

# Ubuntu 22.04
sudo apt install ./qsvencc_x.xx_Ubuntu22.04_amd64.deb
```

### 3-1. When mixing Gen11 or earlier and Gen12 or later GPUs

When Gen11 or earlier GPUs (e.g. Kaby Lake iGPU) and Gen12 or later GPUs (e.g. Arc dGPU) are mixed in the same PC, intel-opencl-icd from the standard Ubuntu repo does not support newer Gen12 or later GPUs, while intel-opencl-icd from the Intel repository cannot detect Gen11 or earlier OpenCL devices.

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

### 3-2. Packages required for VA-API-only operation

When using only `--backend vaapi` or the VA-API path selected by `auto`, oneVPL / Media SDK runtimes (`libvpl2`, `libmfx-gen1.2`, etc.) are not required. On Ubuntu 24.04, install the VA-API libraries and Intel driver from the standard repository:

```Shell
sudo apt install --no-install-recommends libva2 libva-drm2 libva-x11-2 intel-media-va-driver
```

Existing prebuilt binaries link to `libva-x11-2`, so install it even for headless operation. For binaries built from source, also install any additional shared libraries listed by `ldd ./qsvencc`. These are runtime requirements for the VA-API path; official deb packages also declare dependencies for QSV / OpenCL and will install those packages as well.

The runtime libva must also be compatible with the build. A binary built against a newer libva that references `vaMapBuffer2` cannot start with Ubuntu 24.04's standard libva 2.20. Use a compatible libva or build against the libva provided by the target environment.

To use OpenCL filters, additionally install an OpenCL runtime appropriate for the GPU, such as `intel-opencl-icd`. It is optional for encoding without filters. Follow the group settings in section 2 so the user can access `/dev/dri/renderD*`.

Encoding capabilities may differ between the free `intel-media-va-driver` and its non-free variant. Check the actual driver's capabilities with `qsvencc --backend vaapi --check-features`.

```Shell
qsvencc --backend vaapi --check-features
qsvencc --backend vaapi --avsw -i input.mp4 -c h264 --cqp 25 -o output.mp4
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

### 5. Others

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
