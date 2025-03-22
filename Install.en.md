
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
  
## Linux (Ubuntu 20.04 - 24.04)

### 1. Add repository for Intel Media driver  

> [!NOTE]
> Please skip this step on Ubuntu 24.04 as Intel Media driver can be installed by default.

Intel media driver can be installed following instruction on [this link](https://dgpu-docs.intel.com/driver/client/overview.html).

First, install required tools.

```Shell
sudo apt-get install -y gpg-agent wget
```

Next, add Intel package repository.

```Shell
# Ubuntu 22.04
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy unified" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list

# Ubuntu 20.04
wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg
echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics-keyring.gpg] https://repositories.intel.com/gpu/ubuntu focal client' | \
  sudo tee /etc/apt/sources.list.d/intel-graphics.list
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

# Ubuntu 20.04
sudo apt install ./qsvencc_x.xx_Ubuntu20.04_amd64.deb
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