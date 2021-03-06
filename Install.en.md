
# How to install QSVEncC

- [Windows 10](./Install.en.md#windows)
- Linux
  - [Linux (Ubuntu 20.04)](./Install.en.md#linux-ubuntu-2004)
  - [Linux (Fedora 32)](./Install.en.md#linux-fedora-32)
  - Other Linux OS  
    For other Linux OS, building from source will be needed. Please check the [build instrcutions](./Build.en.md).


## Windows 10

### 1. Install Intel Graphics driver
### 2. Download Windows binary  
Windows binary can be found from [this link](https://github.com/rigaya/QSVEnc/releases). QSVEncC_x.xx_Win32.7z contains 32bit exe file, QSVEncC_x.xx_x64.7z contains 64bit exe file.

QSVEncC could be run directly from the extracted directory.
  
## Linux (Ubuntu 20.04)

### 1. Install Intel Media driver  
OpenCL driver can be installed following instruction on [this link](https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-focal.html).

```Shell
sudo apt-get install -y gpg-agent wget
wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | sudo apt-key add -
sudo apt-add-repository 'deb [arch=amd64] https://repositories.intel.com/graphics/ubuntu focal main'
sudo apt-get update
sudo apt install intel-media-va-driver-non-free intel-opencl-icd intel-level-zero-gpu level-zero
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