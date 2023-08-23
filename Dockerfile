FROM ubuntu:jammy
ENV TZ=UTC \
    DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt install -qq -y build-essential libtool pkg-config git cmake
RUN apt install -y gpg-agent wget software-properties-common
RUN wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | \
    gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg
RUN echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu jammy main' | \
    tee  /etc/apt/sources.list.d/intel.gpu.focal.list

RUN apt update
RUN apt install -y intel-media-va-driver-non-free intel-opencl-icd opencl-headers
RUN apt install -y \
  libva-drm2 \
  libva-x11-2 \
  libva-glx2 \
  libx11-dev \
  libigfxcmrt7 \
  libva-dev \
  libdrm-dev \
  ffmpeg \
  libavcodec-extra libavcodec-dev libavutil-dev libavformat-dev libswresample-dev libavfilter-dev libavdevice-dev \
  libass9 libass-dev
RUN git clone https://github.com/rigaya/QSVEnc --recursive --depth=1 && cd QSVEnc && ./configure && make -j`nproc`

