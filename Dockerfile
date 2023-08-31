FROM ubuntu:jammy
ENV TZ=UTC \
    DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt install -qq -y build-essential libtool pkg-config git cmake
RUN apt install -y gpg-agent wget software-properties-common
RUN wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | \
    gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg && \
    echo 'deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu jammy arc' | \
    tee  /etc/apt/sources.list.d/intel.gpu.jammy.list

RUN apt update
RUN apt install -y intel-media-va-driver-non-free intel-opencl-icd opencl-headers intel-level-zero-gpu level-zero libmfx1 libmfxgen1
RUN apt install -y \
  libva-drm2 \
  libva-x11-2 \
  libva-glx2 \
  libx11-dev \
  libigfxcmrt7 \
  libva-dev \
  libdrm-dev \
  libfdk-aac-dev \
  libass-dev curl g++ make nasm yasm
#RUN cd /tmp && git clone -b n4.5-dev --depth=1 https://github.com/FFmpeg/FFmpeg && cd FFmpeg && ./configure --prefix=/usr --enable-shared --enable-gpl --enable-nonfree --enable-small --enable-libfdk-aac --enable-libass && make -s -j`nproc` && make install && rm -r ../FFmpeg
RUN cd /tmp && curl https://www.ffmpeg.org/releases/ffmpeg-4.4.2.tar.xz |tar xJf - && cd ffmpeg-4.4.* && ./configure --prefix=/usr --enable-shared --enable-nonfree --enable-libfdk-aac --enable-libass && make -s -j`nproc` && make install && rm -r ../ffmpeg-4.4.*
#RUN git clone https://github.com/rigaya/QSVEnc --recursive --depth=1
#RUN cd QSVEnc && ./configure --prefix=/usr && make -s -j`nproc` && make install && rm -r ../QSVEnc
ADD https://github.com/rigaya/QSVEnc/releases/download/7.48/qsvencc_7.48_Ubuntu22.04_amd64.deb /qsvencc.deb
RUN dpkg -i --force-depends /qsvencc.deb && rm qsvencc.deb
ENTRYPOINT ["/usr/bin/qsvencc"]
