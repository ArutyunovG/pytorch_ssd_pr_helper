FROM nvidia/cuda:9.2-cudnn7-devel

ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN \
apt-get update && \
apt-get install -yq \
    build-essential git cmake \
    python python-pip python-opencv \
    libopencv-dev

RUN python -m pip install pyyaml typing

RUN git clone --recursive https://github.com/ArutyunovG/pytorch
RUN cd pytorch && \
    git checkout ssd_infer && \
    git submodule update --recursive

RUN cd pytorch && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DCMAKE_INSTALL_PREFIX=install && \
    make -j$(nproc) install

RUN python -m pip install protobuf future

RUN cp pytorch/build/install/lib/libcaffe2.so \
    pytorch/build/install/lib/libcaffe2_gpu.so \
    pytorch/build/install/lib/libc10.so \
    pytorch/build/install/lib/libc10_cuda.so \
    pytorch/build/install/lib/python2.7/dist-packages/caffe2/python/

ENV PYTHONPATH=/pytorch/build/install/lib/python2.7/dist-packages

RUN apt install -yq wget

RUN git clone \
    https://github.com/ArutyunovG/pytorch_ssd_pr_helper.git
