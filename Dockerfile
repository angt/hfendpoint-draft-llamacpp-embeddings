FROM ubuntu:noble

ARG TARGETOS
ARG TARGETARCH

ARG llamacpp_version=b5061
ARG llamacpp_native=ON
ARG llamacpp_cpu_arm_arch=native
ARG llamacpp_backend_dl=OFF
ARG llamacpp_cpu_all_variants=OFF
ARG llamacpp_openmp=OFF
ARG msgpack_version=6.1.0
ARG hfendpoint_version=0.1.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    clang \
    cmake \
    curl \
    ca-certificates \
    make \
    git \
    pkg-config \
    tar \
    libomp-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /deps

ADD https://github.com/ggml-org/llama.cpp/archive/refs/tags/${llamacpp_version}.tar.gz /deps
RUN tar -xzf ${llamacpp_version}.tar.gz \
 && cmake -S llama.cpp-${llamacpp_version} -B llama.cpp.build \
        -DCMAKE_INSTALL_PREFIX=/usr \
        -DGGML_NATIVE=${llamacpp_native} \
        -DGGML_CPU_ARM_ARCH=${llamacpp_cpu_arm_arch} \
        -DGGML_BACKEND_DL=${llamacpp_backend_dl} \
        -DGGML_CPU_ALL_VARIANTS=${llamacpp_cpu_all_variants} \
        -DGGML_CCACHE=OFF \
        -DLLAMA_OPENMP=${llamacpp_openmp} \
        -DLLAMA_BUILD_COMMON=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=OFF \
 && cmake --build llama.cpp.build --parallel --config Release \
 && cmake --install llama.cpp.build

ADD https://github.com/msgpack/msgpack-c/releases/download/c-${msgpack_version}/msgpack-c-${msgpack_version}.tar.gz /deps
RUN tar -xzf msgpack-c-${msgpack_version}.tar.gz \
 && cmake -S msgpack-c-${msgpack_version} -B msgpack.build \
        -DCMAKE_INSTALL_PREFIX=/usr \
 && cmake --build msgpack.build --parallel --config Release \
 && cmake --install msgpack.build

RUN case ${TARGETARCH} in (amd64) ARCH=x86_64;; (arm64) ARCH=aarch64;; esac \
 && curl -sSLf https://github.com/angt/hfendpoint-draft/releases/download/v${hfendpoint_version}/hfendpoint-${ARCH}-linux.gz \
        | gunzip > /bin/hfendpoint \
 && chmod +x /bin/hfendpoint

COPY Makefile worker.c .
RUN make

WORKDIR /app
RUN cp /deps/worker /app \
 && rm -rf /deps

ENTRYPOINT ["hfendpoint", "/app/worker"]
