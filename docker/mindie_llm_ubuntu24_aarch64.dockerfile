#============================================= Build Args ==============================================#
# 所有对外暴露的参数及默认值，可以在执行 docker build 时，通过 --build-arg 覆盖。
# example: docker build --build-arg --build-arg  ...
#=======================================================================================================#
ARG TORCH_VERSION=2.6.0
ARG TEMP_DIR=/tmp/build_cache
ARG FILE_SERVER=http://localhost:8000
ARG UBUNTU_MIRROR=http://mirrors.tools.huawei.com/ubuntu-ports/
ARG PIP_INDEX_URL=http://mirrors.tools.huawei.com/pypi/simple
ARG PIP_TRUSTED_HOST=mirrors.tools.huawei.com
ARG PYTHON_REQUIREMENTS_FILE=""

ARG OPENSOURCE_PATH="opensource.tgz"

ARG TORCH_NPU_WHL_PATH

ARG CANN_TOOLKIT_FROM_PATH
ARG CANN_KERNELS_FROM_PATH
ARG CANN_NNAL_FROM_PATH
ARG ASCEND_INSTALL_DIR=/usr/local/Ascend

ARG MINDIE_LLM_FROM_PATH
ARG MINDIE_LLM_INSTALL_DIR=/usr/local/Ascend


#====================================== Stage 1: Build Base Image ======================================#
# 基于 ubuntu 镜像，安装和 Ascend 无关的 Linux 常用工具，制作基础镜像。
#=======================================================================================================#
FROM ubuntu:24.04 AS base

ARG TEMP_DIR
ARG UBUNTU_MIRROR
ARG FILE_SERVER
ARG PIP_INDEX_URL
ARG PIP_TRUSTED_HOST
ARG TORCH_VERSION
ARG PYTHON_REQUIREMENTS_FILE

# 更换 APT 源
RUN echo "Types: deb" > /etc/apt/sources.list.d/ubuntu.sources && \
    echo "URIs: http://mirrors.huaweicloud.com/ubuntu-ports/" >> /etc/apt/sources.list.d/ubuntu.sources && \
    echo "Suites: noble noble-updates noble-security" >> /etc/apt/sources.list.d/ubuntu.sources && \
    echo "Components: main restricted universe multiverse" >> /etc/apt/sources.list.d/ubuntu.sources && \
    echo "Architectures: arm64" >> /etc/apt/sources.list.d/ubuntu.sources

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    PATH="/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin:/sbin:$PATH" \
    LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:/usr/local/lib:${LD_LIBRARY_PATH}"

# 安装开发常用的软件包，可根据实际需要增删
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tzdata locales apt-utils sudo \
        ca-certificates \
        gcc g++ make cmake gfortran \
        python3.12-dev python3-pip python3.12-venv \
        wget curl aria2 unzip \
        xz-utils zlib1g zlib1g-dev libbz2-dev liblzma-dev libffi-dev libssl-dev libsqlite3-dev \
        libxml2 libxslt1-dev pkg-config \
        libblas-dev liblapack-dev libopenblas-dev libhdf5-dev \
        pciutils net-tools ipmitool numactl \
        distro-info-data hwdata lsb-release media-types usb.ids \
        libglx-mesa0 libgmpxx4ldbl \
        vim lcov bc openssh-server git patch file \
    && update-ca-certificates -f \
    && locale-gen en_US.UTF-8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# 设置字符集、编码等，避免中文显示乱码。
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Ubuntu24建议独立管理Python环境
RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/torch/lib:/opt/venv/lib/python3.12/site-packages/torch.libs:$LD_LIBRARY_PATH"

# 设置 PIP 源为内部源，需要设置特定代理才能访问。如果外部不能访问，可根据实际情况自行修改。
# 安装常用的 Python 包，可根据实际需要增删。
RUN echo "[global]" > /etc/pip.conf \
    && echo "index-url = ${PIP_INDEX_URL}" >> /etc/pip.conf \
    && echo "trusted-host = ${PIP_TRUSTED_HOST}" >> /etc/pip.conf \
    && pip install --timeout=600 numpy numba pandas matplotlib scipy tqdm requests pillow pybind11 \
        torch==${TORCH_VERSION} transformers \
        wheel psutil pydantic

RUN if [ -n "${PYTHON_REQUIREMENTS_FILE}" ]; then \
        echo "Installing Python requirements from ${PYTHON_REQUIREMENTS_FILE}" \
        && mkdir -p ${TEMP_DIR} && cd ${TEMP_DIR} \
        && wget --no-proxy -nv ${FILE_SERVER}/${PYTHON_REQUIREMENTS_FILE} \
        && pip install --no-cache-dir -r "${PYTHON_REQUIREMENTS_FILE}"; \
    else \
        echo "No PYTHON_REQUIREMENTS_FILE provided, skip pip install."; \
    fi

# 创建 mindie 普通用户和组
RUN groupadd -g 2000 mindie \
    && useradd -m -u 2000 -g mindie -s /bin/bash mindie \
    && echo 'mindie ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# 默认使用普通用户。
USER mindie
WORKDIR /home/mindie

CMD [ "/bin/bash" ]


#================================== Stage 2: Build MindIE Base Image ===================================#
# 在安装 PTA 的镜像基础上，安装 MindIE 依赖。
#=======================================================================================================#
FROM base AS mindie_base

ARG TEMP_DIR
ARG FILE_SERVER
ARG OPENSOURCE_PATH

USER root

RUN mkdir -p ${TEMP_DIR} && cd ${TEMP_DIR} \
    && wget --no-proxy -nv ${FILE_SERVER}/${OPENSOURCE_PATH} \
    && mkdir -p /opt/opensource \
    && tar -zxf ${OPENSOURCE_PATH} --strip-components=1 -C /opt/opensource \
    && cd - && rm -rf ${TEMP_DIR}

ENV PATH="/opt/opensource/grpc/bin:/opt/opensource/hseceasy/bin:/opt/opensource/openssl/bin:$PATH" \
    LD_LIBRARY_PATH="/opt/opensource/boost/lib:/opt/opensource/grpc/lib:/opt/opensource/libboundcheck/lib:/opt/opensource/openssl/lib:/opt/opensource/prometheus-cpp/lib:/opt/opensource/hseceasy/lib:$LD_LIBRARY_PATH"

USER mindie

#====================================== Stage 3: Build CANN Image ======================================#
# 在基础镜像中安装 CANN。
#=======================================================================================================#
FROM mindie_base AS cann

ARG TEMP_DIR
ARG FILE_SERVER
ARG ASCEND_INSTALL_DIR
ARG CANN_TOOLKIT_FROM_PATH
ARG CANN_KERNELS_FROM_PATH
ARG CANN_NNAL_FROM_PATH

USER root

ENV ASCEND_HOME_PATH=${ASCEND_INSTALL_DIR}/ascend-toolkit/latest \
    ASCEND_TOOLKIT_HOME=${ASCEND_INSTALL_DIR}/ascend-toolkit/latest \
    ATB_HOME_PATH=${ASCEND_INSTALL_DIR}/nnal/atb/latest/atb/cxx_abi_1

# 下载并安装 CANN，由于 CANN 体积较大，安装后镜像膨胀 10G 左右。
RUN mkdir -p ${TEMP_DIR} && cd ${TEMP_DIR} \
    && wget --no-proxy -nv ${FILE_SERVER}/${CANN_TOOLKIT_FROM_PATH} \
    && wget --no-proxy -nv ${FILE_SERVER}/${CANN_KERNELS_FROM_PATH} \
    && wget --no-proxy -nv ${FILE_SERVER}/${CANN_NNAL_FROM_PATH} \
    && bash ${CANN_TOOLKIT_FROM_PATH} --install --install-path=${ASCEND_INSTALL_DIR} --quiet \
    && bash ${CANN_KERNELS_FROM_PATH} --install --install-path=${ASCEND_INSTALL_DIR} --quiet \
    && bash ${CANN_NNAL_FROM_PATH} --install --install-path=${ASCEND_INSTALL_DIR} --quiet \
    && cd - && rm -rf ${TEMP_DIR} \
    && echo "source ${ASCEND_INSTALL_DIR}/ascend-toolkit/set_env.sh" >> /etc/profile \
    && echo "source ${ASCEND_INSTALL_DIR}/nnal/atb/set_env.sh --cxx_abi=1" >> /etc/profile

USER mindie

#====================================== Stage 4: Build PTA Image =======================================#
# 在安装 CANN 的镜像基础上，安装 PTA，至此，完成MindIE编译所需依赖。
#=======================================================================================================#
FROM cann AS mindie_compile

ARG TEMP_DIR
ARG FILE_SERVER
ARG TORCH_NPU_WHL_PATH

USER root

# 安装 torch-npu
RUN mkdir -p ${TEMP_DIR} && cd ${TEMP_DIR} \
    && wget --no-proxy -nv ${FILE_SERVER}/${TORCH_NPU_WHL_PATH} \
    && pip install ${TORCH_NPU_WHL_PATH} --no-cache-dir \
    && cd - && rm -rf ${TEMP_DIR}

ENV LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/torch_npu/lib:$LD_LIBRARY_PATH"

USER mindie

#===================================== Stage 5: Build MindIE Image =====================================#
# 在安装 pta 的镜像基础上，编译安装 MindIE-LLM
#=======================================================================================================#
FROM mindie_compile AS mindie

ARG TEMP_DIR
ARG FILE_SERVER
ARG MINDIE_LLM_FROM_PATH
ARG MINDIE_LLM_INSTALL_DIR

USER root

# 安装 torch-npu
RUN mkdir -p ${TEMP_DIR} && cd ${TEMP_DIR} \
    && wget --no-proxy -nv ${FILE_SERVER}/${MINDIE_LLM_FROM_PATH} \
    && bash ${MINDIE_LLM_FROM_PATH} --install --install-path=${MINDIE_LLM_INSTALL_DIR} \
    && cd - && rm -rf ${TEMP_DIR}

ENV PATH="${MINDIE_LLM_INSTALL_DIR}/bin:$bin" \
    LD_LIBRARY_PATH="${MINDIE_LLM_INSTALL_DIR}/lib:$LD_LIBRARY_PATH" \
    PYTHONPATH="${MINDIE_LLM_INSTALL_DIR}/lib/python3.12/site-packages:$PYTHONPATH"

USER mindie


# For quickly debug, use root user.
USER root
