# 安装指南

## 1 安装说明

**MindIE-LLM** 是与昇腾系列产品高度亲和的大模型推理框架，安装时请注意版本号一致。

MindIE LLM的安装包含镜像,源码和pip install方式。各安装方案的使用场景如下所示，请根据实际场景选择合适的安装方式。



## 2 安装前准备

### 2.1 硬件环境

| **硬件** | **操作系统** |
|-----|-----|
| Atlas 800I A3 超节点服务器 | AArch64：<br>openEuler 22.03，CULinux 3.0，Kylin V10 SP3 2403 |
| Atlas 800I A2 推理服务器 | AArch64：<br>CentOS 7.6，Ubuntu 24.04 LTS，openEuler 22.03 LTS，openEuler 22.03 LTS SP4，openEuler 24.03 LTS SP1，<br>BCLinux 21.10 U4，CTYunOS 23.01，CULinux 3.0，Kylin V10 GFB，Kylin V10 SP2，Kylin V10 SP3，AliOS3 |
| Atlas 300I Duo 推理卡 + Atlas 800 推理服务器（型号 3000）| AArch64：<br>Ubuntu 20.04，Ubuntu 22.04，openEuler 22.03 LTS SP4，openEuler 24.03 SP1，<br>BCLinux 21.10，Debian 10.8，Kylin V10 SP1，UOS20-1020e |
| Atlas 300I Duo 推理卡 + Atlas 800 推理服务器（型号 3010）| X86_64：<br>Ubuntu 22.04 |

可通过以下命令查询当前操作系统：

```sh
uname -m && cat /etc/*release
```

### 2.2 软件环境

| 软件 | 版本 |
| ---- | ----- |
| [Python](https://www.python.org/) | 3.10 ~ 3.13 |
| [GCC](https://gcc.gnu.org/) | 需支持 C++17 标准 |
| [Cmake](https://cmake.org/) | 版本不能低于 3.19 |
| [git](https://git-scm.com/) | 推荐稳定版本 2.34.x - 2.42.x |
| [Pybind11](https://pypi.org/project/pybind11/) | 推荐使用 3.0 以上版本 |
| [NPU 固件和驱动](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=32&cann=All&driver=Ascend+HDK+25.3.RC1)<sup>*1</sup> | 根据服务器型号下载对应的安装包。 |
| [CANN](https://www.hiascend.cn/developer/download/community/result?module=cann)<sup>*2</sup> | 根据 CPU 架构和 NPU 型号选择 toolkit 和 kernel 包 |
| [PyTorch (NPU)](https://gitcode.com/Ascend/pytorch)<sup>*3</sup> | 根据 Python 版本和 CPU 类型选择版本，支持 2.1 ~ 2.6 |
| [Ascend-Transformer-Boost (ATB)](https://gitcode.com/cann/ascend-transformer-boost)<sup>*4</sup> | 根据 CANN 版本选择相应的安装包 |

<sup>*1</sup>: **固件和驱动** 安装请参见[安装NPU驱动和固件](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_0005.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit)。

- 第一次安装时，先安装 driver，再安装 firmwire，最后执行 reboot 指令重启服务器生效。
- 若服务器上已安装驱动固件，进行版本升级时，先安装 firmwire，再安装 driver，最后执行 reboot 指令重启服务器生效。
- 示例脚本：
  ```sh
  ./Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run --full --install-for-all
  ./Ascend-hdk-<chip_type>-npu-firmware_<version>.run --full
  reboot        # 安装成功后必须重启服务器
  npu-smi info  # 检查是否安装成功
  ```

<sup>*2</sup>: **CANN** 安装请参见[安装CANN](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit)。

- 先安装 toolkit 再安装 kernel。
- 示例脚本：
  ```sh
  ./Ascend-cann-toolkit_<version>_linux-aarch64.run --install --install-path=${HOME}
  ./Atlas-A3-cann-kernels_<version>_linux-aarch64.run --install --install-path=${HOME}
  source ${HOME}/Ascend/cann/set_env.sh
  ```

<sup>*3</sup>: **Pytorch** 安装请参见[ 安装Pytorch](https://gitcode.com/Ascend/pytorch/releases)。
- 示例脚本：
  ```sh
  pip install torch_npu-2.6.0.post4-cp311-cp311-manylinux_2_28_aarch64.whl
  ```

<sup>*4</sup>: **ATB** 安装请参见[安装CANN](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=Debian&Software=cannToolKit)。

- 安装 NNAL 神经网络加速库。
- 示例脚本：
  ```sh
  ./Ascend-cann-nnal_<version>_linux-aarch64.run --install --install-path=${HOME}
  source ${HOME}/Ascend/nnal/atb/set_env.sh
  ```

## 3 安装

### 3.1 使用 Docker 镜像安装

- 默认宿主机环境已成功安装了固件与驱动、Docker（版本要求大于或等于 24.x.x）。
- 下载镜像，根据 CPU 架构和操作系统选择镜像，请参见[MindIE镜像](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f)；
- 加载镜像：`docker load < mindie_*.tar.gz`，或使用 `docker pull xxx` 接口从仓库下载，请参见[MindIE镜像](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f)；
- 使用镜像：通过以下 `start-docker.sh` 脚本创建容器：
    ```sh
    IMAGES_ID=$1
    NAME=$2
    if [ $# -ne 2 ]; then
        echo "error: need two arguments describing your image ID and container name."
        exit 1
    fi
    docker run --name ${NAME} -it -d --net=host --shm-size=100g \
        --privileged=true \
        -w /home \
        --device=/dev/davinci_manager \
        --device=/dev/hisi_hdc \
        --device=/dev/devmm_svm \
        --entrypoint=bash \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /usr/local/sbin:/usr/local/sbin \
        -v /home:/home \
        -v /mnt:/mnt \
        -v /tmp:/tmp \
        -e http_proxy=$http_proxy \
        -e https_proxy=$https_proxy \
        ${IMAGES_ID}
    ```
    查看镜像 ID：
    ```sh
    docker images
    ```
    假设在上一步查询到镜像 ID 是 abcdef12345
    ```sh
    bash start-docker.sh abcdef12345 new_mindie_env
    ```
- 进入容器：
    ```sh
    docker exec -it new_mindie_env bash
    ```

MindIE-LLM 镜像已具备模型运行所需的基础环境，包括：CANN、NNAL、FrameworkPTAdapter、MindIE 与 ATB-Models，可实现模型推理开箱即用。

注意，为方便使用，该镜像中内置了 ATB-Models 压缩包，并放置于/opt/package之下，如需使用，可从镜像中获取。

### 3.2 使用源码安装

默认已经安装了固件、驱动、CANN、NNAL 等依赖包，且网络状况良好。

```sh
git clone https://gitcode.com/Ascend/MindIE-LLM.git
cd MindIE-LLM
```
#### 3.2.1 编译 MindIE-LLM 框架
- 编译

    - release 模式编译
    ```sh
    bash ./build.sh release
    ```
    - debug 模式编译
    ```sh
    bash ./build.sh debug
    ```
    - 只编译所有第三方包
    ```sh
    bash ./build.sh 3rd
    ```
    - 清除构建产物
    ```sh
    bash ./build.sh clean
    ```

    第一次编译耗时较长，请耐心等待。如出现 `c++: fatal error: Killed signal terminated program cc1plus`，说明默认使用全部线程导致内存不足，适当减小线程池大小，例如：
    ```sh
    export MAX_COMPILE_CORE_NUM=8
    ```
- 安装
    ```sh
    ./output/{arch}/Ascend-mindie-llm_{version}_linux-{arch}.run --install --install-path=${HOME}
    source ${HOME}/mindie_llm/set_env.sh
    ```
#### 3.2.2 编译 atb-model 模型仓
- 编译
    ```sh
    cd examples/atb_models
    bash script/build.sh
    source ./output/atb_models/set_env.sh
    ```
    编译产物：`output/Ascend-mindie-atb-models_${VERSION}_linux-${ARCH}_torch${TORCH_VISION}-abi${ABI}.tar.gz`
- 安装
    ```sh
    # 工作目录：${working_dir}
    cd ${working_dir}
    mkdir atb_model
    cd atb_model
    tar -zxvf ../Ascend-mindie-atb-models_${VERSION}_linux-${ARCH}_torch${TORCH_VISION}-abi${ABI}.tar.gz
    pip install atb_llm-{version}-none-any.whl
    ```
- 模型依赖
  
  根据不同模型选择安装不同 Python 包依赖，详细参见 `./examples/atb_models/requirements` 目录
  ```sh
  cd ./examples/atb_models/requirements
  pip install -r ./requirements.txt
  pip install -r ./models/requirements_{模型}.txt
  ```

## 4 卸载

### 4.1 通过脚本卸载

```sh
# git clone https://gitcode.com/Ascend/MindIE-LLM.git
# cd MindIE-LLM
bash scripts/uninstall.sh
```

### 4.2 通过 .run 包卸载

```sh
./软件包名.run --uninstall --install-path=${HOME}
```
