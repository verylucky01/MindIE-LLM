# MindIE LLM


## 🔥Latest News

- [2025/12] MindIE LLM启动开源。

## 🚀简介

MindIE LLM（Mind Inference Engine Large Language Model，大语言模型）是MindIE下的大语言模型推理组件，基于昇腾硬件提供业界通用大模型推理能力，同时提供多并发请求的调度功能，支持Continuous Batching、Page Attention、FlashDecoding等加速特性，使能用户高性能推理需求。

MindIE LLM主要提供大模型推理和大模型调度C++ API。

MindIE LLM架构图如下所示：

![](./docs/zh/figures/mindie_llm_architecture_diagram.png)

MindIE LLM总体架构分为四层：Server、LLM Manager、Text Generator和Modeling。
- Server：推理服务端，提供模型推理服务化能力。EndPoint面向推理服务开发者提供RESTful接口，推理服务化协议和接口封装，支持Triton/OpenAI/TGI/vLLM主流推理框架请求接口。

- LLM Manager：负责状态管理及任务调度，基于调度策略实现用户请求组batch，统一内存池管理kv缓存，返回推理结果，提供状态记录接口。
    - LLM manager Interface：MindIE LLM推理引擎的对外接口。

    - Engine：负责将schedule，executor，worker等协同串联起来，利用组件间的协同，实现多场景下请求的推理处理能力。
    - Scheduler: 在1个DP域内，将多条请求在Prefill或者Decode阶段组成batch，实现计算和通信的充分利用。
    - Block manager：管理在DP内的kv资源，支持池化后，支持对offload的kv位置感知。
    - Executor：将调度完成的信息分发给Text Generator模块。支持跨机、跨卡的任务下发。

- Text Generator：负责模型配置、初始化、加载、自回归推理流程、后处理等，向LLM Manager提供统一的自回归推理接口，支持并行解码插件化运行。
    - Preprocess：将调度的任务转换为模型的输入。
    - Generator：对模型运行过程的抽象。
    - Sampler：对模型输出的logits做token选择、停止判断、上下文更新与清除。

- Modeling：提供性能调优后的模块和内置模型，支持ATB Models（Ascend Transformer Boost Models）和MindSpore Models两种框架。

    - 内置模块包括Attention、Embedding、ColumnLinear、RowLinear、MLP（multilayer perceptron），支持Weight在线Tensor切分加载。

    - 内置模型使用内置模块进行组网拼接，支持Tensor切分，支持多种量化方式，用户亦可参照样例通过内置模块组网自定义模型。

    - 组网后的模型经过编译优化后，会生成能在昇腾NPU设备上加速推理的可执行图。

## 🔍目录结构







## 版本说明

|MindIE软件版本|CANN版本兼容性|MindCluster|Ascend Extension for Pytorch|CCAE|
|:---|:---|:---|:---|:---|
|2.3.0|8.5.0|7.2.RC1.SPC1|7.3.0|iMaster CCAE V100R025C20SPC011|

## ⚡️环境部署

MindIE LLM安装前的相关软硬件环境准备，以及安装步骤，请参见[安装指南](./docs/zh/user_guide/installation_guide.md)。


## ⚡️快速入门

  快速体验使用MindIE进行大模型推理的全流程，请参见[快速入门](docs/zh/user_guide/quick_start.md)。

## 📝学习教程

- [LLM使用指南](./docs/zh/user_guide/user_manual/README.md)：MindIE LLM使用指南，包括推理参数配置、在线和离线推理、参数调优等。
- [特性介绍](./docs/zh/user_guide/feature/README.md)：介绍MindIE LLM支持的相关特性。
- [模型支持列表](./docs/zh/user_guide/model_support_list.md)：MindIE LLM支持的模型。

## 📝免责声明

版权所有© 2025 MindIE Project.

您对"本文档"的复制、使用、修改及分发受知识共享（Creative Commmons）署名——相同方式共享4.0国际公共许可协议（以下简称"CC BY-SA 4.0"）的约束。为了方便用户理解，您可以通过访问https://creativecommons.org/licenses/by-sa/4.0/了解CC BY-SA 4.0的概要（但不是替代）。CC BY-SA 4.0的完整协议内容您可以访问如下网址获取：https://creativecommons.org/licenses/by-sa/4.0/legalcode。

## 📝相关信息

- [安全声明](./security.md)
- [LICENSE](LICENSE.md)
