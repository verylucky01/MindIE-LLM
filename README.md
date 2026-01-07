# MindIE LLM


## 🔥Latest News

- [2025/12] MindIE LLM 正式宣布开源并面向公众开放！ [会议日历](https://meeting.ascend.osinfra.cn/?sig=sig-MindIE-LLM)


## 🚀简介

MindIE LLM（Mind Inference Engine Large Language Model，大语言模型）是MindIE下的大语言模型推理组件，基于昇腾硬件提供业界通用大模型推理能力，同时提供多并发请求的调度功能，支持Continuous Batching、Page Attention、FlashDecoding等加速特性，使能用户高性能推理需求。

MindIE LLM主要提供大模型推理和大模型调度C++ API。

MindIE LLM架构图如下所示：

![](./docs/zh/figures/mindie_llm_architecture_diagram.png)

MindIE LLM总体架构分为四层：Server、LLM Manager、Text Generator和Modeling。
- Server：推理服务端，提供模型推理服务化能力。EndPoint面向推理服务开发者提供RESTful接口，推理服务化协议和接口封装，支持Triton/OpenAI/TGI/vLLM主流推理框架请求接口。

- LLM Manager：负责状态管理及任务调度，基于调度策略实现用户请求组batch，统一内存池管理kv缓存，返回推理结果，提供状态记录接口。
    - LLM manager Interface：MindIE LLM推理引擎的对外接口。

    - Engine：负责将scheduler，executor，worker等协同串联起来，利用组件间的协同，实现多场景下请求的推理处理能力。
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

```
├── mindie_llm                                     # Python 推理框架主模块
│   ├── connector                                  # 请求接入层
│   ├── text_generator                             # 核心推理引擎
│   │   ├── adapter                                # 后端适配 
│   │   ├── plugins                                # 高阶特性插件
│   │   │   ├── prefix_cache                       # Prefix Cahce (KV缓存复用)
│   │   │   ├── splitfuse                          # SplitFuse 
│   │   │   ├── memory_decoding                    # Memory Decoding
│   │   │   ├── la                                 # Lookahead 
│   │   ├── cpp                                    # C++ 扩展（Sampler/Prefix Tree/Memory Bridge）
│   ├── modeling                                   # 模型封装抽象
│   │   ├── model_wrapper/atb                      # ATB 模型后端支持
│   ├── utils                                      # 工具模块：日志/张量/profilling/验证等
├── examples                                       # 示例代码
│   ├── atb_models                                 # ATB 模型集成与打包入口
│   │   ├── atb_framework                          # ATB 底层框架
│   │   ├── atb_llm                                # 面向 LLM 的 ATB 封装
│   │   ├── examples                               # 运行示例
│   │   ├── scripts                                # 构建脚本
│   │   ├── tests                                  # 单元/集成测试
│   │   ├── setup.py                               # Python 包构建入口
│   │   ├── CMakeLists.txt                         # CMake 构建配置
│   ├── ms_models                                  # MindSpore 模型示例
│   ├── pt_models                                  # PyTorch 模型示例
├── docs                                           # 项目文档介绍
├── src                                            # C++ 核心引擎
│   ├── engine                                     # LLM 引擎主逻辑（调度/执行）
│   ├── scheduler                                  # 调度器（FCFS/PDDS/Layerwise）
│   ├── block_manager                              # KV Cache 块管理（LRU/Prefix Chche/CoW）
│   ├── llm_manager                                # Python/C++ 桥接API 
│   ├── server                                     # 服务端 （gRPC/HTTP接入点）
│   ├── utils                                      # 基础工具（共享内存/加密/日志/ID 生成等）
│   ├── include                                    # 对外头文件接口
├── scripts                                        # 构建与部署脚本
├── tools                                          # 辅助工具
│   ├── llm_manager_python_api_demo                # Python API 使用示例
├── tests                                          # 测试
├── ...                                            
├── CMakeLists.txt                                 #  CMake 构建配置                         
├── README.md   
├── requirements.txt                               # Python 安装依赖                               
```
## ⚡️版本说明

|MindIE软件版本&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|CANN版本兼容性&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|
|:----------------------------|:----------------------------|
|2.2.RC1|8.3.RC2|
## ⚡️环境部署

MindIE LLM安装前的相关软硬件环境准备，以及安装步骤，请参见[安装指南](./docs/zh/user_guide/installation_guide.md)。


## ⚡️快速入门

  快速体验使用MindIE进行大模型推理的全流程，请参见[快速入门](docs/zh/user_guide/quick_start.md)。

## 📝学习文档

- [LLM使用指南](./docs/zh/user_guide/user_manual/README.md)：MindIE LLM使用指南，包括推理参数配置、在线和离线推理、参数调优等。
- [特性介绍](./docs/zh/user_guide/feature/README.md)：介绍MindIE LLM支持的相关特性。
- [模型支持列表](./docs/zh/user_guide/model_support_list.md)：MindIE LLM支持的模型。

## 📝贡献声明

1. 提交错误报告：如果您在MindIE LLM中发现了一个不存在安全问题的漏洞，请在MindIE LLm仓库中的Issues搜索，以防该漏洞被重复提交，如果找不到漏洞可以创建一个新的Issues。如果发现了一个安全问题请不要将其公开，请参阅安全问题处理方式。提交错误报告时应包含完整信息。
2. 安全问题处理：本项目中对安全问题处理的形式，请通过邮箱通知项目核心人员确认编辑。
3. 解决现有问题：通过查看仓库的Issues列表可以发现需要处理的问题信息，可以尝试解决其中的某个问题。
4. 如何提出新功能：请使用Issues的Feature标签进行标记，我们会定期处理和确认开发。
5. 开始贡献：
    <br>a. Fork本项目的仓库。
    <br>b. Clone到本地。
    <br>c. 创建开发分支。
    <br>d. 本地自测，提交前请通过所有的单元测试，包括为您要解决的问题新增的单元测试。
    <br>e. 提交代码。
    <br>f. 新建Pull Request。
    <br>g. 代码检视，您需要根据评审意见修改代码，并重新提交更新。此流程可能涉及多轮迭代。
    <br>h. 当您的PR获取足够数量的检视者批准后，Committer会进行最终审核。
    <br>i. 审核和测试通过后，CI会将您的PR合并入到项目的主干分支。

更多贡献相关文档请参见[共享指南](contributing.md)。


## 📝免责声明

版权所有© 2025 MindIE Project.

您对"本文档"的复制、使用、修改及分发受知识共享（Creative Commmons）署名——相同方式共享4.0国际公共许可协议（以下简称"CC BY-SA 4.0"）的约束。为了方便用户理解，您可以通过访问https://creativecommons.org/licenses/by-sa/4.0/了解CC BY-SA 4.0的概要（但不是替代）。CC BY-SA 4.0的完整协议内容您可以访问如下网址获取：https://creativecommons.org/licenses/by-sa/4.0/legalcode。

## 📝相关信息

- [安全声明](./security.md)
- [LICENSE](LICENSE.md)
