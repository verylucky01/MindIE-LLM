# MindIE LLM


## 📢 Latest News

- [2025/12] MindIE LLM 正式宣布开源并面向公众开放！ [会议日历](https://meeting.ascend.osinfra.cn/?sig=sig-MindIE-LLM)



## 🚀 简介

**MindIE LLM**（Mind Inference Engine for Large Language Models）是 MindIE **推理引擎**中的大语言模型（Large Language Model，LLM）推理组件，面向昇腾（Ascend）系列 AI 加速器提供通用的大模型推理能力。它同时提供面向多并发请求的**调度机制**，并支持 Continuous Batching、PagedAttention、FlashDecoding 等**推理加速特性**，以提升推理吞吐并降低端到端时延，满足高性能推理场景需求。

<div align="center">

[![Zread](https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/verylucky01/MindIE-LLM)&nbsp;&nbsp;&nbsp;&nbsp;
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/verylucky01/MindIE-LLM)

</div>

MindIE LLM 主要对外提供 **C++ 与 Python API**（Application Programming Interface），包括大模型推理、并发请求调度和 LLM Manager API 等，便于用户在业务系统中集成与调用。

**MindIE LLM 的整体架构如下图所示：**

<div align="center">

![](./docs/zh/figures/mindie_llm_architecture_diagram.png)

</div>


    - Engine：负责将scheduler，executor，worker等协同串联起来，利用组件间的协同，实现多场景下请求的推理处理能力。
    - Scheduler: 在1个DP域内，将多条请求在Prefill或者Decode阶段组成batch，实现计算和通信的充分利用。
    - Block manager：管理在DP内的kv资源，支持池化后，支持对offload的kv位置感知。
    - Executor：将调度完成的信息分发给Text Generator模块。支持跨机、跨卡的任务下发。

- **Server**：推理服务层，对外提供模型推理的服务化能力与统一接入能力。Endpoint 面向推理服务开发者提供 RESTful 接口，同时，Endpoint 负责推理服务化协议与接口的封装，并兼容 Triton/OpenAI/TGI/vLLM 等主流推理框架的请求接口。

- **LLM Manager**：负责请求状态管理与任务调度。其基于调度策略将用户请求组成 Batch，并通过统一性内存池管理键值缓存（KV Cache）。LLM Manager 汇总并返回推理结果，同时提供状态记录与查询接口。

    - LLM Manager Interface：MindIE-LLM 推理引擎对外暴露的接口层，用于对接上层服务调用与能力集成。
    - Engine：负责对 Scheduler、Executor、Worker 等组件进行编排与串联。通过组件间的协同，Engine 为不同推理场景提供统一的请求处理与执行能力。
    - Scheduler：在一个 DP（Data Parallel，数据并行）域内，将多条请求在 Prefilling（预填充）或 Decoding（解码）阶段组成 Batch。该策略用于提升计算与通信资源的利用率，从而提高整体吞吐与效率。
    - Block Manager：管理 DP 域内的 KV Cache 资源，并支持池化（Pooling）管理以提升内存复用效率。同时，Block Manager 支持对 Offload（卸载到 Host 端或外部存储）的 KV Cache 进行位置感知与索引管理。
    - Executor：将调度阶段生成的执行计划与元信息下发至 Text Generator 模块。Executor 支持分布式推理场景下的任务派发，包括跨机与跨卡执行。

- **Text Generator**：负责模型配置、初始化与加载，并实现自回归推理流程及结果后处理。其向 LLM Manager 提供统一的自回归推理接口，并支持并行解码能力的插件化扩展与运行。

    - Preprocess：将调度后的任务转换为模型可直接消费的输入表示。
    - Generator：对模型运行过程进行抽象封装，覆盖前向计算、状态更新以及自回归式解码等核心执行逻辑。
    - Sampler：基于模型输出的 Logits 完成 Token 选择（如贪心搜索、束搜索、Top-p 采样、基于温度的采样等策略）、停止条件判断，并负责上下文状态的更新与必要的清理（如缓存回收）。

- **Modeling**：提供经过性能调优的算子模块与内置模型实现，支持 ATB Models（Ascend Transformer Boost Models）。

    - 内置模块包括 Attention、Embedding、ColumnLinear、RowLinear、MLP（Multi-Layer Perceptron，多层感知机）与 MoE（Mixture of Experts）。这些模块支持对权重（Weight）进行在线 Tensor 切分与加载。

    - 内置模型基于上述模块完成完整网络构建与组合，并支持 Tensor 切分。同时，内置模型支持多种量化方式。用户也可参考示例，使用内置模块自行构建并定制模型结构。

    - 模型完成组网后将进入编译与优化流程，最终生成可在昇腾 NPU 设备上进行加速推理的可执行计算图。


## 📁 目录结构

```
├── mindie_llm                                     # Python 推理框架主模块
│   ├── connector                                  # 请求接入层
│   ├── text_generator                             # 核心推理引擎
│   │   ├── adapter                                # 后端适配
│   │   ├── plugins                                # 高阶特性插件
│   │   │   ├── prefix_cache                       # Prefix Cache (KV 缓存复用)
│   │   │   ├── splitfuse                          # SplitFuse 
│   │   │   ├── memory_decoding                    # Memory Decoding
│   │   │   ├── la                                 # Lookahead Decoding
│   │   ├── cpp                                    # C++ 扩展（Sampler/Prefix Tree/Memory Bridge）
│   ├── modeling                                   # 模型封装抽象
│   │   ├── model_wrapper/atb                      # ATB 模型后端支持
│   ├── utils                                      # 工具模块：日志/张量/Profiling/验证等
├── examples                                       # 示例代码
│   ├── atb_models                                 # ATB 模型集成与打包入口
│   │   ├── atb_framework                          # ATB 底层框架
│   │   ├── atb_llm                                # 面向 LLM 的 ATB 封装
│   │   ├── examples                               # 运行示例
│   │   ├── scripts                                # 构建脚本
│   │   ├── tests                                  # 单元/集成测试
│   │   ├── setup.py                               # Python 包构建入口
│   │   ├── CMakeLists.txt                         # CMake 构建配置
├── docs                                           # 项目文档介绍
├── src                                            # C++ 核心引擎
│   ├── engine                                     # LLM 引擎的主逻辑（调度/执行）
│   ├── scheduler                                  # 调度器（FCFS/PDDS/Layerwise）
│   ├── block_manager                              # KV Cache 块管理（LRU/Prefix Cache/CoW）
│   ├── llm_manager                                # Python/C++ 桥接 API 
│   ├── server                                     # 服务端（gRPC/HTTP 接入端点）
│   ├── utils                                      # 基础工具（共享内存/加密/日志/ID 生成等）
│   ├── include                                    # 对外头文件接口
├── scripts                                        # 构建与部署脚本
├── tools                                          # 辅助工具
│   ├── llm_manager_python_api_demo                # Python API 使用示例
├── tests                                          # 测试
├── ...                                            
├── CMakeLists.txt                                 # CMake 构建配置                         
├── README.md   
├── requirements.txt                               # Python 安装依赖                               
```


## 📢 版本说明

| MindIE 软件版本&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| CANN 版本兼容性&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|
|:----------------------------|:----------------------------|
| 2.2.RC1 | 8.3.RC2 |


## ⚡️ 环境部署

MindIE LLM 安装前的相关软硬件环境准备，以及安装步骤，请参见[安装指南](./docs/zh/user_guide/installation_guide.md)。


## ⚡️ 快速入门

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



## 📝 学习文档

- [LLM 使用指南](./docs/zh/user_guide/user_manual/README.md)：MindIE LLM 使用指南，包括推理参数配置、在线和离线推理、参数调优等。
- [特性介绍](./docs/zh/user_guide/feature/README.md)：MindIE LLM 支持的推理特性。
- [模型支持列表](./docs/zh/user_guide/model_support_list.md)：MindIE LLM 支持的模型。


## 🚀 贡献声明


## 💖 免责声明

版权所有© 2025-2026 MindIE Project.

您对 "本文档" 的复制、使用、修改及分发受知识共享（Creative Commons，CC）署名 —— 相同方式共享 4.0 国际公共许可协议（以下简称 "CC BY-SA 4.0"）的约束。为了方便用户理解，您可以通过访问 [https://creativecommons.org/licenses/by-sa/4.0/](https://creativecommons.org/licenses/by-sa/4.0/) 了解 CC BY-SA 4.0 的概要（但不是替代）。关于 CC BY-SA 4.0 的完整协议内容，您可以访问如下网址获取：[https://creativecommons.org/licenses/by-sa/4.0/legalcode](https://creativecommons.org/licenses/by-sa/4.0/legalcode)。


## 🌟 相关信息

- [安全声明](./security.md)
- [LICENSE](LICENSE.md)
