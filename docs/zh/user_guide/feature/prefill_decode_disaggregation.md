# PD分离

Transformer大模型推理PD分离部署特性，主要是指模型推理的Prefill阶段和Decode阶段分别实例化部署在不同的机器资源上同时进行推理，其结合Prefill阶段的计算密集型特性，以及Decode阶段的访存密集型特性，通过调节PD节点数量配比来提升Decode节点的batch size来充分发挥NPU卡的算力，进而提升集群整体吞吐。此外，在Decode平均低时延约束场景，PD分离相比PD混合部署，更加能够发挥性能优势。

PD分离分为单机PD分离和多机PD分离。

MindIE LLM提供PD分离部署所需要的关键能力，包括PD按角色实例化，PD KV Cache高性能传输，计算传输并行，batch调度。

## 限制与约束

-  多机PD分离：
    -  Atlas 800I A2 推理服务器和Atlas 800I A3 超节点服务器支持此特性。
    -  不同P、D节点使用的NPU卡数量必须相同。
    -  NPU网口互联（带宽：200Gbps）。
    -  Server服务化支持PD分离，包括重计算、集群调度等特性。
    -  不支持和Multi-LoRA、并行解码、SplitFuse、Prefix Cache、多机推理特性同时使用。
    -  该特性暂不支持n、best\_of、use\_beam\_search、logprobs、top\_logprobs等与多序列推理相关的后处理参数。
    -  LLaMA3系列、Qwen2系列、Qwen3系列和DeepSeek系列模型支持该特性。
    -  不支持和稀疏量化、KV Cache int8量化配合使用。

-  单机PD分离：

    -  Atlas 800I A2 推理服务器和Atlas 800I A3 超节点服务器支持此特性。
    -  不同P、D节点使用的NPU卡数量必须相同。
    -  LLaMA3-8B、Qwen2.5-7B和Qwen3-8B支持此特性。
    -  支持与Prefix Cache特性同时使用。
    -  不支持和稀疏量化、KV Cache int8量化配合使用。
    -  该特性暂不支持n、best\_of、use\_beam\_search、logprobs、top\_logprobs等与多序列推理相关的后处理参数。

    **表 1**  依赖部件说明

|组件|用途说明|
|--|--|
|MindIE MS|MindIE MS主要负责P、D实例的生命周期管理、状态采集、请求调度等。逻辑上包含2部分：Controller和Coordinator。其中Controller主要负责P、D实例生命周期管理，Coordinator主要负责P、D请求的调度。|
|MindIE Motor|通过endpoint方式接收Coordinator推理请求。|
|MindIE LLM BatchScheduler|调度batch能力，单独调度prefill或decode类型的请求并下发batch。|
|MindIE LLM|提供基础的模型执行能力。|
|CANN KV库|提供基于RDMA的KV Cache传输能力。|


## 参数说明

**表 2**  ServerConfig补充参数

|配置项|取值类型|取值范围|配置说明|
|--|--|--|--|
|InferMode|std::string|"dmi"或者"standard"|dmi为PD分离模式，该模式下服务化和模型启动解耦。待下发P/D身份后才拉起模型。standard为非PD分离模式，服务化和模型启动不解耦。必填，默认值："standard"。|


## 接口说明

请参见**SetReqType接口**~**GetSrcBlockTable**接口章节。

## 执行推理

此特性需要配合MindIE Motor使用，请参考《MindIE Motor开发指南》中的“集群服务部署 \> PD分离服务部署”章节。

