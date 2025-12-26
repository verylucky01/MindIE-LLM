# 特性列表

MindIE LLM支持的特性包括基础特性、长序列特性、调度特性、加速特性和交互特性。

## 基础特性

基础特性如[表1](#table1)所示。

**表 1**  基础特性介绍 <a id="table1"></a>

|特性|说明|
|----|----|
|量化|通过降低模型数值精度，从而减小模型体积、提升推理速度并降低能耗。其特性介绍详情请参见量化。|
|多模态理解|多模态理解模型是指能够处理和理解包括多种模态数据的深度学习模型。其特性介绍详情请参见[多模态理解](./multimodel_understanding.md)。|
|Multi-Lora|使用Multi-LoRA来执行基础模型和不同的LoRA权重进行推理。其特性介绍详情请参见[Multi-LoRA](multi_loRA.md)。|
|MoE|通过引入稀疏激活的专家网络，在不显著增加计算成本的前提下大幅扩展模型参数规模，从而提升模型能力。其特性介绍详情请参见[MoE](./moe.md)。|
|负载均衡|降低NPU卡间的不均衡度，从而提升模型推理的性能。其特性介绍详情请参见[负载均衡](./load_balancing.md)。|
|共享专家外置|将共享专家独立部署在单独的NPU卡上，与路由专家/冗余专家分离。其特性介绍详情请参见[共享专家外置](./externalized_shared_expert.md)。|
|MLA|利用低秩键值联合压缩来消除推理时键值缓存的瓶颈，从而支持高效推理。其特性介绍详情请参见[MLA](./mla.md)。|
|Expert Parallel|通过将专家分别部署在不同的设备上，实现专家级别的并行计算。其特性介绍详情请参见[Expert Parallel](./expert_parallel.md)。|
|Data Parallel|将推理请求划分为多个批次，并将每个批次分配给不同的设备进行并行处理。其特性介绍详情请参见[Data Parallel](./data_parallel.md)。|
|Tensor Parallel|通过将张量（如权重矩阵、激活值等）在多个设备（如NPU）之间进行切分 ，从而实现模型的分布式推理。其特性介绍详情请参见[Tensor Parallel](./tensor_parallel.md)。|


## 长序列特性

长序列特性如[表2](#table2)所示。

**表 2**  长序列特性介绍  <a id="table2"></a>

|特性|说明|
|----|----|
|Context Parallel|通过将长序列在上下文维度进行切分，分配到不同设备并行处理，减少首token响应时间，其特性介绍详情请参见[Context Parallel](./context_parallel.md)。|
|Sequence Parallel|通过对KV Cache进行切分，使得每个sprank保存的KV Cache各不相同，达到节省显存，支持长序列的功能，其特性介绍详情请参见[Sequence Parallel](./sequence_parallel.md)。|


## 调度特性

调度特性如[表3](#table3)所示。

**表 3**  调度特性介绍  <a id="table3"></a>

|特性|说明|
|----|----|
|异步调度|对于maxBatchSize较大，且输入输出长度比较长的场景，该特性使用模型推理阶段的耗时掩盖数据准备阶段和数据返回阶段的耗时，避免NPU计算资源和显存资源浪费。其特性介绍详情请参见[异步调度](./asynchronous_scheduling.md)。|
|PD分离|将模型推理的Prefill阶段和Decode阶段，分别实例化部署在不同的机器资源上同时进行推理。其特性介绍详情请参见[PD分离](./prefill_decode_disaggregation.md)。|
|SplitFuse|将长提示词分解成更小的块，并在多个forward step中进行调度，降低Prefill时延。其特性介绍详情请参见[SplitFuse](./splitFuse.md)。|
|SLO调度优化|为应对客户端的高并发请求，在确保SLO的前提下提升系统吞吐量。其特性介绍详情请参见[SLO调度优化](./slo_aware_scheduling_optimization.md)。|


## 加速特性

加速特性如[表4](#table4)所示。

**表 4**  加速特性介绍 <a id="table4"></a>


|特性|说明|
|----|----|
|Micro Batch|批处理过程中，将数据切分为更小粒度的多个batch运行，使得硬件资源得以充分利用，以提高推理吞吐。其特性介绍详情请参见[Micro Batch](./micro_batch.md)。|
|Buffer Response|通过配置Prefill阶段和Decode阶段的SLO期望时延，可达到平衡两者时延，使其在都不超时的情况下，收益最大化的目的。其特性介绍详情请参见[Buffer Response](./buffer_response.md)。|
|并行解码|利用算力优势弥补访存带宽受限的影响，提升算力利用率。其特性介绍详情请参见[并行解码](./parallel_decoding.md)。|
|MTP|在推理过程中，模型不仅预测下一个token，而且会同时预测多个token，从而显著提升模型生成速度。其特性介绍详情请参见[MTP](./mtp.md)。|
|Prefix Cache|复用跨session的重复token序列对应的KV Cache，减少一部分前缀token的KV Cache计算时间，从而减少Prefill的时间。其特性介绍详情请参见[Prefix Cache](./prefix_cache.md)。|
|KV Cache池化|支持将DRAM甚至SSD等更大容量的存储介质纳入前缀缓存池，从而突破片上内存的容量限制。该特性有效提升了Prefix Cache的命中率，显著降低大模型推理的成本。其特性介绍详情请参见[KV Cache池化](./kv_cache_pooling.md)。|


## 交互特性

交互特性如[表5](#table5)所示。

**表 5**  交互特性介绍  <a id="table5"></a>

|特性|说明|
|----|----|
|Function Call|支持Function Call函数调用，使大模型具备使用工具能力。其特性介绍详情请参见[Function Call](./function_call.md)。|
|思考解析|对大模型的输出内容进行结构化解析，将思考过程和输出结果进行分离。其特性介绍详情请参见[思考解析](./enable_reasoning.md)。|


