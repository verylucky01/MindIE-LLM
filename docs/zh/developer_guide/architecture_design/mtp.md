# mtp综述

MTP（Multi-Token Prediction，多Token预测）是DeepSeek中提出的一种用于单次生成多个token的并行解码方法。 MTP并行解码的核心思想是在推理过程中会同时预测多个token，从而显著提升模型生成速度。

原始论文详见：https://arxiv.org/pdf/2404.19737

MTP推理的流程简图如下所示（以MTP=1为例）：

<img src="../../figures/mtp_instruction.png" alt="mtp_instruction" width="1100"/>

先主模型推理，输入的token是t1到tN，经过一轮推理之后，可以得到1个输出token tN+1，同时输出最后一层的hiddenstates。之后进行MTP层推理，MTP层的输入token是将主模型的prefilltokens进行roll操作，从t2开始输入，最后拼接上主模型输出的token tN+1。经过1轮推理得到草稿token tN+2。获得草稿token tN+2之后，我们将上一轮主模型输出的token tN+1和草稿token tN+2拼接，一起输入给主模型进行推理，得到token tN+2和token tN+3.  之后继续将输入token tN+1和草稿token tN+2时的最后一层的hiddenstates和token tN+2和token tN+3输入给MTP层，得到新的草稿token，往后依次类推。

# 使能方法

在服务化的config.json文件中的ModelDeployConfig中的ModelConfig里新增如下字段(以MTP=1为例)

```
"plugin_params": "{\"plugin_type\":\"mtp\",\"num_speculative_tokens\": 1}"
```

其中num_speculative_tokens表示MTP开启时每一轮猜测草稿token个数。

# 推理流程示例

以MTP=2为例，给出每一轮推理的具体流程

## prefill阶段

```
|---------------|--------------|----------------------|----------------------|----------------------|----------------------|----------------------|
|target model   | input        | input_ids            | A                    | B                    | C                    | D                    |
|               |              | slot                 | 0                    | 1                    | 2                    | 3                    |
|               |              | position             | 0                    | 1                    | 2                    | 3                    |
|               |              | context length       | 4                    |                      |                      |                      |
|               |              | lm head indices      | 3                    |                      |                      |                      |
|               |--------------|----------------------|----------------------|----------------------|----------------------|----------------------|
|               | output       | output_tokenid       | E                    |                      |                      |                      |
|               |              | output_hiddenstates  | hiddenstates(ABCD)   |                      |                      |                      |
|---------------|--------------|----------------------|----------------------|----------------------|----------------------|----------------------|
|mtp            | input        | input_ids            | B                    | C                    | D                    | E                    |
|               |              | slot                 | 0                    | 1                    | 2                    | 3                    |
|               |              | position             | 0                    | 1                    | 2                    | 3                    |
|               |              | context length       | 4                    |                      |                      |                      |
|               |              | lm head indices      | 3                    |                      |                      |                      |
|               |              | hiddenstates         | hiddenstates(ABCD)   |                      |                      |                      |
|               |--------------|----------------------|----------------------|----------------------|----------------------|----------------------|
|               | output       | output_tokenid       | (ignore)             |                      |                      |                      |
|               |              | output_hiddenstates  | (ignore)             |                      |                      |                      |
|---------------|--------------|----------------------|----------------------|----------------------|----------------------|----------------------|
|                              | final output tokens  | E                    |                      |                      |                      |
|                              | savehiddenstates     | hiddenstates(D)      |                      |                      |                      |
|---------------|--------------|----------------------|----------------------|----------------------|----------------------|----------------------|
```

Prefill阶段的分别在主模型和MTP层（后续称小模型）进行prefill处理。
在主模型对输入的prompt ABCD完成推理后，会输出以ABCD为输入计算得到的中间结果hiddenstates(ABCD)。hiddenstates(ABCD)会作为小模型的输入。
同时，prefill的输出token E，会拼接成BCDE作为小模型的prompt进行一次prefill推理。以token E来说，mtp层输入了tokenE和对应的D的hiddenstates进行推理。至此，主模型和小模型均已包含完整的prompt的kvcache。

为了后续的decode的流程能够归一，我们先将这一轮输出的草稿token丢弃了，在decode阶段我们可以重新获取这个token

## decode阶段

### 第一次decode

```
|---------------|--------------|----------------------|----------------------|----------------------|----------------------|
|mtp1           | input        | input_ids            | E                    | 0                    | 0                    |
|               |              | slot                 | 3                    | 4                    | 5                    |
|               |              | position             | 4                    | 5                    | 6                    |
|               |              | context length       | 6                    |                      |                      |
|               |              | lm head indices      | 0                    |                      |                      |
|               |              | hiddenstates         | hiddenstates(D00)    |                      |                      |
|               |--------------|----------------------|----------------------|----------------------|----------------------|
|               | output       | output_tokenid       | f                    |                      |                      |
|               |              | output_hiddenstates  | hiddenstates_mtp(Exx)|                      |                      |
|---------------|--------------|----------------------|----------------------|----------------------|----------------------|
|mtp2           | input        | input_ids            | f                    | 0                    | 0                    |
|               |              | slot                 | 4                    | 5                    | 6                    |
|               |              | position             | 5                    | 6                    | 7                    |
|               |              | context length       | 7                    |                      |                      |
|               |              | lm head indices      | 0                    |                      |                      |
|               |              | hiddenstates         | hiddenstates_mtp(Exx)|                      |                      |
|               |--------------|----------------------|----------------------|----------------------|----------------------|
|               | output       | output_tokenid       | g                    |                      |                      |
|               |              | output_hiddenstates  | (ignore)             |                      |                      |
|---------------|--------------|----------------------|----------------------|----------------------|----------------------|
|target model   | input        | input_ids            | E                    | f                    | g                    |
|               |              | slot                 | 4                    | 5                    | 6                    |
|               |              | position             | 4                    | 5                    | 6                    |
|               |              | context length       | 7                    |                      |                      |
|               |              | lm head indices      | 0                    | 1                    | 2                    |
|               |--------------|----------------------|----------------------|----------------------|----------------------|
|               | output       | output_tokenid       | F                    | x                    | x                    |
|               |              | output_hiddenstates  | hiddenstates(Efg)    |                      |                      |
|               |--------------|----------------------|----------------------|----------------------|----------------------|
|               | verify       | accept tokens        | F                    |                      |                      |
|               | 校验未命中    | savehiddenstates     | hiddenstates(E)      |                      |                      |
|---------------|--------------|----------------------|----------------------|----------------------|----------------------|
```

Decode阶段的推理流程类似大小模型投机推理，先进行小模型推理，再进行大模型推理：
① 小模型（MTP层）推理输出草稿token
② 草稿token拼接给大模型输入得到推理结果
③ 校验verify操作进行token-by-token的比对，判断可接受的token数

对于第一次decode，将prefill输出的token E作为小模型的输入，这里为了处理方便，会将小模型的输入长度pad到```num_speculative_tokens + 1```个，hiddenstates使用的是prefill最后一个token D的hiddenstates，并pad到```num_speculative_tokens + 1```的shape，多轮MTP shape一致。

【说明】PD分离的场景，D节点无法获取到正确的P输出的hiddenstates，这里会用全0代替。而第一轮MTP需要的kv cache在P节点已经计算好并pull kv到了D节点，因此第一次decode中的第一层mtp不需要save kv cache。当前实现使用存到dummy blocktable的方式实现，保证正确的kvcache不被污染

### 非第一次decode

```
|---------------|--------------|----------------------|----------------------|----------------------|----------------------|
|mtp1           | input        | input_ids            | F                    | 0                    | 0                    |
|               |              | slot                 | 4                    | 5                    | 6                    |
|               |              | position             | 5                    | 6                    | 7                    |
|               |              | context length       | 7                    |                      |                      |
|               |              | lm head indices      | 0                    |                      |                      |
|               |              | hiddenstates         | hiddenstates(E00)    |                      |                      |
|               |--------------|----------------------|----------------------|----------------------|----------------------|
|               | output       | output_tokenid       | G                    |                      |                      |
|               |              | output_hiddenstates  | hiddenstates_mtp(Fxx)|                      |                      |
|---------------|--------------|----------------------|----------------------|----------------------|----------------------|
|mtp2           | input        | input_ids            | G                    | 0                    | 0                    |
|               |              | slot                 | 5                    | 6                    | 7                    |
|               |              | position             | 6                    | 7                    | 8                    |
|               |              | context length       | 8                    |                      |                      |
|               |              | lm head indices      | 0                    |                      |                      |
|               |              | hiddenstates         | hiddenstates_mtp(Fxx)|                      |                      |
|               |--------------|----------------------|----------------------|----------------------|----------------------|
|               | output       | output_tokenid       | H                    |                      |                      |
|               |              | output_hiddenstates  | (ignore)             |                      |                      |
|---------------|--------------|----------------------|----------------------|----------------------|----------------------|
|target model   | input        | input_ids            | F                    | G                    | H                    |
|               |              | slot                 | 5                    | 6                    | 7                    |
|               |              | position             | 5                    | 6                    | 7                    |
|               |              | context length       | 8                    |                      |                      |
|               |              | lm head indices      | 0                    | 1                    | 2                    |
|               |--------------|----------------------|----------------------|----------------------|----------------------|
|               | output       | output_tokenid       | I                    | x                    | x                    |
|               |              | output_hiddenstates  | hiddenstates(FGH)    |                      |                      |
|               |--------------|----------------------|----------------------|----------------------|----------------------|
|               | verify       | accept tokens        | I                    |                      |                      |
|               | 校验全命中    | savehiddenstates     | hiddenstates(FGH)    |                      |                      |
|---------------|--------------|----------------------|----------------------|----------------------|----------------------|
```

对于后续的decde，第一层MTP层的输入即为上一轮decode中主模型的输出pad到```num_speculative_tokens + 1```得到，hidden states类似。每一轮MTP层的输入长度都是```num_speculative_tokens + 1```，为复用lm_head_indice的值，每轮MTP的输入均为上一轮MTP的输入向左roll一位后，将上一轮MTP输出的token换到lm_head_indice的位置上。slots、position_id做相应的更新。
主模型则是将MTP层输出的多个草稿token拼入输入的input_ids中。因此无论MTP层还是主模型，输入的shape是一致的，每个bs的长度都是```num_speculative_tokens + 1```。

### token-by-token比对的verify方法

verify的目的是保证在开启和关闭MTP时能保证精度完全无损，即开启MTP时和自回归的输出一致。

按自回归的推理，如下图所示，由token D推理得到E，由E推理得到F，以此类推。

<img src="../../figures/mtp_autoregressive.png" alt="verify" width="400"/>

对于MTP开启的场景，如下图所示，需要比对草稿E和由D输出的自回归的tokenE是否是相同的。如果相同，就意味着由草稿E得到的tokenF也是正确的。反之，如果不相等，说明由这个草稿token e推理得到的f就是错误的，所以不可以接收这个token。

<img src="../../figures/mtp_verify.png" alt="verify" width="1000"/>

# 代码调用流程图示

以集中式、同步调度场景为例（分布式省去generator torch中的dp切分和padding计算），给出MindIE_LLM仓中MTP的代码运行流程。

<img src="../../figures/mtp.jpg" alt="mtp" width="1500"/>

# 模块间输入输出参数汇总

## plugin_manager与generator_torch

### 输入参数

#### prefill阶段

model_inputs（host，np数组）：主模型的输入

【说明】prefill阶段只需要构造主模型的输入，草稿模型的输入除input_ids外与主模型完全一致，因此在flash_causal_deepseekv2.py文件中，在主模型执行完成后，更新input_ids，草稿模型可直接复用主模型输入，无需额外构造。

#### decode阶段

1. model_inputs（host，np数组）：主模型的输入，其中每个batchsize固定输入```num_speculative_tokens + 1```个token，input_ids中空出草稿token的位置。
2. sub_model_inputs（host，np数组）：MTP层的输入，结构与model_inputs一致。MTP>1时，也只有一份sub_model_inputs，多轮MTP之间的参数更新承载在flash_causal_deepseekv2文件中。
3. q_lens（List）：表示每个bs的输入token个数，当前场景固定由bs个```num_speculative_tokens + 1```的值构成的1维List
4. hidden_states（host，torch.tensor）：上一轮主模型输出的hidden_states，shape为```[(num_speculative_tokens + 1) *bs，hidden_size]```

[说明] 当MTP>1时，对于小模型来说，每增加一层MTP推理，就需要增加一个slot存放新增的kv cache，因此在sub_model_inputs中的slots的个数是有可能大于输入input_ids的个数的。slots的shape为```[1，bs * mtp * 2]```。之后在flash_causal_deepseekv2.py文件中，会将slots切分成每一层推理时需要的slots，最终每一层mtp拿到的slots的shape依旧是```[1，bs * (num_speculative_tokens + 1)]```


### 输出参数

此处的输出参数是从flash_causal_deepseekv2.py中透传得到的。

#### prefill阶段

1. logits（device，torch.tensor）：主模型的输出
2. hidden_states（device，torch.tensor）：主模型最后一层的hidden_states，此处只输出最后一个token的hiddenstates，shape为```[1 *bs，hidden_size]```

#### decode阶段

1. logits（device，torch.tensor）：主模型的输出
2. hidden_states（device，torch.tensor）：主模型最后一层的hidden_states，shape为```[(num_speculative_tokens + 1) *bs，hidden_size]```
3. draft_tokens（device，torch.tensor）：格式为 ```[batch0 (draft_token0~num_speculative_tokens-1) batch1 ...batch2...]```

## generator_torch 与 flash_causal_deepseekv2

### 输入参数

1. sub_model_inputs（device，torch.tensor）：小模型model_inputs，仅decode阶段存在
2. lm_head_local_dp：仅集中式有，MTP>1时使用，用于每一轮小模型输入token更新的indice位置
3. q_lens：1维list，表示每个bs的输入token个数，集中式时此处是经过dp切分的qlen，分布式时是plugin_manager透传得到

### 输出参数

同plugin_manager与generator_torch数据接口的输出参数。

# 代码实现

MTP流程中的主要新增代码承载在以下文件中：

1. [mtp_plugin.py](../../../../mindie_llm/text_generator/plugins/mtp/mtp_plugin.py) & [decoding_policy.py](../../../../mindie_llm/text_generator/plugins/mtp/decoding_policy.py)

   [mtp_plugin.py](../../../../mindie_llm/text_generator/plugins/mtp/mtp_plugin.py)文件由[plugin_manager.py](../../../../mindie_llm/text_generator/plugins/plugin_manager.py)文件调用，在主调度流程中，调用mtp的以下功能：

   ① 基于自回归时的基础模型输入，构造mtp场景下主模型和草稿模型需要的模型输入；

   ② 采样参数构造；

   ③ mtp场景每轮decode需要缓存的信息管理；

   ④ 对草稿token的校验处理（当前仅支持token比对）；

   ⑤ 叠加异步调度时的模型输入参数更新。

2. [flash_causal_deepseekv2.py](../../../../examples/atb_models/atb_llm/models/deepseekv2/flash_causal_deepseekv2.py)

   flash_causal_deepseekv2.py的入口函数是 forward()方法，调用顺序：

   [plugin_manager.py](../../../../mindie_llm/text_generator/plugins/plugin_manager.py)（generate_token() or generate_token_async()）--> [generator_torch.py](../../../../mindie_llm/text_generator/adapter/generator_torch.py)（forward()） --> [atb_model_wrapper.py](../../../../mindie_llm/modeling/model_wrapper/atb/atb_model_wrapper.py) （forward()）--> [flash_causal_deepseekv2.py](../../../../examples/atb_models/atb_llm/models/deepseekv2/flash_causal_deepseekv2.py)（forward()）

   ① mtp prefill和decode的入口函数；

   ② 承载多轮草稿模型之间、草稿模型与主模型之间的模型输入更新。

   ③ mtp的权重加载（初始化时调用）

3. [mtp_decoder_model.cpp](../../../../examples/atb_models/atb_framework/models/deepseekv2/model/mtp_decoder_model.cpp)

   承载草稿模型的组图