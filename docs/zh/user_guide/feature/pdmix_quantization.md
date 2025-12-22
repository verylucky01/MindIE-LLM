# PDMIX量化

## 简介

PDMIX量化是指在模型推理的Prefill阶段和Decode阶段使用不同的量化方式。

**表 1**  PDMIX量化

|量化方式|推理阶段|量化特点|适用场景|
|--|--|--|--|
|W8A8 Pertoken|Prefill|每个token（通常指文本中的单词或子词单元）使用独立的量化参数。|精度要求较高的场景。自然语言处理模型中的激活值量化，能够更好地保留每个token的特征，减少量化带来的信息损失。|
|W8A8 Pertensor|Decode|在整个张量上应用相同的量化参数。|性能要求较高的场景。数值分布较为均匀的张量，实现简单且计算效率高。|


> [!NOTE]说明
> 仅支持LLama3.1-70B、Qwen2.5-14B、Qwen2.5-72B、Qwen3-14B和Qwen3-32B。

量化后权重目录结构：

```
├─ config.json
├─ quant_model_weight_w8a8_mix.safetensors
├─ quant_model_description.json
├─ tokenizer_config.json
├─ tokenizer.json
└─ tokenizer.model
```

-  量化输出包含：权重文件quant\_model\_weight\_w8a8\_mix.safetensors和权重描述文件quant\_model\_description.json。
-  目录中的其余文件为推理时所需的配置文件，不同模型略有差异。

以下展示了量化后权重描述文件quant\_model\_description.json中的部分内容：

```
{
"model_quant_type": "W8A8_MIX",
  "model.embed_tokens.weight": "FLOAT",
  "model.layers.0.self_attn.q_proj.weight": "W8A8_MIX",
  "model.layers.0.self_attn.q_proj.weight_scale": "W8A8_MIX",
  "model.layers.0.self_attn.q_proj.weight_offset": "W8A8_MIX",
  "model.layers.0.self_attn.q_proj.input_scale": "W8A8_MIX",
  "model.layers.0.self_attn.q_proj.input_offset": "W8A8_MIX",
  "model.layers.0.self_attn.q_proj.quant_bias": "W8A8_MIX",
  "model.layers.0.self_attn.q_proj.deq_scale": "W8A8_MIX",
}
```

与W8A8量化权重相比，新增weight\_scale和weight\_offset，用于对Matmul的计算结果进行反量化。

量化权重推理流程同W8A8量化。

此量化方式支持量化bfloat16类型的原始权重。

**表 2**  bfloat16权重量化后dtype及shape信息（假设原始权重的shape为\[n, k\]）

|Tensor信息|weight|weight_scale|weight_offset|bias|input_scale|input_offset|quant_bias|deq_scale|
|--|--|--|--|--|--|--|--|--|
|dtype|int8|bf16|bf16|bf16|bfloat16|bfloat16|int32|float32|
|shape|[n, k]|[n, 1]|[n, 1]|[n]|[1]|[1]|[n]|[n]|


> [!NOTE]说明 
> 仅当浮点权重存在bias场景时，量化权重才会有bias。

## 生成权重

以Qwen3-32B为例。

1. 进入msModelSlim工具，修改如下文件：$\{MsModelSlim工具安装位置\}/msmodelslim/practice\_lab/Qwen/qwen3-dense-w8a8.yaml。

    ```
    ...
    calib_params:
        disable_level: L10
    ...
    ```

2. 使用以下指令生成W8A8量化权重。

    ```
    msmodelslim quant --model_path {浮点权重路径} --save_path {pdmix量化权重路径} --device npu --model_type Qwen3-32B --config_path ./msmodelslim/practice_lab/Qwen/qwen3-dense-w8a8.yaml --trust_remote_code True
    ```

-  以上指令展示了生成Qwen3-32B pdmix权重的最优参数配置，不同模型的参数配置不同，请参考模型Readme文件。

## 执行推理

以Qwen3-32B为例，您可以使用以下指令执行对话测试，推理内容为"What's deep learning?"，最长输出20个token。

```
cd ${ATB_SPEED_HOME_PATH}
torchrun --nproc_per_node 2 --master_port 12350 -m examples.run_pa --model_path {pdmix量化权重路径}
```

