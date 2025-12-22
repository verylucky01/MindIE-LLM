# Attention量化

## 简介

此量化方式将q，k，v量化为8bit，通过减少KV Cache的显存占用，优化decode阶段attention算子的速度，提升吞吐。

> [!NOTE]说明 
>- 仅Atlas 800I A2 推理服务器支持Attention量化。
>- 仅支持W8A8配合使用。
>- 仅支持LLaMA3.1-70B。
>- 仅支持和长序列特性、Function Call配合使用。

Attention量化搭配W8A8量化后权重目录结构：

```
├─ config.json
├─ quant_model_weight_w8a8.safetensors
├─ quant_model_description.json
├─ tokenizer_config.json
├─ tokenizer.json
└─ tokenizer.model
```

- 量化输出包含：权重文件quant\_model\_weight\_w8a8.safetensors和权重描述文件quant\_model\_description.json。
- 目录中的其余文件为推理时所需的配置文件，不同模型略有差异。

以下展示了量化后权重描述文件quant\_model\_description.json中的部分内容：

```
{
  "model_quant_type": "W8A8",
  "fa_quant_type": "FAQuant",
  "model.embed_tokens.weight": "FLOAT",
  "model.layers.0.self_attn.q_proj.weight": "W8A8",
  "model.layers.0.self_attn.q_proj.input_scale": "W8A8",
  "model.layers.0.self_attn.q_proj.input_offset": "W8A8",
  "model.layers.0.self_attn.q_proj.quant_bias": "W8A8",
  "model.layers.0.self_attn.q_proj.deq_scale": "W8A8",
  "model.layers.0.self_attn.k_proj.weight": "W8A8",
  "model.layers.0.self_attn.k_proj.input_scale": "W8A8",
  "model.layers.0.self_attn.k_proj.input_offset": "W8A8",
  "model.layers.0.self_attn.k_proj.quant_bias": "W8A8",
  "model.layers.0.self_attn.k_proj.deq_scale": "W8A8",
  "model.layers.0.self_attn.v_proj.weight": "W8A8",
  "model.layers.0.self_attn.v_proj.input_scale": "W8A8",
  "model.layers.0.self_attn.v_proj.input_offset": "W8A8",
  "model.layers.0.self_attn.v_proj.quant_bias": "W8A8",
  "model.layers.0.self_attn.v_proj.deq_scale": "W8A8",
  "model.layers.0.self_attn.o_proj.weight": "W8A8",
  "model.layers.0.self_attn.o_proj.input_scale": "W8A8",
  "model.layers.0.self_attn.o_proj.input_offset": "W8A8",
  "model.layers.0.self_attn.o_proj.quant_bias": "W8A8",
  "model.layers.0.self_attn.o_proj.deq_scale": "W8A8",

}
```

和W8A8量化权重相比，新增fa\_quant\_type描述字段，新增self\_attn字段及下面包含的内容，input\_scale用于将q，k，v特征量化为int8类型，deq\_scale用于将q，k，v输出反量化成浮点类型。

**图 1**  量化权重推理时流程  

![](../../figures/attention_quantization.png)

**表 1**  float16权重量化后dtype及shape信息（假设原始权重的shape为\[n, k\]）

|Tensor信息|dtype|shape|
|--|--|--|
|q_scale|float16|[q_head_num, head_dim]|
|q_offset|float16|[q_head_num, head_dim]|
|k_scale|float16|[kv_head_num, head_dim]|
|k_offset|float16|[kv_head_num, head_dim]|
|v_scale|float16|[kv_head_num, head_dim]|
|v_offset|float16|[kv_head_num, head_dim]|


**表 2**  bfloat16权重量化后dtype及shape信息（假设原始权重的shape为\[n, k\]）

|Tensor信息|dtype|shape|
|--|--|--|
|q_scale|bfloat16|[q_head_num, head_dim]|
|q_offset|bfloat16|[q_head_num, head_dim]|
|k_scale|bfloat16|[kv_head_num, head_dim]|
|k_offset|bfloat16|[kv_head_num, head_dim]|
|v_scale|bfloat16|[kv_head_num, head_dim]|
|v_offset|bfloat16|[kv_head_num, head_dim]|


## 生成权重

您需要修改modeling文件及权重路径下的config.json文件，具体修改方法参见[FA量化使用说明](https://gitcode.com/Ascend/msit/blob/master/msmodelslim/docs/%E5%8A%9F%E8%83%BD%E6%8C%87%E5%8D%97/%E8%84%9A%E6%9C%AC%E9%87%8F%E5%8C%96%E4%B8%8E%E5%85%B6%E4%BB%96%E5%8A%9F%E8%83%BD/pytorch/llm_ptq/FA%E9%87%8F%E5%8C%96%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E.md)。

修改后的config文件为：

```
{
    "architectures": ["LlamaForCausalLM"],     
    // 新增配置     
    // --------------------------------------------------     
    "auto_map": {
    "AutoModelForCausalLM": "modeling_llama_fa3.LlamaForCausalLM"   
},     
    // --------------------------------------------------
    ...
    // 其他未修改的代码部分
    ... 
```

您可以使用以下指令生成W8A8 +Attention量化权重。

```
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd ${ATB_SPEED_HOME_PATH}
python examples/models/llama3/convert_quant_weights.py --model_path {模型权重路径} 
--save_directory {量化模型保存路径} --w_bit 8 --a_bit 8 --device_type npu 
--act_method 3 
--anti_method m3 --disable_level L0 
--calib_file { boolq.jsonl路径}
--use_fa_quant True 
```

-  相比于W8A8的量化方式，需要新增use\_fa\_quant参数。
-  Attention量化权重的quant\_model\_description.json中应包含"fa\_quant\_type": "FAQuant"键值对。

## 执行推理

和W8A8量化权重执行推理的方式相同，请参考[W8A8](w8a8.md)。

