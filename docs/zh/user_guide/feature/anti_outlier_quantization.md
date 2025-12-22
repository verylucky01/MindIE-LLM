# Anti-Outlier离群值处理

## 简介

离群值处理旨在解决在模型量化过程中由于数据分布异常导致的量化精度损失问题。在大模型量化中，离群值可能会导致量化后的模型性能下降，因为量化过程需要将连续的浮点数值转换为离散的整数值，而离群值可能会超出预期的量化范围。通过离群值抑制的技术手段可以减少或消除离群值对量化模型性能的影响，确保量化后的模型具有良好的精度表现。

> [!NOTE]说明 
> Anti-Outlier量化可以配合其他量化方式一起使用。当前支持：W8A8 + Anti-Outlier，W8A16 + Anti-Outlier和W8A8SC + Anti-Outlier。

以下展示了W8A8 + Anti-Outlier量化后权重描述文件quant\_model\_description.json中的部分内容：

```
{
  "model_quant_type": "W8A8",
  "model.embed_tokens.weight": "FLOAT",
  "model.layers.0.input_layernorm.weight": "FLOAT",
  "model.layers.0.input_layernorm.module.weight": "W8A8",
  "model.layers.0.input_layernorm.module.bias": "W8A8"
}
```

新增input\_layernorm.module.weight和input\_layernorm.module.bias权重，用于对激活值进行离群值处理。

**图 1**  量化权重推理时流程<a name="fig1591891567"></a>  
![](../../figures/anti_outlier_quantization.png "量化权重推理时流程-5")

**表 1**  权重量化后dtype及shape信息（假设原始权重的shape为\[n\]）

|Tensor信息|input_layernorm.module.weight|input_layernorm.module.bias|
|--|--|--|
|dtype|float32|float32|
|shape|[n]|[n]|


## 生成权重<a name="section14202946115415"></a>

以LLaMA3.1-8B为例，您可以使用以下指令生成W8A8量化权重。

```
cd ${ATB_SPEED_HOME_PATH}
python examples/convert/model_slim/quantifier.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file $ATB_SPEED_HOME_PATH/examples/convert/model_slim/boolq.jsonl --disable_names $disable_names --device_type npu --disable_level L0 --anti_method m1 --act_method 1 --tokenizer_args {\"padding_side\":\"left\",\"pad_token\":\"<unk>\"}
```

-   相比于W8A8的量化方式，需要新增anti\_method参数。
-   不同模型对anti\_method参数的配置不同，请参考模型Readme文件。

## 执行推理<a name="section1788515529541"></a>

以LLaMA3.1-8B为例，您可以使用以下指令执行对话测试，推理内容为"What's deep learning?"。

```
cd ${ATB_SPEED_HOME_PATH}
bash examples/models/llama/run_pa.sh {W8A8量化权重路径}
```

