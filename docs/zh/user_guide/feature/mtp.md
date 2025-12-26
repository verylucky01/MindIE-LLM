# MTP

MTP（Multi-Token Prediction，多Token预测）是DeepSeek中提出的一种用于单次生成多个token的并行解码方法。 MTP并行解码的核心思想是在推理过程中，模型不仅预测下一个token，而且会同时预测多个token，从而显著提升模型生成速度。

## 限制与约束

-  Atlas 800I A2 推理服务器和Atlas 800I A3 超节点服务器支持此特性。
-  当前仅DeepSeek-R1和DeepSeek-V3的W8A8量化模型、KV Cache int8量化模型支持此特性。
-  该特性不能和并行解码、Multi-LoRA、SplitFuse、长序列特性同时使用。
-  该特性暂不支持n、best\_of、use\_beam\_search、logprobs等与多序列推理相关的后处理参数。
-  MTP惩罚类后处理仅支持重复惩罚。

## 参数说明

开启MTP特性，需要配置的参数如[表1](#table1)所示。

**表 1**  MTP特性补充参数：**ModelDeployConfig中的ModelConfig参数**  <a id="table1"></a>

|配置项|取值类型|取值范围|配置说明|
|--|--|--|--|
|plugin_params|std::string|plugin_type: mtpnum_speculative_tokens: [1]|**plugin_type**设置为“mtp”，表示选择mtp特性。**num_speculative_tokens**表示MTP的层数，可设置为1或2。不需要生效任何插件功能时，请删除该配置项字段。配置示例：{\"plugin_type\":\"mtp\",\"num_speculative_tokens\": 1}【注】num_speculative_tokens配置建议：对于低时延场景，可配置使用1或2，对于高吞吐场景，建议配置不超过1|


## 执行推理

1. 打开Server的config.json文件。

    ```bash
    cd {MindIE安装目录}/latest/mindie-service/
    vi conf/config.json
    ```

2. 配置服务化参数。在Server的config.json文件添加“plugin\_params“字段（以下加粗部分），参数字段解释请参见[表1](#table1)，服务化参数说明请参见[配置参数说明（服务化）](../user_manual/service_parameter_configuration.md)章节，参数配置示例如下。

    ```json
    "ModelDeployConfig" :
    {
       "maxSeqLen" : 2560,
       "maxInputTokenLen" : 2048,
       "truncation" : false,
       "ModelConfig" : [
         {
             "plugin_params": "{\"plugin_type\":\"mtp\",\"num_speculative_tokens\": 1}",
             "modelInstanceType" : "Standard",
             "modelName" : "DeepSeek-R1_w8a8",
             "modelWeightPath" : "/data/weights/DeepSeek-R1_w8a8",
             "worldSize" : 8,
             "cpuMemSize" : 5,
             "npuMemSize" : -1,
             "backendType" : "atb",
             "trustRemoteCode" : false
          }
       ]
    },
    ```

3.  启动服务。

    ```
    ./bin/mindieservice_daemon
    ```

