# Buffer Response

该特性主要适用于对大模型推理吞吐和时延有要求，且需要满足SLO（Service Level Objective，服务级别目标）时延指标的场景。

在主流LLM推理系统（如vLLM、TGI）中，对Prefill和Decode阶段的请求调度是分离的，但是分时使用同一份计算资源的。调度策略的选择（优先处理Prefill还是Decode请求）会直接影响吞吐和时延。但是在PD混部场景下，Prefill和Decode阶段的相互干扰可能导致时延波动，难以满足SLO，因此需要更加严格的调度策略和时延控制。

该特性的核心是感知SLO时延，延迟响应避免TTFT（Time to First Token，首Token时延）和TPOT（Time Per Output Token，每个输出Token（不含首token）的延迟）超时。通过配置Prefill阶段和Decode阶段的SLO期望时延，可达到平衡两者时延，使其在都不超时的情况下，收益最大化的目的。

## 限制与约束

- Atlas 800I A2 推理服务器支持此特性。
- Qwen2系列模型支持对接此特性。

## 参数说明

开启Buffer Response特性，需要配置的参数如**表1**。

**表 1**  Buffer Response特性补充参数：**ScheduleConfig的参数**

|配置项|取值类型|取值范围|配置说明|
|--|--|--|--|
|bufferResponseEnabled|bool|truefalse|是否开启Buffer Response特性。true：开启Buffer Response特性。false：不开启Buffer Response特性。选填，默认值：false。|
|prefillExpectedTime|uint32_t|≥1|Prefill阶段Token生成的SLO期望时延。选填，默认值：1500。建议值：按客户SLO时延约束填写。|
|decodeExpectedTime|uint32_t|≥1|Decode阶段Token生成的SLO期望时延。选填，默认值：50。建议值：按客户SLO时延约束填写。|


## 执行推理

1. 打开Server的config.json文件。

    ```
    cd {MindIE安装目录}/latest/mindie-service/
    vi conf/config.json
    ```

2. 配置服务化参数。在Server的config.json文件添加“bufferResponseEnabled”、“prefillExpectedTime”、“decodeExpectedTime”字段（以下加粗部分），参数字段解释请参见[参数说明](#参数说明)，服务化参数说明请参见[配置参数说明（服务化）](../user_manual/service_parameter_configuration.md)章节，参数配置示例如下。

    ```
    "bufferResponseEnabled" : true,
    "prefillExpectedTime" : 1000,
    "decodeExpectedTime" : 50
    ```

3. 启动服务。

    ```
    ./bin/mindieservice_daemon
    ```

4. 本样例以MindIE Benchmark工具展示调优方式，执行如下MindIE Benchmark启动命令，MindIE Benchmark参数详细介绍请参见《MindIE Motor开发指南》中的“MindIE Benchmark \> 输入参数”章节。

    ```
    benchmark \
    --DatasetPath "数据集路径" \
    --DatasetType "gsm8k" \
    --ModelName $model_name \
    --ModelPath $model_path \
    --TestType client \
    --Http https://{ipAddress}:{port} \
    --ManagementHttp https://{managementIpAddress}:{managementPort} \
    --Concurrency 1000 \
    --RequestRate $1 \
    --MaxOutputLen 输出长度 \
    --Tokenizer True
    ```

